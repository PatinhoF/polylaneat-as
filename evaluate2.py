import os
import cv2
import json
import time
import torch
import numpy as np
from typing import List, Dict
from pycocotools import mask as maskUtils

from lib.config import Config

# -----------------------------
# Paths / settings (edit here)
# -----------------------------
CFG_PATH   = "cfgs/coco_lanes.yaml" # change according to the model
CKPT_PATH  = "model_150_aug0.5_v2.pt"
#CKPT_PATH  = "experiments/v2/models/model_150.pt"
TEST_ROOT = "datasets/test"   # evaluate each subdirectory independently
CONFIDENCE = 0.25                  # override model conf threshold

# Lane rendering thickness (pixels) for predicted lane polyline -> mask
PRED_LINE_THICKNESS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PolyLaneNet normalizes inputs with ImageNet statistics when
# `datasets.test.parameters.normalize` is true in the YAML config.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Number of points sampled along each predicted polynomial when turning it
# into a polyline (matches PolyLaneNet's own visualization code).
POLY_SAMPLE_POINTS = 100

# Images with at least this many GT lane annotations are treated as
# "crossing" images for the stratified metrics breakdown; everything below
# is treated as a "normal" image.
CROSSING_LANE_COUNT = 6

# ---- Lane deduplication ----
# Two decoded lanes closer than this (normalized x distance, evaluated at
# their shared y-range midpoint) are treated as duplicate detections of the
# same physical lane; the lower-confidence one is suppressed. This is a
# post-decode step and requires no retraining.
DEDUP_X_THRESHOLD = 0.05  # fraction of image width; tune as needed

# ---- Confidence / thickness sweep mode ----
# When RUN_SWEEP is True, main() runs inference once per image (cached), then
# re-thresholds/re-renders masks for every (confidence, thickness) pair below
# and prints a comparison table -- no re-running the model per combo.
RUN_SWEEP = False
CONFIDENCE_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
THICKNESS_SWEEP = [6, 10, 14, 20]


def dedup_lane_rows(lane_rows: np.ndarray, x_threshold: float = DEDUP_X_THRESHOLD) -> np.ndarray:
    """
    Given a (num_lanes, 7) array of decoded lane rows
    [conf, y_start, y_end, c0, c1, c2, c3], suppress lanes that are
    near-duplicates of a higher-confidence lane (i.e. predicted to pass
    through nearly the same point at a shared y-value).

    Two lanes are compared at the midpoint of their overlapping
    [y_start, y_end] range. If their predicted x at that y differs by less
    than `x_threshold` (normalized, i.e. fraction of image width), the
    lower-confidence lane is suppressed by zeroing its confidence -- this
    lets downstream confidence-threshold filtering drop it naturally.

    Rows with conf <= 0 are left untouched (already invalid).
    """
    rows = lane_rows.copy()
    n = len(rows)
    keep = np.ones(n, dtype=bool)

    # Compare in descending-confidence order so higher-confidence lanes
    # always win when suppressing a duplicate.
    order = np.argsort(-rows[:, 0])

    for i_idx in range(n):
        i = order[i_idx]
        if rows[i, 0] <= 0 or not keep[i]:
            continue

        y_start_i, y_end_i = rows[i, 1], rows[i, 2]
        coeffs_i = rows[i, 3:7]

        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if rows[j, 0] <= 0 or not keep[j]:
                continue

            y_start_j, y_end_j = rows[j, 1], rows[j, 2]
            coeffs_j = rows[j, 3:7]

            # Shared y-range between the two lanes; skip if they don't overlap.
            lo = max(y_start_i, y_start_j)
            hi = min(y_end_i, y_end_j)
            if hi <= lo:
                continue

            y_mid = (lo + hi) / 2.0
            x_i = np.polyval(coeffs_i, y_mid)
            x_j = np.polyval(coeffs_j, y_mid)

            if abs(x_i - x_j) < x_threshold:
                keep[j] = False  # j has lower (or equal) confidence than i

    rows[~keep, 0] = 0  # zero out confidence so downstream filtering drops it
    return rows


def lane_points_to_mask(lane_points_norm: np.ndarray, w: int, h: int, thickness: int = 6) -> np.ndarray:
    """
    lane_points_norm: Nx2 normalized points in [0,1], format (x, y).
    Returns binary mask HxW (uint8: 0/1).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if lane_points_norm is None or len(lane_points_norm) < 2:
        return mask

    pts = np.asarray(lane_points_norm, dtype=np.float32).copy()
    pts[:, 0] = pts[:, 0] * (w - 1)
    pts[:, 1] = pts[:, 1] * (h - 1)
    pts = pts.round().astype(np.int32)

    for i in range(len(pts) - 1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), color=1, thickness=thickness)

    return mask


def decode_coco_ann_to_binary_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """
    Supports:
      - polygon segmentation (list)
      - RLE segmentation (dict)
    Returns mask HxW (uint8 0/1)
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(seg, list):
        rle = maskUtils.frPyObjects(seg, h, w)
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = np.any(m > 0, axis=2).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    if isinstance(seg, dict):
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = np.any(m > 0, axis=2).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    raise ValueError(f"Unknown segmentation type: {type(seg)}")


def compute_metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    eps = 1e-12
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def find_test_dirs(test_root: str) -> List[str]:
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test root not found: {test_root}")

    dirs = []
    for name in sorted(os.listdir(test_root)):
        p = os.path.join(test_root, name, "train")
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "_annotations.coco.json")):
            dirs.append(p)
    return dirs


def polylanenet_row_to_norm_points(lane_row: np.ndarray, num_points: int = POLY_SAMPLE_POINTS) -> np.ndarray:
    """
    Converts a single decoded PolyLaneNet lane row into an Nx2 array of
    normalized (x, y) points in [0,1], suitable for lane_points_to_mask().

    A decoded row (see PolyRegression.decode) has 7 values:
        [conf, y_start, y_end, c0, c1, c2, c3]
    where x = c0*y^3 + c1*y^2 + c2*y + c3, with x and y already normalized
    to [0,1] (this is how PolyLaneNet represents/trains on lanes).

    Rows with conf == 0 were zeroed out by decode() (i.e. below the
    confidence threshold) and should be skipped by the caller.
    """
    y_start, y_end = float(lane_row[1]), float(lane_row[2])
    coeffs = lane_row[3:7]

    ys = np.linspace(y_start, y_end, num_points)
    xs = np.polyval(coeffs, ys)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # Keep only points whose x falls inside the image (mirrors PolyLaneNet's
    # own visualization code, which drops points outside [0, img_w]).
    valid = (pts[:, 0] >= 0.0) & (pts[:, 0] <= 1.0)
    pts = pts[valid]

    return pts


def cache_directory_predictions(model, ann_file: str, img_w: int, img_h: int) -> Dict:
    """
    Runs inference exactly once per image in a directory and caches everything
    needed to re-render prediction masks at different confidence/thickness
    settings without re-running the model.

    Lanes are decoded with conf_threshold=0.0 so decode() doesn't zero out any
    lane's coefficients/confidence (sigmoid output is always >= 0). Real
    thresholding is applied later, per sweep combo, by comparing each row's
    conf score directly. Deduplication (merging near-identical duplicate
    lanes) is applied once here, right after decoding, since it does not
    depend on the confidence/thickness sweep settings.

    Returns:
      {
        "images": [{"gt_mask", "w0", "h0", "lane_rows"}, ...],
        "infer_total_time": float,
        "n_images": int,
      }
    """
    folder = os.path.dirname(ann_file)

    with open(ann_file, "r") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    anns_per_image = {}
    for ann in coco["annotations"]:
        anns_per_image.setdefault(ann["image_id"], []).append(ann)

    cached = []
    infer_total_time = 0.0
    n_images = 0

    with torch.no_grad():
        for image_id, im in images.items():
            file_name = im["file_name"]
            img_path = os.path.join(folder, file_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image: {img_path}")
                continue

            w0, h0 = int(im["width"]), int(im["height"])

            gt_mask = np.zeros((h0, w0), dtype=np.uint8)
            for ann in anns_per_image.get(image_id, []):
                lane_mask = decode_coco_ann_to_binary_mask(ann, h0, w0)
                gt_mask = np.maximum(gt_mask, lane_mask)

            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            resized = cv2.resize(bgr, (img_w, img_h))
            x = resized.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            output = model(x)
            decoded, _ = model.decode(output, None, conf_threshold=0.0)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            infer_total_time += (t1 - t0)

            lane_rows = decoded.cpu().numpy()[0]  # (num_lanes, 7)
            lane_rows = dedup_lane_rows(lane_rows)

            cached.append({
                "gt_mask": gt_mask,
                "w0": w0,
                "h0": h0,
                "lane_rows": lane_rows,
            })
            n_images += 1

    return {
        "images": cached,
        "infer_total_time": infer_total_time,
        "n_images": n_images,
    }


def metrics_for_settings(cached_images: List[Dict], confidence: float, thickness: int) -> Dict[str, float]:
    """
    Recomputes confusion-matrix metrics over already-cached predictions for a
    given (confidence, thickness) pair. No model inference happens here.
    """
    total_tp = total_fp = total_fn = total_tn = 0

    for item in cached_images:
        gt_mask = item["gt_mask"]
        w0, h0 = item["w0"], item["h0"]

        pred_mask = np.zeros((h0, w0), dtype=np.uint8)
        for lane_row in item["lane_rows"]:
            if lane_row[0] < confidence:
                continue
            lane_points_norm = polylanenet_row_to_norm_points(lane_row)
            m = lane_points_to_mask(lane_points_norm, w0, h0, thickness=thickness)
            pred_mask = np.maximum(pred_mask, m)

        gt = gt_mask.astype(bool)
        pr = pred_mask.astype(bool)

        tp = np.logical_and(pr, gt).sum()
        fp = np.logical_and(pr, ~gt).sum()
        fn = np.logical_and(~pr, gt).sum()
        tn = np.logical_and(~pr, ~gt).sum()

        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)
        total_tn += int(tn)

    return compute_metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)


def run_confidence_thickness_sweep(model, test_dirs: List[str], img_w: int, img_h: int) -> None:
    """
    Runs inference once per image across all test directories (cached), then
    sweeps CONFIDENCE_SWEEP x THICKNESS_SWEEP, re-thresholding/re-rendering
    masks per combo without any further model calls, and prints a comparison
    table sorted by F1 (best first).
    """
    print("===== CACHING PREDICTIONS (single inference pass per directory) =====")
    dir_caches = []
    total_infer_time = 0.0
    total_images = 0

    for d in test_dirs:
        ann_file = os.path.join(d, "_annotations.coco.json")
        dir_name = os.path.basename(os.path.dirname(d))
        print(f"  Running inference on: {dir_name} ...")
        cache = cache_directory_predictions(model, ann_file, img_w, img_h)
        dir_caches.append(cache)
        total_infer_time += cache["infer_total_time"]
        total_images += cache["n_images"]

    fps = (total_images / total_infer_time) if total_infer_time > 0 else 0.0
    print(f"\nCached predictions for {total_images} images in {total_infer_time:.3f}s ({fps:.2f} FPS)\n")

    all_cached_images = [img for cache in dir_caches for img in cache["images"]]

    results = []
    for confidence in CONFIDENCE_SWEEP:
        for thickness in THICKNESS_SWEEP:
            m = metrics_for_settings(all_cached_images, confidence, thickness)
            m["confidence"] = confidence
            m["thickness"] = thickness
            results.append(m)

    results.sort(key=lambda r: r["f1"], reverse=True)

    print("===== CONFIDENCE / THICKNESS SWEEP (sorted by F1, best first) =====")
    header = f"{'Conf':>6} {'Thick':>6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['confidence']:>6.2f} {r['thickness']:>6d} "
              f"{r['accuracy']:>10.6f} {r['precision']:>10.6f} {r['recall']:>10.6f} "
              f"{r['f1']:>10.6f} {r['iou']:>10.6f}")


def evaluate_one_directory(
    model,
    test_params: dict,
    ann_file: str,
    img_w: int,
    img_h: int
) -> Dict[str, float]:
    folder = os.path.dirname(ann_file)

    with open(ann_file, "r") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    anns_per_image = {}
    for ann in coco["annotations"]:
        anns_per_image.setdefault(ann["image_id"], []).append(ann)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    n_images = 0

    # Stratified accumulators: bucket each image by its GT lane count so we
    # can see whether errors concentrate on crossing (many-lane) images.
    bucket_totals = {
        "normal": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0},
        "crossing": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0},
    }

    infer_total_time = 0.0

    with torch.no_grad():
        for image_id, im in images.items():
            file_name = im["file_name"]
            img_path = os.path.join(folder, file_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image: {img_path}")
                continue

            w0, h0 = int(im["width"]), int(im["height"])

            # Ground truth mask
            gt_mask = np.zeros((h0, w0), dtype=np.uint8)
            for ann in anns_per_image.get(image_id, []):
                lane_mask = decode_coco_ann_to_binary_mask(ann, h0, w0)
                gt_mask = np.maximum(gt_mask, lane_mask)

            # Inference
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            resized = cv2.resize(bgr, (img_w, img_h))
            x = resized.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Inference timing
            t0 = time.perf_counter()
            output = model(x)
            decoded, _ = model.decode(output, None, conf_threshold=test_params["conf_threshold"])
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            infer_total_time += (t1 - t0)

            lane_rows = decoded.cpu().numpy()[0]  # (num_lanes, 7)
            lane_rows = dedup_lane_rows(lane_rows)

            # Prediction mask on original image size
            pred_mask = np.zeros((h0, w0), dtype=np.uint8)
            for lane_row in lane_rows:
                if lane_row[0] <= 0:  # invalid / below confidence threshold
                    continue
                lane_points_norm = polylanenet_row_to_norm_points(lane_row)
                m = lane_points_to_mask(lane_points_norm, w0, h0, thickness=PRED_LINE_THICKNESS)
                pred_mask = np.maximum(pred_mask, m)

            gt = gt_mask.astype(bool)
            pr = pred_mask.astype(bool)

            tp = np.logical_and(pr, gt).sum()
            fp = np.logical_and(pr, ~gt).sum()
            fn = np.logical_and(~pr, gt).sum()
            tn = np.logical_and(~pr, ~gt).sum()

            total_tp += int(tp)
            total_fp += int(fp)
            total_fn += int(fn)
            total_tn += int(tn)
            n_images += 1

            n_gt_lanes = len(anns_per_image.get(image_id, []))
            bucket = "crossing" if n_gt_lanes >= CROSSING_LANE_COUNT else "normal"
            bucket_totals[bucket]["tp"] += int(tp)
            bucket_totals[bucket]["fp"] += int(fp)
            bucket_totals[bucket]["fn"] += int(fn)
            bucket_totals[bucket]["tn"] += int(tn)
            bucket_totals[bucket]["images"] += 1

    m = compute_metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    fps = (n_images / infer_total_time) if infer_total_time > 0 else 0.0

    m.update({
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "images": n_images,
        "fps": fps,
        "infer_time_sec": infer_total_time,
    })

    by_lane_count = {}
    for bucket_name, b in bucket_totals.items():
        bm = compute_metrics_from_confusion(b["tp"], b["fp"], b["fn"], b["tn"])
        bm.update({
            "tp": b["tp"],
            "fp": b["fp"],
            "fn": b["fn"],
            "tn": b["tn"],
            "images": b["images"],
        })
        by_lane_count[bucket_name] = bm
    m["by_lane_count"] = by_lane_count

    return m


def main():
    # Load model
    cfg = Config(CFG_PATH)
    # cfg.get_model() builds the model from `model.name` / `model.parameters`
    # in the YAML. For this config that resolves to PolyLaneNet's
    # PolyRegression(num_outputs=56, backbone="efficientnet-b0", ...).
    model = cfg.get_model().to(DEVICE).eval()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    # PolyLaneNet doesn't store img_w/img_h under model.parameters; the
    # input resolution comes from the dataset config's img_size = [h, w].
    img_h, img_w = cfg["datasets"]["test"]["parameters"]["img_size"]
    test_params = cfg.get_test_parameters()
    test_params["conf_threshold"] = CONFIDENCE

    # Find test directories
    test_dirs = find_test_dirs(TEST_ROOT)
    if not test_dirs:
        raise RuntimeError(f"No test directories with _annotations.coco.json found in: {TEST_ROOT}")

    if RUN_SWEEP:
        run_confidence_thickness_sweep(model, test_dirs, img_w, img_h)
        return

    # Global accumulators
    g_tp, g_fp, g_fn, g_tn = 0, 0, 0, 0
    g_images = 0
    g_fps = 0
    g_inference_time = 0

    g_bucket_totals = {
        "normal": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0},
        "crossing": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0},
    }

    print("===== PER-DIRECTORY METRICS =====")
    for d in test_dirs:
        ann_file = os.path.join(d, "_annotations.coco.json")
        dir_name = os.path.basename(os.path.dirname(d))  # e.g. "apagadas1" instead of "train"

        metrics = evaluate_one_directory(model, test_params, ann_file, img_w, img_h)

        print(f"\nDirectory: {dir_name}")
        print(f"  Images       : {metrics['images']}")
        print(f"  Accuracy     : {metrics['accuracy']:.6f}")
        print(f"  Precision    : {metrics['precision']:.6f}")
        print(f"  Recall       : {metrics['recall']:.6f}")
        print(f"  F1 score     : {metrics['f1']:.6f}")
        print(f"  IoU          : {metrics['iou']:.6f}")
        print(f"  FPS          : {metrics['fps']:.3f}")
        print(f"  Infer time(s): {metrics['infer_time_sec']:.3f}")

        for bucket_name in ("normal", "crossing"):
            bm = metrics["by_lane_count"][bucket_name]
            label = f"<{CROSSING_LANE_COUNT} lanes" if bucket_name == "normal" else f">={CROSSING_LANE_COUNT} lanes"
            print(f"  -- {bucket_name} ({label}) --")
            print(f"     Images    : {bm['images']}")
            if bm["images"] == 0:
                print("     (no images in this bucket)")
                continue
            print(f"     Precision : {bm['precision']:.6f}")
            print(f"     Recall    : {bm['recall']:.6f}")
            print(f"     F1 score  : {bm['f1']:.6f}")
            print(f"     IoU       : {bm['iou']:.6f}")

        g_tp += metrics["tp"]
        g_fp += metrics["fp"]
        g_fn += metrics["fn"]
        g_tn += metrics["tn"]
        g_images += metrics["images"]
        g_fps += metrics['fps']
        g_inference_time += metrics['infer_time_sec']

        for bucket_name in ("normal", "crossing"):
            bm = metrics["by_lane_count"][bucket_name]
            g_bucket_totals[bucket_name]["tp"] += bm["tp"]
            g_bucket_totals[bucket_name]["fp"] += bm["fp"]
            g_bucket_totals[bucket_name]["fn"] += bm["fn"]
            g_bucket_totals[bucket_name]["tn"] += bm["tn"]
            g_bucket_totals[bucket_name]["images"] += bm["images"]


    g = compute_metrics_from_confusion(g_tp, g_fp, g_fn, g_tn)
    g['fps'] = g_fps/len(test_dirs)
    g['infer_time_sec'] = g_inference_time/len(test_dirs)

    print("\n===== GLOBAL (ALL TEST DIRECTORIES COMBINED) =====")
    print(f"Images evaluated: {g_images}")
    print(f"Accuracy     : {g['accuracy']:.6f}")
    print(f"Precision    : {g['precision']:.6f}")
    print(f"Recall       : {g['recall']:.6f}")
    print(f"F1 score     : {g['f1']:.6f}")
    print(f"IoU          : {g['iou']:.6f}")
    print(f"FPS          : {g['fps']:.3f}")
    print(f"Infer time(s): {g['infer_time_sec']:.3f}")

    print("\n===== GLOBAL BY LANE COUNT =====")
    for bucket_name in ("normal", "crossing"):
        b = g_bucket_totals[bucket_name]
        label = f"<{CROSSING_LANE_COUNT} lanes" if bucket_name == "normal" else f">={CROSSING_LANE_COUNT} lanes"
        print(f"\n{bucket_name} ({label})")
        print(f"  Images    : {b['images']}")
        if b["images"] == 0:
            print("  (no images in this bucket)")
            continue
        bg = compute_metrics_from_confusion(b["tp"], b["fp"], b["fn"], b["tn"])
        print(f"  Accuracy  : {bg['accuracy']:.6f}")
        print(f"  Precision : {bg['precision']:.6f}")
        print(f"  Recall    : {bg['recall']:.6f}")
        print(f"  F1 score  : {bg['f1']:.6f}")
        print(f"  IoU       : {bg['iou']:.6f}")


if __name__ == "__main__":
    main()

