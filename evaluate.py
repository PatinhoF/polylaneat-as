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
CKPT_PATH  = "model_150.pt"
TEST_ROOT = "datasets/test"   # evaluate each subdirectory independently
CONFIDENCE = 0.25                  # override model conf threshold

# Lane rendering thickness (pixels) for predicted lane polyline -> mask
PRED_LINE_THICKNESS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        p = os.path.join(test_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "_annotations.coco.json")):
            dirs.append(p)
    return dirs


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
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Inference timing
            t0 = time.perf_counter()
            output = model(x, **test_params)
            lanes = model.decode(output, as_lanes=True)[0]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            infer_total_time += (t1 - t0)

            # Prediction mask on original image size
            pred_mask = np.zeros((h0, w0), dtype=np.uint8)
            for lane in lanes:
                m = lane_points_to_mask(lane.points, w0, h0, thickness=PRED_LINE_THICKNESS)
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
    return m


def main():
    # Load model
    cfg = Config(CFG_PATH)
    model = cfg.get_model().to(DEVICE).eval()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    img_w = cfg["model"]["parameters"]["img_w"]
    img_h = cfg["model"]["parameters"]["img_h"]
    test_params = cfg.get_test_parameters()
    test_params["conf_threshold"] = CONFIDENCE

    # Find test directories
    test_dirs = find_test_dirs(TEST_ROOT)
    if not test_dirs:
        raise RuntimeError(f"No test directories with _annotations.coco.json found in: {TEST_ROOT}")

    # Global accumulators
    g_tp, g_fp, g_fn, g_tn = 0, 0, 0, 0
    g_images = 0
    g_fps = 0
    g_inference_time = 0

    print("===== PER-DIRECTORY METRICS =====")
    for d in test_dirs:
        ann_file = os.path.join(d, "_annotations.coco.json")
        dir_name = os.path.basename(d)

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

        g_tp += metrics["tp"]
        g_fp += metrics["fp"]
        g_fn += metrics["fn"]
        g_tn += metrics["tn"]
        g_images += metrics["images"]
        g_fps += metrics['fps']
        g_inference_time += metrics['infer_time_sec']


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


if __name__ == "__main__":
    main()
