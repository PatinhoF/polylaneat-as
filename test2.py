
import cv2
import torch
import numpy as np
import time
from lib.config import Config


CFG_PATH   = "cfgs/coco_lanes.yaml" # change according to the model
CKPT_PATH  = "model_150.pt"
#VIDEO_PATH = "/data/temp/lane_detection/bfmc/21-09-17-12-39-53.mp4"
VIDEO_PATH = "apagadas1.mp4"
OUT_PATH   = "output/lane_detection_out_" + str(time.time()) +  ".mp4"
LIMIT_LANES = 100
CONFIDENCE = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def points_to_pixels(points01: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert normalized [0..1] points to pixel coordinates in an image of size (w,h)."""
    pts = np.asarray(points01, dtype=np.float32)
    if pts.size == 0:
        return pts.astype(np.int32)

    pts_px = np.empty_like(pts)
    pts_px[:, 0] = pts[:, 0] * (w - 1)
    pts_px[:, 1] = pts[:, 1] * (h - 1)
    return pts_px.astype(np.int32)

def main():
    cfg = Config(CFG_PATH)
    model = cfg.get_model().to(DEVICE).eval()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    img_w = cfg["model"]["parameters"]["img_w"]
    img_h = cfg["model"]["parameters"]["img_h"]
    test_params = cfg.get_test_parameters()
    test_params["conf_threshold"] = CONFIDENCE

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    t_start = time.perf_counter()

    try:
        with torch.no_grad():
            print("Starting the video processing")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break

                frame_idx += 1

                # Resize to model input
                fr = cv2.resize(frame, (img_w, img_h))

                # Preprocess
                x = fr.astype(np.float32) / 255.0
                x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                # Forward + decode
                output = model(x, **test_params)
                lanes = model.decode(output, as_lanes=True)[0]

                # Draw on ORIGINAL frame size
                lane_num = 0
                for lane in lanes:
                    if lane_num == LIMIT_LANES:
                        break

                    pts = points_to_pixels(lane.points, out_w, out_h)
                    if len(pts) < 2:
                        continue

                    for i in range(len(pts) - 1):
                        cv2.line(
                            frame,
                            tuple(pts[i]),
                            tuple(pts[i + 1]),
                            (0, 255, 0),
                            3
                        )

                    lane_num += 1

                out.write(frame)


    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
        print(f"Stopping after {frame_idx} processed frames...")
        print("Finalizing output video...")

    finally:
        cap.release()
        out.release()

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        effective_fps = frame_idx / elapsed if elapsed > 0 else 0

        print(f"Saved: {OUT_PATH}")
        print(
            f"Processed {frame_idx} frames in "
            f"{elapsed:.3f} s => {effective_fps:.2f} FPS"
        )

if __name__ == "__main__":
    main()
