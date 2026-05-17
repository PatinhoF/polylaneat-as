import torch
import cv2
import numpy as np
from tqdm import tqdm
from lib.models import PolyRegression

CHECKPOINT = "experiments/tusimple/models/model_2695.pt"
VIDEO_PATH = "dia1.mp4"
OUT_PATH   = "out_dia1.mp4"
CONF       = 0.5
TOP_K      = 4
H, W       = 360, 640
DEVICE     = torch.device('cuda')

# load model
model = PolyRegression(
    num_outputs=35,
    pretrained=False,
    backbone='efficientnet-b0',
    pred_category=False
)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
model.eval()

def draw_lanes(frame, lanes_out):
    canvas = frame.copy()
    for lane in lanes_out:
        conf_logit = lane[0]
        conf = 1 / (1 + np.exp(-conf_logit))
        y_min, y_max = lane[1], lane[2]
        coeffs = lane[3:][::-1]
        if conf < CONF:
            continue
        ys = np.linspace(y_min, y_max, 50)
        xs = sum(c * ys**i for i, c in enumerate(coeffs))
        for x, y in zip(xs, ys):
            px, py = int(x * W), int(y * H)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)
    return canvas

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Could not open {VIDEO_PATH}"

# use input fps if available, else 30
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (W, H))

for _ in tqdm(range(total_frames), desc="Processing"):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (W, H))
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = ((t - mean) / std).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out, _ = model(t)
        out = out.squeeze(0).cpu().numpy().reshape(5, 7)
        confs = 1 / (1 + np.exp(-out[:, 0]))
        idx = np.argsort(confs)[-TOP_K:]
        out = out[idx]
    canvas = draw_lanes(frame, out)
    writer.write(canvas)

cap.release()
writer.release()
print(f"Done → {OUT_PATH}")
