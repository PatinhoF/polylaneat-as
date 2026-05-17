import torch
import cv2
import numpy as np
from lib.models import PolyRegression

CHECKPOINT = "experiments/tusimple/models/model_2695.pt"
IMAGE_PATH = "image1.jpg"
OUT_PATH   = "result1.jpg"
CONF       = 0.5
TOP_K      = 4
H, W       = 360, 640
DEVICE     = torch.device('cuda')

# load model
model = PolyRegression(
    num_outputs=35,
    pretrained=False,
    backbone='efficientnet-b0', # it's what's written in experiments/tusimple/config.yaml
    pred_category=False
)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
model.eval()

# preprocess -> try to understand it more properly
img_orig = cv2.imread(IMAGE_PATH)
assert img_orig is not None, f"Could not read {IMAGE_PATH}"
img  = cv2.resize(img_orig, (W, H))
t    = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
t    = ((t - mean) / std).unsqueeze(0).to(DEVICE)

# infer
with torch.no_grad():
    out, _ = model(t)
    out = out.squeeze(0).cpu().numpy()  # (35,)

out = out.reshape(5, 7)  # 5 lanes × 7 values each
confs = 1 / (1 + np.exp(-out[:, 0]))
idx = np.argsort(confs)[-TOP_K:]
out = out[idx]

# draw
canvas = img.copy()
for lane in out:
    conf_logit = lane[0]
    conf = 1 / (1 + np.exp(-conf_logit))   # sigmoid
    y_min, y_max = lane[1], lane[2]
    coeffs = lane[3:][::-1]                 # reverse: now a0 + a1*y + a2*y^2 + a3*y^3
    if conf < CONF:
        continue
    ys = np.linspace(y_min, y_max, 50)
    xs = sum(c * ys**i for i, c in enumerate(coeffs))
    for x, y in zip(xs, ys):
        px, py = int(x * W), int(y * H)
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)

cv2.imwrite(OUT_PATH, canvas)
print(f"Done → {OUT_PATH}")
