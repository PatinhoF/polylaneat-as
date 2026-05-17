# SETUP

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (fast Python package manager, I recommend using it because of their new feature that allows using different python versions)
- Python 3.10
- CUDA-compatible GPU (recommended)

## 1. Virtual environment

```sh
uv venv .venv --python 3.10

# activate (adjust for your shell)
source .venv/bin/activate       # bash / zsh
source .venv/bin/activate.fish  # fish
```

## 2. Dependencies

```sh
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install opencv-python efficientnet-pytorch pyyaml numpy tqdm
```

> If you get a `weights_only` warning loading the checkpoint, add `weights_only=False` to `torch.load()` in the inference scripts.

## 3. Weights

Download the TuSimple checkpoint from the [PolyLaneNet Google Drive](https://drive.google.com/drive/folders/1oyZncVnUB1GRJl5L4oXz50RkcNFM_FFC).

Place the weights at the path expected by the scripts:

```sh
mkdir -p experiments/tusimple/models
# e.g. cp ~/Downloads/model_2695.pt experiments/tusimple/models/
```

The scripts look for `experiments/tusimple/models/model_2695.pt` by default — edit the `CHECKPOINT` variable in `infer.py` / `infer_video.py` to use a different checkpoint.

## 4. Run inference

Edit the paths inside the scripts (`IMAGE_PATH`, `VIDEO_PATH`, `OUT_PATH`), then:

```sh
python infer.py         # single image
python infer_video.py   # video
```

## Notes

- The infer scripts were made to only work with CUDA.
- If you installed newer PyTorch (≥2.x) and get a `weights_only` pickle warning, add `weights_only=False` to `torch.load()`.
- `lib/models.py` line 39 was already patched to use `num_classes=num_outputs` instead of the deprecated `override_params`. The changes I am refering to were:

```diff
--- a/lib/models.py
self.model = EfficientNet.from_name(backbone, override_params={'num_classes': num_outputs})

+++ b/lib/models.py
self.model = EfficientNet.from_name(backbone, num_classes=num_outputs)
```
