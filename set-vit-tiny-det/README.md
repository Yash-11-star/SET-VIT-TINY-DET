# SET-ViT: Spectral-Enhanced Vision Transformer for Tiny/Small Object Detection

This project integrates **frequency-domain background smoothing (HBS)** and **feature-level adversarial perturbation injection (API)**—originally proposed for CNN detectors—into a **transformer-based detector** (e.g., Deformable DETR / ViT neck) to improve **AP_S / AP for tiny objects** with **zero inference overhead**. Inspired by SET’s findings on frequency signatures and high-frequency clutter, and by the SOD transformer survey. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

> Stretch goal: validate on aerial SOD datasets (AI-TOD, SODA-D/SODA-A); consider coarse-to-fine insights from CFINet for future work. :contentReference[oaicite:2]{index=2}

## Quickstart

```bash
# 0) Python env
conda create -n setvit python=3.10 -y
conda activate setvit
pip install -r requirements.txt

# 1) COCO setup (adjust paths inside configs/coco_small.yaml)
bash scripts/setup_coco.sh

# 2) Train (Deformable DETR baseline + HBS + API)
bash scripts/train.sh

# 3) Evaluate
bash scripts/eval.sh
