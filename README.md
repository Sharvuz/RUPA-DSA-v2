# RUPA-DSA v0

**Video-specific counterfactual normal reconstruction with category-aware residual semantic alignment for weakly supervised video anomaly detection.**

RUPA-DSA v0 extends [DSANet](https://github.com/lessiYin/DSANet) with counterfactual reconstruction, residual semantic alignment, and fixed closed-loop routing:

```text
F_video --video-specific DNP--> F_rec
R = F_video - F_rec

F_rec + DNP <--> normal text
R             <--> anomaly-category text

S* = w_det S_det + w_rec S_rec + w_sem S_sem
```

The narrow research contribution is the structured combination of video-specific normal reconstruction and category-aware residual–text alignment. This repository does not claim that prototypes, reconstruction error, residual learning, or CLIP alignment are individually new.

## Repository structure

```text
RUPA-DSA-v0/
├── RUPA_DSA_Kaggle.ipynb   # Kaggle runner; clones this repository directly
├── RUPA_DSA.md              # method and ablation protocol
├── environment.yml
├── requirements.txt
├── assets/
├── list/                    # CSV manifests and evaluation annotations
└── src/
    ├── model.py             # RUPA reconstruction/residual/routing
    ├── ucf_train.py
    ├── ucf_test.py
    ├── xd_train.py
    ├── xd_test.py
    ├── clip/
    └── utils/
```

Feature archives, checkpoints, and outputs are intentionally excluded by `.gitignore`.

## Train on Kaggle

1. Import `RUPA_DSA_Kaggle.ipynb` into Kaggle.
2. Enable **GPU** and **Internet**.
3. For UCF-Crime, attach `UCFClipFeatures.zip` and set `DATASET = "ucf"`.
4. For XD-Violence, attach `XDTrainClipFeatures.zip` plus `XDTestClipFeatures.zip` and set `DATASET = "xd"`.
5. Run all cells.
6. Download `RUPA_DSA_v0_<dataset>_results.zip` from `/kaggle/working/rupa_v0_artifacts`.

The notebook executes:

```bash
git clone --depth 1 https://github.com/Sharvuz/RUPA-DSA-v0.git
```

There is no embedded/base64 source overlay. Every Kaggle run therefore uses the current code on GitHub, or the optional `REPO_REF` commit/tag configured in the notebook.

## Local environment

```bash
conda env create -f environment.yml
conda activate rupa-dsa
```

Alternatively, install a CUDA-compatible PyTorch build first and then run:

```bash
pip install -r requirements.txt
```

## Direct CLI training

Update the feature paths in the CSV manifests or pass mapped CSV files explicitly:

```bash
python src/ucf_train.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --model-path model/best_ucf.pth \
  --checkpoint-path model/checkpoint_ucf.pth

python src/xd_train.py \
  --train-list list/xd_CLIP_rgb.csv \
  --test-list list/xd_CLIP_rgbtest.csv \
  --model-path model/best_xd.pth \
  --checkpoint-path model/checkpoint_xd.pth
```

All RUPA routing and loss weights are exposed in `src/ucf_option.py` and `src/xd_option.py`. See [RUPA_DSA.md](RUPA_DSA.md) for the recommended ablations.

## Attribution

This codebase is derived from the official DSANet implementation:

```bibtex
@inproceedings{yin2026learning,
  title={Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment},
  author={Yin, Wenti and Zhang, Huaxin and Wang, Xiang and Lu, Yuqing and Zhang, Yicheng and Gong, Bingquan and Zuo, Jialong and Yu, Li and Gao, Changxin and Sang, Nong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
