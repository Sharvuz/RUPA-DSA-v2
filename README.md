# RUPA-DSA v2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Video-specific counterfactual normal reconstruction with adaptive normal selection and baseline-preserving risk gates for weakly supervised video anomaly detection.**

RUPA-DSA v2 represents a major evolution of the RUPA-DSA architecture (extending [DSANet](https://github.com/lessiYin/DSANet)). Building upon the residual semantic alignment introduced in v0, this version solves the "fixed normal ratio contamination" problem by introducing **Adaptive Normal Selection via Otsu's Thresholding**, along with a **Baseline-Preserving Risk Gate** (introduced in v1) to stabilize early-stage training.

---

## 🚀 Key Contributions (v2 Architecture)

RUPA-DSA v2 tackles the limitations of fixed parameterizations in earlier versions with three core pillars:

1. **Adaptive Normal Selection (v2 Core Update):**
   Previous versions (and traditional MIL approaches) assumed a fixed ratio of normal frames (e.g., 12.5% or 80%). This caused feature contamination in videos that were overwhelmingly anomalous. v2 utilizes **Otsu's Thresholding** on 1D anomaly scores (`S_det`) to dynamically and adaptively determine the threshold for normal frames per-video. 
   ```text
   anomaly_scores = sigmoid(logits_det)
   threshold = otsu_threshold(valid_scores)
   normal_frames = frames[anomaly_scores < threshold]  --> Extract DNP
   ```

2. **Baseline-Preserving Risk Gate (from v1):**
   Instead of a static score fusion which could degrade the performance of a warm-started pre-trained detector, v2 uses a per-snippet risk gate $g(video)$ initialized near zero:
   ```text
   S_fixed = w_det * S_det + w_rec * S_rec + w_sem * S_sem
   S_safe  = S_det + g(video) * (S_fixed - S_det)
   ```
   This ensures RUPA-DSA safely explores residual adjustments without collapsing the baseline accuracy.

3. **Counterfactual Reconstruction & Residual Learning (from v0):**
   Extracts Dynamic Normal Patterns (DNPs) to reconstruct the normal feature space ($F_{rec}$) and isolates anomalies using residual learning ($R = F_{video} - F_{rec}$) for accurate semantic alignment.

---

## 📂 Repository Structure

```text
RUPA-DSA-v2/
├── RUPA_DSA_Kaggle.ipynb    # One-click Kaggle training runner
├── RUPA_DSA.md              # Detailed method and ablation protocol
├── implementation_plan.md   # Architectural documentation of v2 Adaptive Selection
├── environment.yml          # Conda environment specifications
├── requirements.txt         # Pip requirements
├── assets/                  # Images and diagrams
├── list/                    # CSV manifests and evaluation annotations
└── src/
    ├── model.py             # RUPA v2 architecture (Otsu thresholding, Risk Gate)
    ├── ucf_train.py / ucf_test.py
    ├── xd_train.py / xd_test.py
    ├── clip/                # CLIP ViT-B/16 modules
    └── utils/
```

---

## ☁️ Train on Kaggle

The fastest way to reproduce results is via Kaggle using the provided notebook:

1. Import `RUPA_DSA_Kaggle.ipynb` into your Kaggle workspace.
2. Enable **GPU** (T4/P100) and **Internet** access.
3. **For UCF-Crime:** Attach the `UCFClipFeatures.zip` dataset and set `DATASET = "ucf"`.
4. **For XD-Violence:** Attach `XDTrainClipFeatures.zip` and `XDTestClipFeatures.zip`, then set `DATASET = "xd"`.
5. Run all cells. 
6. Retrieve your results and weights from `/kaggle/working/rupa_v2_artifacts/`.

---

## 💻 Local Installation

To run the repository on your local workstation:

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate rupa-dsa
```

**Using Pip:**
Ensure you have a CUDA-compatible PyTorch build installed, then run:
```bash
pip install -r requirements.txt
```

---

## 🏃 Direct CLI Training

To train the model locally, configure your CSV manifests in `list/` and utilize the new v2 arguments (`--adaptive_normal_selection`).

**Train on UCF-Crime:**
```bash
python src/ucf_train.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --model-path model/best_ucf.pth \
  --checkpoint-path model/checkpoint_ucf.pth \
  --adaptive_normal_selection True \
  --min_normal_frames 4 \
  --max_normal_ratio 0.8
```

**Train on XD-Violence:**
```bash
python src/xd_train.py \
  --train-list list/xd_CLIP_rgb.csv \
  --test-list list/xd_CLIP_rgbtest.csv \
  --model-path model/best_xd.pth \
  --checkpoint-path model/checkpoint_xd.pth \
  --adaptive_normal_selection True
```

*Fallback:* To reproduce v1/v0 behavior with a fixed ratio, simply pass `--adaptive_normal_selection False`.

---

## 📜 Attribution & Acknowledgments

This codebase is a heavily modified evolution of the official **DSANet** implementation. If you use this code in your research, please cite the original AAAI 2026 paper:

```bibtex
@inproceedings{yin2026learning,
  title={Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment},
  author={Yin, Wenti and Zhang, Huaxin and Wang, Xiang and Lu, Yuqing and Zhang, Yicheng and Gong, Bingquan and Zuo, Jialong and Yu, Li and Gao, Changxin and Sang, Nong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
