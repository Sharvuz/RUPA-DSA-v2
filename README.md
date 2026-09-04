# RUPA-DSA v2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**RUPA-DSA v2** là phiên bản tối ưu hóa cho bài toán Phát hiện Dị thường Video (Weakly Supervised Video Anomaly Detection). Phiên bản này xây dựng dựa trên DSANet, tích hợp cơ chế mô hình hóa counterfactual (tái tạo đặc trưng bình thường) và đối chiếu ngữ nghĩa thặng dư (residual semantics), đồng thời khắc phục triệt để điểm yếu của các phiên bản trước thông qua thuật toán **Adaptive Normal Selection (Otsu's Thresholding)**.

---

## 🌟 Điểm nổi bật của v2

Trong bài toán học yếu (weakly supervised), việc chọn nhầm khung hình dị thường vào tập bình thường (contamination) là nguyên nhân chính gây giảm hiệu suất. RUPA-DSA v2 giải quyết vấn đề này qua 3 trụ cột kiến trúc:

1. **Adaptive Normal Selection (SGNM với Otsu's Thresholding):**
   Thay vì lấy tỷ lệ khung hình bình thường cố định (vd: 12.5% hoặc 80%) như các mạng MIL truyền thống, v2 áp dụng thuật toán Otsu trên phân bố điểm dị thường (1D Anomaly Scores) của từng video để động (adaptively) tìm ra ngưỡng cắt. Nhờ đó, mạng tự động điều chỉnh số lượng khung hình bình thường được chọn, chống nhiễu tuyệt đối ở các video có quá nhiều sự kiện dị thường.

2. **Counterfactual Normal Reconstruction:**
   Sử dụng mạng DNP (Dual Normal Prompts) để từ các khung hình bình thường được chọn, tái tạo (reconstruct) lại không gian đặc trưng bình thường lý tưởng cho toàn bộ video.

3. **Residual Semantic Alignment:**
   Lấy đặc trưng gốc trừ đi đặc trưng bình thường vừa tái tạo để ra phần "thặng dư" (Residual). Phần thặng dư này chứa thông tin dị thường nguyên chất nhất và được đối chiếu trực tiếp với text prompt (CLIP) để phân loại.

---

## 📊 Kết quả Benchmark (Dataset: UCF-Crime)

Dưới đây là kết quả huấn luyện tốt nhất được ghi nhận từ log của mô hình (Train/Test trên CLIP features của bộ dữ liệu UCF-Crime):

| Metric | Score (v2) | Ý nghĩa |
|--------|:---:|---|
| **Max AUC** | **87.61%** | (Area Under Curve) Đo lường khả năng phân tách tổng thể giữa Normal và Abnormal. Mức 87.6+ là mức rất cao so với các baseline MIL truyền thống. |
| **Max AP** | **38.31%** | (Average Precision) Đặc biệt quan trọng vì dataset UCF mất cân bằng nghiêm trọng giữa số khung hình bình thường và bất thường. |
| **mAP@0.1** | **18.08%** | Đo độ chính xác tại top 10% các frame được dự đoán dị thường cao nhất (Top-tier predictions). |
| **Average MAP** | **9.95%** | Trung bình Mean Average Precision ở nhiều ngưỡng IoU khác nhau. |

> **Lưu ý:** Ở phiên bản v2 này, hiệu năng lập đỉnh tuyệt đối tại cấu hình **Batch Size 16**. Việc tăng Batch Size lên cao hơn (như 32) mà không có Warm-Start sẽ làm giảm độ sắc bén của Contrastive Loss, khiến AP bị tụt. Do đó, hãy giữ nguyên cấu hình BS 16.

---

## 🛠 Hướng dẫn huấn luyện (Training & Usage)

Để tái lập lại đúng mức điểm số kỷ lục như trên (với cấu hình Batch Size 16), hãy sử dụng chính xác lệnh huấn luyện sau:

```bash
python src/ucf_train.py \
  --train-list /kaggle/working/rupa_artifacts/ucf_train.csv \
  --test-list /kaggle/working/rupa_artifacts/ucf_test.csv \
  --model-path /kaggle/working/rupa_artifacts/ucf/best_ucf.pth \
  --checkpoint-path /kaggle/working/rupa_artifacts/ucf/checkpoint_ucf.pth \
  --max-epoch 10 \
  --batch-size 16 \
  --num-workers 2 \
  --seed 234 \
  --rupa-use true \
  --routing-det-weight 0.5 \
  --routing-rec-weight 0.3 \
  --routing-sem-weight 0.2 \
  --loss-residual-weight 1.0 \
  --loss-reconstructed-normal-weight 1.0 \
  --loss-dnp-normal-weight 0.1 \
  --loss-consistency-weight 1.0 \
  --loss-gather-weight 1.0
```

### Giải thích các tham số cực kỳ quan trọng (Hyperparameters):
- `--batch-size 16`: Cấu hình "điểm ngọt" (sweet spot) giúp Gradient cập nhật đủ nhạy bén để bắt được các dị thường tinh vi nhất ở ngay Epoch 1.
- Các hệ số `--routing-*-weight`: Được điều chỉnh để pha trộn hoàn hảo điểm số từ **Base Detector (0.5)**, **Reconstruction (0.3)** và **Residual Semantics (0.2)**.

---

## 📓 Notebook
Tệp `rupa.ipynb` đi kèm cung cấp một luồng chạy hoàn chỉnh (End-to-End) có thể được nạp trực tiếp lên Kaggle hoặc Google Colab để tiến hành thí nghiệm.

---

## ⚖️ License
Dự án được phân phối dưới giấy phép MIT License.
