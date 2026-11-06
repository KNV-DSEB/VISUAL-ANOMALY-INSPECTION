# Protocol note (D-01)

Giao thức dùng chung cho Autoencoder, PatchCore và One-class. Không đổi sau khi bắt đầu chạy benchmark chính thức.
Nguồn cấu hình duy nhất: [`configs/protocol.yaml`](../configs/protocol.yaml). Schema dự đoán, quy tắc ngưỡng và định nghĩa metric:
[`evaluation_schema.md`](evaluation_schema.md).

## 1. Category

5 category cố định, chọn để phủ cả bề mặt texture và vật thể, lỗi lớn lẫn lỗi nhỏ. Số ảnh đối chiếu với `results/eda_counts.csv`
(khớp số liệu chính thức của MVTec AD).

| Category | Nhóm | Train | Val (20%) | Test good / defect | Loại lỗi trong test |
|---|---|---:|---:|---:|---|
| `bottle` | vật thể cứng, đối xứng | 167 | 42 | 20 / 63 | broken_large, broken_small, contamination |
| `carpet` | texture vải dệt | 224 | 56 | 28 / 89 | color, cut, hole, metal_contamination, thread |
| `leather` | texture da | 196 | 49 | 32 / 92 | color, cut, fold, glue, poke |
| `screw` | chi tiết nhỏ, xoay góc | 256 | 64 | 41 / 119 | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top |
| `transistor` | linh kiện điện tử | 170 | 43 | 60 / 40 | bent_lead, cut_lead, damaged_case, misplaced |

Tổng: 1.267 ảnh train + val (normal) và 584 ảnh test. Ảnh `screw` là grayscale, loader chuyển sang RGB.

## 2. Tiền xử lý (chung cho cả 3 phương pháp)

- Ảnh: resize 256 (bilinear), center-crop 224, chuẩn hóa ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Mask ground-truth: resize 256 (nearest), center-crop 224, nhị phân hóa {0, 1}. Ảnh normal có mask toàn 0.
- Cài đặt: `src/data/mvtec.py` (`build_transforms`, `MVTecDataset`).

## 3. Chia dữ liệu và chống rò rỉ

- `train`: 80% của `train/good`. Chỉ ảnh normal. Dùng để fit mô hình (AE, memory bank PatchCore, scaler/OC-SVM).
- `val`: 20% của `train/good`, seed 42 theo từng category. Chỉ ảnh normal. Chỉ dùng cho: (1) chọn checkpoint Autoencoder theo val loss,
  (2) ngưỡng quyết định.
- `test`: tập test chính thức của MVTec AD. Không dùng để chọn checkpoint, hyperparameter hay ngưỡng.
- Ngưỡng: percentile P95 (`threshold.percentile`) của điểm trên `val` normal, tính riêng từng category, qua `percentile_threshold()`.
  Hàm này từ chối split khác `val` hoặc `val` có ảnh lỗi.
- Vì `val` không có ảnh lỗi nên không thể chọn hyperparameter theo AUROC hay F1 trên `val`. Hyperparameter của One-class
  (`nu`, `gamma`, layer đặc trưng) được cố định từ trước trong `configs/oneclass.yaml`; mọi cấu hình cải tiến (ví dụ
  `configs/experiments/patchcore_refined.yaml`) phải được khai báo trước và báo cáo cạnh baseline, không chọn bằng kết quả test.

## 4. Ghi nhận khi chạy

Mỗi file dự đoán lưu `meta` gồm: method, category, config (đường dẫn tương đối), seed, backbone và trọng số được ghim, số ảnh
train/val/test, thời gian fit, thời gian suy luận mỗi ảnh, kích thước checkpoint, git commit và phiên bản thư viện.
