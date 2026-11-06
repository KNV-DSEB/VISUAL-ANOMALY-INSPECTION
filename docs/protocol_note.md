# Evaluation Protocol & Prediction Schema Specification (D-01, E-01)

Tài liệu này quy định giao thức thực nghiệm dùng chung cho cả 3 phương pháp (**Autoencoder**, **PatchCore**, **One-Class SVM**) trong dự án **Topic 2: Automated Visual Quality Inspection with Few Defect Samples**. Không thay đổi giao thức này sau khi bắt đầu chạy benchmark chính thức.

---

## 1. Lựa chọn Categories trên MVTec AD

Dự án chọn **5 categories cố định** đại diện cho đầy đủ các nhóm bề mặt và hình thái lỗi công nghiệp trong MVTec AD (tổng cộng 1.267 ảnh train normal và 584 ảnh test):

| Category | Nhóm sản phẩm | Train (`good`) | Val (`good`, 20%) | Test (`good` / `defect`) | Đặc trưng kỹ thuật & Lý do chọn |
|---|---|---:|---:|---:|---|
| `bottle` | Object (cứng, đối xứng) | 167 | 42 | 20 / 63 | Lỗi vỡ cạnh lớn/nhỏ và nhiễm bẩn (`broken_large`, `broken_small`, `contamination`). |
| `carpet` | Texture (vải dệt) | 224 | 56 | 28 / 89 | Lỗi kết cấu bề mặt (`color`, `cut`, `hole`, `metal_contamination`, `thread`). |
| `leather` | Texture (da thuộc) | 196 | 49 | 32 / 92 | Lỗi vết cắt, thủng, gấp nếp tinh vi trên nền đồng nhất (`cut`, `fold`, `glue`, `poke`). |
| `screw` | Object (chi tiết nhỏ, xoay) | 256 | 64 | 41 / 119 | Category khó: vật thể xoay nhiều góc, vết xước ở đầu/ren ốc rất nhỏ (`scratch_head`, `thread_side`). |
| `transistor` | Object (linh kiện điện tử) | 170 | 43 | 60 / 40 | Lỗi cấu trúc/vị trí (`misplaced`, `bent_lead`, `cut_lead`, `damaged_case`) — thách thức các phương pháp cục bộ. |

---

## 2. Tiền xử lý ảnh và mặt nạ (Shared Preprocessing)

Được định nghĩa tại [configs/protocol.yaml](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/configs/protocol.yaml) và thực thi tại [`build_transforms`](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/src/data/mvtec.py#L101-L115):
- **Ảnh đầu vào (`image`):**
  1. `Resize(256, interpolation=BILINEAR)`
  2. `CenterCrop(224)` $\rightarrow$ kích thước chuẩn $[3, 224, 224]$
  3. `ToTensor()` đưa về $[0, 1]$
  4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (chuẩn ImageNet).
- **Mặt nạ Ground-Truth (`mask`):**
  1. `Resize(256, interpolation=NEAREST)` (bảo toàn nhãn nhị phân, không sinh giá trị trung gian ở biên)
  2. `CenterCrop(224)` $\rightarrow$ kích thước chuẩn $[1, 224, 224]$, nhị phân hóa $\{0.0, 1.0\}$.
  3. Ảnh normal (`defect_type == "good"`) có mặt nạ toàn giá trị $0$.

---

## 3. Quy tắc chia tập dữ liệu & Chống rò rỉ nhãn (Zero Test Leakage)

- **`train` (80% của `train/good`):** Chỉ chứa ảnh normal. Dùng để huấn luyện Autoencoder, xây dựng Memory Bank của PatchCore, hoặc fit Scaler/PCA/OC-SVM.
- **`val` (20% của `train/good`, cố định `seed: 42` theo từng category):** Chỉ chứa ảnh normal. Dùng duy nhất cho:
  1. Chọn best checkpoint của Autoencoder theo `val_loss`.
  2. Xác định ngưỡng phán quyết (`threshold`) ở phân vị thứ 95 (`P95`) của điểm bất thường trên tập `val`.
- **`test` (Official MVTec AD test split):** Chứa cả ảnh normal và defect. **Tuyệt đối không dùng tập `test` để chọn checkpoint, tinh chỉnh hyperparameter hay tối ưu threshold.**

> [!IMPORTANT]
> **Quy ước chọn Hyperparameter cho One-Class SVM (`OC-01`):**
> Vì tập `val` chỉ chứa ảnh normal (không có nhãn defect), không thể grid-search $(\nu, \gamma)$ theo AUROC/F1 trên `val`. Do đó, `gamma` được chọn bằng heuristic thống kê `'scale'` ($\gamma = \frac{1}{D \cdot \text{Var}(X)}$) chỉ phụ thuộc tập `train`, và $\nu = 0.05$ được neo trực tiếp theo tỷ lệ phần đuôi 5% của quy tắc ngưỡng `P95`.

---

## 4. Chuẩn đầu ra dự đoán (Shared Prediction Bundle Schema — `E-01`)

Mỗi lần chạy (`run_name`) lưu kết quả tại `artifacts/predictions/<run_name>/` gồm:
1. **`predictions.csv`:** Bảng gồm 6 cột bắt buộc (`REQUIRED_COLUMNS`):
   - `path`: Đường dẫn tương đối tới ảnh (ví dụ `bottle/test/broken_large/000.png`).
   - `category`: Tên category (`bottle`, `carpet`, ...).
   - `split`: Thuộc `val` hoặc `test`.
   - `label`: Nhãn thực tế (`0` = normal, `1` = defect). Kiểm tra tự động bắt buộc mọi dòng `split == 'val'` phải có `label == 0`.
   - `defect_type`: Loại lỗi cụ thể (`good`, `broken_large`, `scratch_head`, ...).
   - `score`: Điểm bất thường mức ảnh (`float`, **chiều điểm đồng nhất: điểm càng cao càng bất thường**).
2. **`pixel_maps.npz` (nếu phương pháp hỗ trợ localization — Autoencoder & PatchCore):**
   - `anomaly_maps`: Mảng `float32` kích thước `[N, 224, 224]` khớp thứ tự dòng với `predictions.csv`.
   - `gt_masks`: Mảng `uint8` kích thước `[N, 224, 224]` chứa giá trị $\{0, 1\}$.
3. **`metadata.json`:** Lưu thời gian huấn luyện (`total_fit_seconds`), độ trễ suy luận (`mean_infer_ms_per_image`), dung lượng mô hình (`mean_artifact_size_mb`) và cấu hình YAML.

---

## 5. Bộ chỉ số đánh giá (Evaluation Metrics)

- **Mức ảnh (Image-level — áp dụng cho cả 3 phương pháp):**
  - `Image AUROC`: Diện tích dưới đường cong ROC (không phụ thuộc ngưỡng).
  - `Precision`, `Recall`, `F1`, `FPR` (False Positive Rate), `Accuracy`: Tính tại ngưỡng $\tau_c = \text{Percentile}_{95}(\{s_i \mid i \in \text{val}_c\})$ của từng category $c$.
- **Mức điểm ảnh (Pixel-level — áp dụng cho Autoencoder và PatchCore):**
  - `Pixel AUROC`: AUROC tính trên toàn bộ điểm ảnh của tập test trong mỗi category.
  - `PRO (Per-Region Overlap)`: Chia từng vùng lỗi liên thông (connected component) trên mặt nạ ground-truth thành các vùng độc lập có trọng số ngang nhau, tích phân đường cong trung bình độ phủ vùng theo FPR của điểm ảnh nền trên đoạn $\text{FPR} \in [0, 0.3]$ rồi chuẩn hóa chia cho $0.3$.
