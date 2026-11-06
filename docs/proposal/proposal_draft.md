# Project Proposal — Topic 2: Automated Visual Quality Inspection with Few Defect Samples

---

## 1. Problem Statement & Objectives (P-01)
Trong dây chuyền sản xuất công nghiệp thực tế, nhà máy có thể thu thập hàng nghìn bức ảnh sản phẩm đạt chuẩn (**normal/good parts**), nhưng số lượng mẫu lỗi (**defective parts**) lại cực kỳ khan hiếm và hình thái lỗi mới có thể xuất hiện bất ngờ. Do đó, bài toán kiểm định chất lượng thị giác không thể giải quyết bằng phân loại có giám sát truyền thống mà phải đặt dưới góc nhìn **Unsupervised / One-Class Anomaly Detection & Localization** (chỉ huấn luyện trên ảnh normal).

**Mục tiêu đo lường được của dự án:**
1. Xây dựng và so sánh công bằng **3 hướng tiếp cận đại diện** trên cùng một giao thức dữ liệu MVTec AD:
   - **Hướng 1 (Reconstruction-based):** Convolutional Autoencoder (`AE`).
   - **Hướng 2 (Patch feature memory):** PatchCore-style Memory Bank kèm k-Center Greedy Coreset (`PC`).
   - **Hướng 3 (Classical one-class boundary):** Pretrained CNN Multi-scale Features + One-Class SVM (`OC`).
2. Đánh giá khả năng **định vị vết lỗi mức điểm ảnh (Pixel-level Localization)** giữa Autoencoder và PatchCore qua `Pixel AUROC` và `PRO`.
3. Khảo sát **độ nhạy khi số lượng mẫu huấn luyện normal giảm dần** ($N \in \{10, 25, 50, 100, \text{full}\}$) và **phân tích chi phí báo động giả (False-alarm cost analysis)** để đề xuất điểm vận hành thực tế kèm ứng dụng Demo Gradio.

---

## 2. Dataset & Shared Evaluation Protocol (P-01)
- **Dataset:** Sử dụng 5 categories từ bộ dữ liệu chuẩn **MVTec AD** (`bottle`, `carpet`, `leather`, `screw`, `transistor`) gồm 1.267 ảnh train normal và 584 ảnh test (bao phủ cả nhóm vật thể cứng, bề mặt kết cấu dệt/da, vết xước siêu nhỏ và lỗi sai lệch cấu trúc).
- **Tiền xử lý chung:** Resize $256 \times 256$ (bilinear cho ảnh, nearest-neighbor cho mask), Center-crop $224 \times 224$, chuẩn hóa theo ImageNet mean/std.
- **Chia tập dữ liệu chống rò rỉ (Zero Test Leakage):**
  - `train` (80% `train/good`): Chỉ dùng để fit mô hình.
  - `val` (20% `train/good`, `seed=42`): Chỉ chứa ảnh normal; dùng để chọn best checkpoint của Autoencoder và xác định ngưỡng phán quyết tại phân vị thứ 95 (`P95`).
  - `test` (Official test set): Chỉ dùng 1 lần để báo cáo kết quả cuối cùng.

---

## 3. Methodology Overview (P-01)
1. **Convolutional Autoencoder (`AutoencoderMethod`):**
   - Kiến trúc Encoder–Decoder đối xứng 4 tầng nén không gian ảnh $224 \to 14$ (`latent_channels=128`), nhánh Decoder dùng `Bilinear Upsampling + Conv2d` để tránh nhiễu ô bàn cờ.
   - Huấn luyện với hàm mất mát kết hợp $\mathcal{L} = \text{MSE} + 0.5 \cdot L_1$. Bản đồ lỗi pixel được làm mượt Gaussian ($\sigma=2.0$); điểm bất thường mức ảnh tính bằng trung bình top 2% pixel có sai số tái tạo lớn nhất (`topk_mean`).
2. **PatchCore Feature Memory (`PatchCoreMethod`):**
   - Trích xuất đặc trưng cục bộ đa tầng từ `layer2` và `layer3` của mạng pretrained `WideResNet50-2` (hoặc `ResNet18`), gom lân cận bằng `AvgPool2d(3x3)` và chiếu ngẫu nhiên bảo toàn khoảng cách về chiều $D=384$.
   - Nén ngân hàng bộ nhớ bằng thuật toán **k-Center Greedy Coreset** giữ lại 10% patch đại diện. Điểm mức ảnh được tính từ khoảng cách k-NN có trọng số mật độ (PatchCore Eq. 6); bản đồ khoảng cách patch được nội suy lên $224 \times 224$ và làm mượt Gaussian ($\sigma=4.0$).
3. **One-Class Classification (`OneClassMethod`):**
   - Trích xuất vector đặc trưng toàn cục (ghép Global Average + Max Pooling từ `layer2`, `layer3`, `layer4`), chuẩn hóa `StandardScaler`, giảm chiều bằng `PCA` (giữ 95% phương sai) và huấn luyện `OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')`.
   - Điểm bất thường được định nghĩa là $s(x) = -f_{\text{decision}}(x)$.

---

## 4. Evaluation Plan, Team Ownership & Timeline (P-02)
- **Chỉ số đánh giá cốt lõi:**
  - *Mức ảnh (3 methods):* `Image AUROC`, `F1`, `Precision`, `Recall`, `FPR` (tại ngưỡng Validation `P95`), thời gian suy luận (`ms/image`) và kích thước artifact (`MB`).
  - *Mức điểm ảnh (AE vs. PatchCore):* `Pixel AUROC` và `PRO` (tích phân trên $\text{FPR} \in [0, 0.3]$).
- **Các mở rộng (Extensions):** (1) So sánh trực quan Localization Gallery; (2) Đường cong suy giảm hiệu năng khi số mẫu normal giảm về 10 mẫu; (3) Tối ưu ngưỡng vận hành theo hàm chi phí bất đối xứng $C = c_{\text{FP}}\text{FP} + c_{\text{FN}}\text{FN}$.

**Phân công nhiệm vụ cho 6 thành viên:**

| Thành viên | Vai trò chính (Ownership) | Các đầu việc phụ trách |
|---|---|---|
| **Khánh** | Team Lead & Integration | Điều phối chung, tích hợp repo (`F-01`, `F-04`), đối chiếu bảng tổng hợp (`E-02`, `I-02`). |
| **Vũ** | Data EDA & Autoencoder | Thống kê EDA (`D-02`), cài đặt & huấn luyện Autoencoder (`AE-01`, `AE-02`). |
| **Hoàng** | Data Pipeline & PatchCore | Thiết kế DataLoader/Protocol (`D-01`, `D-02`), cài đặt PatchCore & Coreset (`PC-01`, `PC-02`). |
| **Phong** | One-Class & Reproducibility | Cài đặt One-Class SVM (`OC-01`, `OC-02`), quản lý configs, tài liệu tái lập (`F-01`). |
| **Kiên** | Evaluation & Metrics Lead | Xây dựng bộ khung đánh giá, chỉ số PRO, phân tích lỗi & độ nhạy (`E-01`, `E-03`, `I-01`). |
| **Huy** | Localization & Demo UI | Đánh giá định vị lỗi, xuất Heatmap Gallery (`L-01`, `L-02`) và xây dựng Demo Gradio (`DM-01`). |
