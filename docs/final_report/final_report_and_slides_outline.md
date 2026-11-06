# Final Report (6–8 Pages) & 15-Minute Presentation Guide — With Official Experimental Numbers (F-02, F-03)

Tài liệu này tổng hợp đầy đủ **số liệu thực nghiệm chính thức** từ thư mục [results/metrics/](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/main_comparison_table.csv) và [results/figures/](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/figures/sample_sensitivity_curve.png) để nhóm đưa trực tiếp vào bài báo cáo cuối kỳ 6–8 trang và bộ Slide thuyết trình 15 phút.

---

## Phần 1: Các Bảng Số Liệu Chính Thức Dùng Cho Báo Cáo & Slide

### 1. Bảng so sánh tổng thể (Main Comparison — E-02 & I-02)
*(Trung bình trên 5 categories: `bottle`, `carpet`, `leather`, `screw`, `transistor` tại ngưỡng chuẩn Validation P95)*

| Method | Image AUROC $\uparrow$ | F1 (P95) $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | FPR $\downarrow$ | Pixel AUROC $\uparrow$ | Pixel PRO $\uparrow$ | Latency (ms/img) $\downarrow$ | Artifact (MB/cat) $\downarrow$ | Fit Time (s) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoencoder** | 0.6823 | 0.5212 | 0.8097 | 0.4734 | 0.3295 | 0.7916 | 0.6133 | **47.41** | **3.40** | 2410.60 |
| **One-Class SVM** | 0.8046 | 0.7149 | 0.8022 | 0.7007 | 0.1972 | — | — | 52.74 | 9.48 | **51.30** |
| **PatchCore (Baseline)** | **0.9702** | **0.8692** | 0.9441 | **0.8296** | 0.1057 | **0.9820** | 0.9291 | 56.99 | 15.66 | 310.53 |
| **PatchCore (Refined — I-02)** | 0.9674 | 0.8519 | **0.9462** | 0.7974 | **0.0896** | 0.9819 | **0.9298** | 59.89 | 29.92 | 566.11 |

### 2. Bảng độ nhạy theo số lượng mẫu huấn luyện Normal (Sample-Count Sensitivity — I-01)
*(Nguồn: [results/metrics/sample_sensitivity.csv](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/sample_sensitivity.csv) & [results/figures/sample_sensitivity_curve.png](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/figures/sample_sensitivity_curve.png))*

| Số mẫu Normal / Category ($N$) | AE Image AUROC | AE Pixel PRO | OC-SVM Image AUROC | PatchCore Image AUROC | PatchCore Pixel AUROC | PatchCore Pixel PRO |
|---|---:|---:|---:|---:|---:|---:|
| **$N = 10$ mẫu** | 0.6126 | 0.4609 | 0.8115 | **0.8793** | **0.9662** | **0.8862** |
| **$N = 25$ mẫu** | 0.6401 | 0.5122 | 0.7954 | **0.9045** | **0.9764** | **0.9072** |
| **$N = 50$ mẫu** | 0.6516 | 0.5259 | 0.8009 | **0.9229** | **0.9776** | **0.9155** |
| **$N = 100$ mẫu** | 0.6879 | 0.5630 | 0.8061 | **0.9434** | **0.9815** | **0.9274** |
| **Full ($\approx 167\text{–}256$ mẫu)** | 0.6770 | 0.6059 | 0.8046 | **0.9702** | **0.9820** | **0.9291** |

> [!TIP]
> **Insight quan trọng cho Báo cáo & Thuyết trình:**
> Ngay cả khi chỉ có **10 bức ảnh normal mỗi category ($N=10$)**, `PatchCore` vẫn đạt **`Image AUROC = 0.8793`**, **`Pixel AUROC = 0.9662`** và **`Pixel PRO = 0.8862`** — vượt xa `Autoencoder` được huấn luyện trên toàn bộ dữ liệu (`Image AUROC = 0.6823`, `Pixel PRO = 0.6133`)! Lý do là 10 bức ảnh normal đã cung cấp $10 \times 28 \times 28 = 7.840$ patch cục bộ có đặc trưng tiền huấn luyện giàu ngữ nghĩa.

### 3. Bảng phân tích chi phí Báo động giả vs. Bỏ lọt lỗi trên PatchCore (False-Alarm Cost Trade-off — I-01)
*(Nguồn: [results/metrics/false_alarm_cost.csv](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/false_alarm_cost.csv) — Trung bình trên 5 categories)*

| Validation Percentile | Mean Threshold | FPR (False Alarm) $\downarrow$ | Defect Recall (TPR) $\uparrow$ | Precision $\uparrow$ | Mean F1 $\uparrow$ | Cost ($C_{\text{FN}}/C_{\text{FP}} = 1$) $\downarrow$ | Cost ($C_{\text{FN}}/C_{\text{FP}} = 5$) $\downarrow$ | Cost ($C_{\text{FN}}/C_{\text{FP}} = 10$) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **P80.0** | 6.7057 | 29.35% | **95.80%** | 88.32% | 0.9152 | 0.1155 | **0.2410** | **0.3979** |
| **P85.0** | 6.8136 | 24.58% | 95.40% | 89.99% | 0.9229 | 0.1044 | 0.2417 | 0.4134 |
| **P90.0 (Tối ưu F1)** | 7.0390 | 18.24% | 94.73% | 92.59% | **0.9332** | **0.0904** | 0.2478 | 0.4445 |
| **P92.5** | 7.1574 | 13.65% | 90.88% | 94.02% | 0.9191 | 0.0998 | 0.3441 | 0.6496 |
| **P95.0 (Baseline)** | 7.3661 | 10.57% | 82.96% | 94.41% | 0.8692 | 0.1372 | 0.5624 | 1.0939 |
| **P97.5** | 7.7094 | 6.74% | 73.19% | 96.17% | 0.7914 | 0.1844 | 0.8459 | 1.6728 |
| **P99.0** | 7.9488 | **2.54%** | 69.06% | **97.26%** | 0.7516 | 0.1998 | 0.9636 | 1.9184 |

---

## Phần 2: Cấu trúc Báo cáo cuối kỳ 6–8 trang (Conference Format)

1. **Abstract (~150–200 từ):**
   - Nêu bài toán kiểm định ngoại quan công nghiệp khi hiếm mẫu lỗi; trình bày giao thức đánh giá thống nhất trên 5 categories MVTec AD cho 3 phương pháp (`Autoencoder`, `PatchCore`, `One-Class SVM`); tóm tắt kết quả định lượng: `PatchCore` dẫn đầu với `Image AUROC = 0.9702`, `Pixel AUROC = 0.9820`, `Pixel PRO = 0.9291` và duy trì `Pixel PRO = 0.8862` ngay cả khi chỉ dùng 10 ảnh normal/category.
2. **Section 1: Introduction & Problem Formulation (0.75 trang)**
3. **Section 2: Dataset & Experimental Protocol (0.75 trang)**
4. **Section 3: Compared Anomaly Detection Methods (1.5 trang)**
5. **Section 4: Quantitative Results & Localization Analysis (1.75 trang):**
   - Đưa Bảng 1, Bảng 2, Bảng 3 và nhúng các hình Gallery từ `results/figures/gallery_*.png`.
6. **Section 5: Extensions — Sample Sensitivity, Cost Analysis & Refinement (1.5 trang):**
   - Đưa Bảng độ nhạy ít mẫu + hình `sample_sensitivity_curve.png` và Bảng chi phí báo động giả + hình `false_alarm_cost_patchcore.png`.
   - Phân tích cấu hình tinh chỉnh `patchcore_refined` (giúp tăng F1 trên `screw` từ `0.6667` $\rightarrow$ `0.7128` và giảm FPR trung bình từ `10.57%` xuống `8.96%`) cùng hệ thống Demo Gradio 3 Tab.
7. **Section 6: Limitations & Conclusion (0.5 trang)**

---

## Phần 3: Kịch bản Slide thuyết trình 15 phút cho 6 thành viên (F-03)

| Thời lượng | Người trình bày | Nội dung Slide | Điểm nhấn số liệu cần nói rõ |
|---|---|---|---|
| **0:00 – 2:30** | **1. Vũ** | **Bài toán, Dữ liệu MVTec AD & Giao thức Đánh giá** | 5 categories (1.267 ảnh train/val normal, 584 ảnh test); tách 20% `val` chọn ngưỡng `P95`, chống rò rỉ dữ liệu test 100%. |
| **2:30 – 5:00** | **2. Vũ & Phong** | **Autoencoder & One-Class SVM** | Autoencoder (`Image AUROC = 0.6823`, nhanh nhất `47.4 ms`, nhẹ nhất `3.4 MB` nhưng yếu trên `carpet`); One-Class SVM (`Image AUROC = 0.8046`, fit chỉ `51s`, đạt `1.000` trên `leather` nhưng yếu trên `screw`). |
| **5:00 – 7:30** | **3. Hoàng** | **PatchCore & Tinh chỉnh `I-02`** | `PatchCore` đạt `Image AUROC = 0.9702`, `bottle = 1.000`; bản `patchcore_refined` (tăng coreset 20%, $k=5$) nâng F1 trên category khó nhất `screw` từ `0.6667` lên `0.7128` và giảm FPR xuống `8.96%`. |
| **7:30 – 10:00** | **4. Kiên** | **Phân tích Lỗi, Độ nhạy ít mẫu & Chi phí (`I-01`)** | Khi giảm về $N=10$ ảnh normal, PatchCore vẫn đạt `PRO = 0.8862`; phân tích chi phí chỉ ra hạ ngưỡng xuống `P90` giúp nâng Mean F1 từ `0.8692` lên `0.9332` và giảm chi phí kỳ vọng xuống thấp nhất. |
| **10:00 – 12:30** | **5. Huy** | **Định vị vùng lỗi (`Pixel AUROC` vs. `PRO`) & Live Demo Gradio** | PatchCore đạt `Pixel AUROC = 0.9820`, `PRO = 0.9291` (vượt AE `+31.6% PRO`); chiếu Gallery 4 cột và **demo trực tiếp giao diện Gradio 3 Tab**. |
| **12:30 – 15:00** | **6. Khánh** | **Hạn chế, Bài học Kỹ thuật & Kết luận** | Tổng kết đánh đổi Độ chính xác – Tốc độ – Dung lượng giữa 3 phương pháp, nêu hạn chế khi vật thể xoay (`screw`) và mở đầu Q&A. |
