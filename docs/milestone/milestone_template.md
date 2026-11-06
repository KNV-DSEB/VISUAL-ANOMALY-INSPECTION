# Milestone Report (Week 10) — Baseline Results, What Broke & Revised Plan (MS-01, MS-02)

---

## 1. Baseline Quantitative Results (MS-01)

### Bảng 1: So sánh tổng thể 3 phương pháp mức ảnh và chi phí suy luận (Mean trên 5 categories)
*(Nguồn dữ liệu: [results/metrics/main_comparison_table.csv](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/main_comparison_table.csv))*

| Run / Method | Image AUROC $\uparrow$ | F1 (Val P95) $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | FPR $\downarrow$ | Accuracy $\uparrow$ | Pixel AUROC $\uparrow$ | Pixel PRO $\uparrow$ | Latency (ms/img) $\downarrow$ | Mean Artifact (MB) $\downarrow$ | Total Fit (s) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoencoder (`autoencoder`)** | 0.6823 | 0.5212 | 0.8097 | 0.4734 | 0.3295 | 0.5682 | 0.7916 | 0.6133 | **47.41** | **3.40** | 2410.60 |
| **One-Class SVM (`one_class`)** | 0.8046 | 0.7149 | 0.8022 | 0.7007 | 0.1972 | 0.7453 | N/A | N/A | 52.74 | 9.48 | **51.30** |
| **PatchCore (`patchcore`)** | **0.9702** | **0.8692** | 0.9441 | **0.8296** | 0.1057 | **0.8628** | **0.9820** | 0.9291 | 56.99 | 15.66 | 310.53 |
| **PatchCore Refined (`patchcore_refined`)** | 0.9674 | 0.8519 | **0.9462** | 0.7974 | **0.0896** | 0.8558 | 0.9819 | **0.9298** | 59.89 | 29.92 | 566.11 |

### Bảng 2: Phân rã Image AUROC & F1 (tại ngưỡng Validation P95) theo từng Category
*(Nguồn dữ liệu: `results/metrics/*_per_category.csv`)*

| Category | AE Image AUROC | AE F1 | OC-SVM Image AUROC | OC-SVM F1 | PatchCore Image AUROC | PatchCore F1 | PatchCore Refined AUROC | PatchCore Refined F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bottle` | 0.8897 | 0.8393 | 0.9623 | 0.9593 | **1.0000** | **1.0000** | **1.0000** | 0.9921 |
| `carpet` | 0.4438 | 0.4308 | 0.9516 | 0.9198 | 0.9575 | **0.9333** | **0.9595** | 0.9249 |
| `leather` | 0.7687 | 0.8465 | **1.0000** | **0.9892** | 0.9976 | 0.9684 | 0.9973 | 0.9735 |
| `screw` | 0.7649 | 0.1260 | 0.4036 | 0.1727 | **0.9324** | 0.6667 | 0.9258 | **0.7128** |
| `transistor` | 0.5446 | 0.3636 | 0.7054 | 0.5333 | **0.9638** | **0.7778** | 0.9546 | 0.6563 |
| **Mean** | 0.6823 | 0.5212 | 0.8046 | 0.7149 | **0.9702** | **0.8692** | 0.9674 | 0.8519 |

### Bảng 3: Kết quả định vị lỗi mức điểm ảnh theo từng Category (L-01)
*(Nguồn dữ liệu: [results/metrics/pixel_localization_table.csv](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/pixel_localization_table.csv))*

| Category | AE Pixel AUROC | AE Pixel PRO | PatchCore Pixel AUROC | PatchCore Pixel PRO | PatchCore Refined Pixel AUROC | PatchCore Refined Pixel PRO | Nhận xét thực nghiệm |
|---|---:|---:|---:|---:|---:|---:|---|
| `bottle` | 0.7926 | 0.5594 | **0.9822** | **0.9195** | 0.9813 | 0.9164 | PatchCore khoanh chính xác cả vết nứt lớn và nhỏ |
| `carpet` | 0.5865 | 0.3196 | **0.9835** | **0.9090** | 0.9828 | 0.9086 | Kết cấu sợi dệt làm nhiễu nặng sai số tái tạo của AE |
| `leather` | 0.9634 | 0.8997 | **0.9903** | **0.9649** | 0.9895 | 0.9644 | Nền da đồng nhất nên cả AE và PatchCore đều định vị tốt |
| `screw` | 0.9479 | 0.8077 | 0.9828 | 0.9156 | **0.9841** | **0.9227** | `patchcore_refined` (coreset 20%, $k=5$) cải thiện PRO trên vết xước nhỏ |
| `transistor` | 0.6675 | 0.4803 | 0.9713 | 0.9366 | **0.9720** | **0.9371** | Lỗi lệch chân/vị trí linh kiện cần đặc trưng ngữ cảnh tầng sâu |
| **Mean** | 0.7916 | 0.6133 | **0.9820** | 0.9291 | 0.9819 | **0.9298** | PatchCore vượt AE **+19.04% Pixel AUROC** và **+31.58% Pixel PRO** |

---

## 2. What Broke & How We Fixed It (MS-01)

1. **Vấn đề 1 — Autoencoder bị thất bại trên bề mặt kết cấu phức tạp (`carpet` Image AUROC = 0.4438, Pixel PRO = 0.3196) và vết lỗi diện tích nhỏ (`screw` Recall = 6.72% tại P95):**
   - *Hiện tượng:* Mạng Autoencoder cố gắng tái tạo chi tiết tần số cao của sợi vải dệt (`carpet`) dẫn tới sai số tái tạo trên nền bình thường cũng rất cao, gây nhầm lẫn với vết lỗi. Trên `screw`, mặc dù `Pixel AUROC` đạt `0.9479`, nhưng do vết xước đầu ốc rất nhỏ nên điểm mức ảnh bị sát ngưỡng `P95`.
   - *Cách khắc phục:* Đã áp dụng hàm gom điểm `topk_mean` (trung bình 2% pixel có lỗi cao nhất) kèm làm mượt Gaussian ($\sigma=2.0$) và dùng `Bilinear Upsampling` ở Decoder; đồng thời chứng minh định lượng sự vượt trội của đặc trưng cục bộ tiền huấn luyện (PatchCore đạt `0.9575` trên `carpet` và `0.9324` trên `screw`).
2. **Vấn đề 2 — One-Class SVM hoạt động rất tốt trên bề mặt đồng nhất (`leather` AUROC = 1.000, `bottle` = 0.9623, `carpet` = 0.9516) nhưng sụp đổ trên `screw` (AUROC = 0.4036):**
   - *Hiện tượng:* `OneClassMethod` sử dụng Global Average + Max Pooling toàn ảnh. Khi chi tiết ốc vít (`screw`) xoay ở các góc độ khác nhau và vết xước chỉ chiếm vài chục pixel, phép gộp toàn cục làm mất thông tin vị trí cục bộ.
   - *Cách khắc phục:* Kết quả này là bằng chứng thực nghiệm đắt giá giải thích vì sao cần giữ lại lưới đặc trưng cục bộ cấp độ patch (`PatchCore`) thay vì gộp toàn cục (`One-Class SVM`).
3. **Vấn đề 3 — Độ lệch giữa ngưỡng phân vị `P95` trên tập Validation nhỏ (42–64 mẫu) và tỷ lệ Recall trên `screw`:**
   - *Hiện tượng:* Tại ngưỡng mặc định `P95`, `PatchCore` đạt `Precision = 98.36%`, `FPR = 2.44%` nhưng `Recall = 50.42%` trên `screw` (F1 = `0.6667`).
   - *Cách khắc phục:* Trong phân tích chi phí `I-01` ([results/metrics/false_alarm_cost.csv](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/results/metrics/false_alarm_cost.csv)), khi dịch điểm vận hành của PatchCore xuống `P90`, `Recall` trên `screw` tăng vọt lên **78.15%** (`F1 = 0.8651`), nâng `Mean F1` toàn bộ 5 categories từ `0.8692` lên **0.9332** với chi phí kỳ vọng thấp nhất!

---

## 3. Revised Plan for Weeks 10–15 (MS-02)

| Tuần | Đầu việc | Người phụ trách | Trạng thái |
|---|---|---|---|
| **Tuần 11–12** | Hoàn thiện Gallery định vị lỗi (`L-02`) & Thí nghiệm độ nhạy số lượng mẫu normal + False-alarm cost (`I-01`) | Kiên, Huy, Hoàng | **Đã hoàn thành** (`results/figures/`, `results/metrics/`) |
| **Tuần 13** | Tinh chỉnh cấu hình mạnh nhất `patchcore_refined` (`I-02`) & Kiểm thử Demo Gradio 3 Tab (`DM-01`) | Khánh, Hoàng, Huy | **Đã hoàn thành** (`patchcore_refined` tăng F1 trên `screw` từ `0.6667` $\rightarrow$ `0.7128` và giảm Mean FPR từ `10.57%` $\rightarrow$ `8.96%`) |
| **Tuần 14** | Đóng băng mã nguồn (`F-01`) & Viết báo cáo cuối kỳ 6–8 trang theo chuẩn hội nghị (`F-02`) | Phong, Vũ, Khánh | Đang hoàn thiện văn bản từ bộ bảng/hình đã chốt |
| **Tuần 15** | Hoàn thiện Slide 15 phút, tập dượt thuyết trình 6 thành viên & Nộp bài (`F-03`, `F-04`) | Cả nhóm (6 người) | Sẵn sàng |
