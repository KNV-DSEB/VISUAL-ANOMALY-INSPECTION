# Milestone Report (Week 10) — Baseline Results, What Broke & Revised Plan (MS-01, MS-02)

> [!NOTE]
> Sau khi chạy xong `notebooks/colab_runner.ipynb` trên Google Colab, bạn chỉ cần chép số liệu từ `results/metrics/main_comparison_table.csv` và `results/metrics/pixel_localization_table.csv` vào Bảng 1 và Bảng 2 dưới đây.

---

## 1. Baseline Quantitative Results (MS-01)

### Bảng 1: So sánh tổng thể 3 phương pháp mức ảnh và chi phí suy luận (Mean trên 5 categories)
*(Nguồn dữ liệu: `results/metrics/main_comparison_table.csv`)*

| Method | Image AUROC $\uparrow$ | F1 (Val P95) $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | FPR $\downarrow$ | Pixel AUROC $\uparrow$ | Pixel PRO $\uparrow$ | Latency (ms/img) $\downarrow$ | Artifact Size (MB) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoencoder** | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` |
| **PatchCore** | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` |
| **One-Class SVM** | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | N/A | N/A | `[từ CSV]` | `[từ CSV]` |

### Bảng 2: Kết quả định vị lỗi mức điểm ảnh theo từng Category (L-01)
*(Nguồn dữ liệu: `results/metrics/pixel_localization_table.csv`)*

| Category | AE Pixel AUROC | AE Pixel PRO | PatchCore Pixel AUROC | PatchCore Pixel PRO | Nhận xét chính |
|---|---:|---:|---:|---:|---|
| `bottle` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | Vật thể cứng, vùng lỗi rõ ràng |
| `carpet` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | Kết cấu sợi dệt làm nhiễu AE reconstruction |
| `leather` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | Vết châm kim/cắt nhỏ trên nền da |
| `screw` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | Lỗi rất nhỏ ở ren/đầu ốc, vật thể xoay góc |
| `transistor` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | Lỗi đặt lệch vị trí (`misplaced`) diện tích lớn |
| **Mean** | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | `[từ CSV]` | PatchCore vượt trội nhờ đặc trưng tiền huấn luyện |

---

## 2. What Broke & How We Fixed It (MS-01)

1. **Vấn đề 1 — Autoencoder bị "pha loãng" điểm bất thường trên các vết lỗi nhỏ (`screw`, `leather`):**
   - *Hiện tượng:* Khi lấy trung bình sai số tái tạo trên toàn bộ $224 \times 224 = 50.176$ điểm ảnh, các vết xước nhỏ chỉ chiếm $< 1\%$ diện tích bị nhiễu nền lấn át, khiến `Image AUROC` trên `screw` giảm sâu.
   - *Cách khắc phục:* Chuyển hàm gom điểm mức ảnh của Autoencoder sang `topk_mean` (trung bình 2% điểm ảnh có sai số tái tạo cao nhất) kết hợp bộ lọc làm mượt Gaussian ($\sigma=2.0$) và dùng `Bilinear Upsampling` ở Decoder để khử nhiễu ô bàn cờ.
2. **Vấn đề 2 — Tràn bộ nhớ (OOM) khi tính ma trận khoảng cách Coreset và chỉ số PRO:**
   - *Hiện tượng:* Tập hợp toàn bộ patch từ 250 ảnh normal tạo ra $> 150.000$ vector; nếu tính ma trận khoảng cách đầy đủ $N \times N$ hoặc lưu toàn bộ ngưỡng 3D khi tính `PRO` sẽ gây tràn RAM trên máy cấu hình thấp.
   - *Cách khắc phục:* Cài đặt thuật toán **k-Center Greedy** cập nhật khoảng cách nhỏ nhất cuốn chiếu với bộ nhớ $O(N)$, giảm chiều ngẫu nhiên cố định seed (`proj_dim=384`) và tính chỉ số `PRO` bằng tìm kiếm nhị phân (`np.searchsorted`) trên mảng 1 chiều đã sắp xếp.
3. **Vấn đề 3 — Chiều điểm số ngược của `OneClassSVM` và thiếu nhãn lỗi trên tập Validation:**
   - *Hiện tượng:* Hàm `decision_function` của `scikit-learn` trả giá trị dương cho mẫu bình thường và âm cho mẫu bất thường (ngược chiều với AE và PatchCore). Ngoài ra tập `val` chỉ có ảnh normal nên không thể tune $(\nu, \gamma)$ bằng F1.
   - *Cách khắc phục:* Đảo dấu $s(x) = -\text{decision\_function}(x)$, dùng `StandardScaler + PCA(0.95)` để ổn định khoảng cách RBF với `gamma='scale'` và cố định $\nu = 0.05$ đồng bộ với quy tắc ngưỡng `P95`.

---

## 3. Revised Plan for Weeks 10–15 (MS-02)

| Tuần | Đầu việc | Người phụ trách | Đầu ra nghiệm thu |
|---|---|---|---|
| **Tuần 11–12** | Hoàn thiện Gallery định vị lỗi (`L-02`) & Thí nghiệm độ nhạy số lượng mẫu normal + False-alarm cost (`I-01`) | Kiên, Huy, Hoàng | `results/figures/gallery_*.png`, `sample_sensitivity.csv`, `false_alarm_cost.csv` |
| **Tuần 13** | Tinh chỉnh cấu hình mạnh nhất `patchcore_refined` (`I-02`) & Kiểm thử Demo Gradio 3 Tab (`DM-01`) | Khánh, Hoàng, Huy | `configs/experiments/patchcore_refined.yaml`, `demo/app.py` trên Hugging Face Spaces |
| **Tuần 14** | Đóng băng mã nguồn (`F-01`) & Viết báo cáo cuối kỳ 6–8 trang theo chuẩn hội nghị (`F-02`) | Phong, Vũ, Khánh | Bản thảo đầy đủ 6–8 trang + Repo sạch vượt qua 100% `pytest` |
| **Tuần 15** | Hoàn thiện Slide 15 phút, tập dượt thuyết trình 6 thành viên & Nộp bài (`F-03`, `F-04`) | Cả nhóm (6 người) | Slide deck, Demo live link, Final PDF & GitHub release tag |
