# Final Report (6–8 Pages) & 15-Minute Presentation Guide (F-02, F-03)

---

## Phần 1: Cấu trúc Báo cáo cuối kỳ 6–8 trang (Conference Format — IEEE / CVPR 2 cột)

1. **Abstract (~150–200 từ):**
   - Bối cảnh kiểm định chất lượng công nghiệp khi thiếu mẫu lỗi; mô tả ngắn gọn giao thức so sánh 3 phương pháp (Autoencoder, PatchCore, One-Class SVM) trên 5 categories của MVTec AD; tóm tắt kết quả định lượng nổi bật nhất (Image AUROC, Pixel PRO, độ nhạy ít mẫu và phân tích chi phí).
2. **Section 1: Introduction & Problem Formulation (0.75 trang):**
   - Tại sao bài toán phát hiện lỗi công nghiệp phải giải bằng One-Class / Unsupervised learning.
   - Đóng góp thực nghiệm của nhóm: (1) Bộ khung đánh giá công bằng chống rò rỉ dữ liệu test; (2) Đối chiếu 3 họ phương pháp về cả độ chính xác, định vị điểm ảnh và tốc độ/dung lượng; (3) Nghiên cứu suy giảm hiệu năng khi số mẫu normal giảm về 10 mẫu và bài toán tối ưu chi phí báo động giả.
3. **Section 2: Dataset & Experimental Protocol (0.75 trang):**
   - Bảng thống kê 5 categories (`bottle`, `carpet`, `leather`, `screw`, `transistor`), quy trình tiền xử lý $224 \times 224$, cách chia `train` (80% normal) / `val` (20% normal) / `test` và quy tắc chọn ngưỡng `P95`.
4. **Section 3: Compared Anomaly Detection Methods (1.5 trang):**
   - Sơ đồ khối kiến trúc chung.
   - *3.1 Convolutional Autoencoder:* Cấu trúc mạng, hàm loss $\text{MSE} + 0.5 L_1$, cơ chế gom điểm `topk_mean`.
   - *3.2 PatchCore Feature Memory:* Trích xuất patch đa tầng (`layer2`, `layer3`), thuật toán k-Center Greedy Coreset và điểm k-NN có trọng số mật độ.
   - *3.3 One-Class SVM:* Đặc trưng toàn cục đa tầng (`avg + max` pooling), chuẩn hóa `StandardScaler + PCA(0.95)` và biên quyết định RBF.
5. **Section 4: Quantitative Results & Localization Analysis (1.75 trang):**
   - Bảng so sánh tổng thể (`main_comparison_table.csv`) và phân tích từng category.
   - Bảng định vị điểm ảnh (`pixel_localization_table.csv` — `Pixel AUROC` & `PRO`) kèm Hình Gallery 4 cột (`Input | GT Mask | AE | PatchCore`) chỉ rõ ca thành công và ca thất bại (`screw`, `transistor`).
6. **Section 5: Extensions — Sample Sensitivity, Cost Analysis & Refinement (1.5 trang):**
   - *5.1 Normal-Sample Count Sensitivity (`I-01`):* Đồ thị suy giảm `Image AUROC` và `Pixel PRO` tại $N \in \{10, 25, 50, 100, \text{full}\}$. Phân tích vì sao PatchCore giữ vững hiệu năng ở $N=10, 25$ tốt hơn Autoencoder.
   - *5.2 Asymmetric False-Alarm Cost Analysis (`I-01`):* Đường cong đánh đổi $\text{FPR}$ vs. $\text{Recall}$ và cách chọn ngưỡng tối ưu theo tỷ lệ chi phí $C_{\text{FN}}/C_{\text{FP}} \in \{1, 5, 10\}$.
   - *5.3 Method Refinement (`I-02`) & Interactive Gradio System (`DM-01`):* So sánh `patchcore` baseline với `patchcore_refined` và minh họa giao diện Demo 3 Tab.
7. **Section 6: Limitations & Conclusion (0.5 trang):**
   - Hạn chế thực tế (vật thể xoay góc trên `screw`, lỗi sai vị trí logic trên `transistor`, ngưỡng `P95` cố định ~5% FPR khi tập `val` nhỏ) và kết luận.

---

## Phần 2: Kịch bản Slide thuyết trình 15 phút cho 6 thành viên (F-03)

| Thời lượng | Người trình bày | Nội dung Slide | Điểm nhấn cần nói rõ |
|---|---|---|---|
| **0:00 – 2:30** | **1. Vũ** | **Bài toán, Dữ liệu MVTec AD & Giao thức Đánh giá** | Nêu bài toán thiếu mẫu lỗi trong nhà máy; lý do chọn 5 categories đại diện; nhấn mạnh nguyên tắc **không dùng tập Test để chọn threshold**. |
| **2:30 – 5:00** | **2. Vũ & Phong** | **Phương pháp 1 (Autoencoder) & Phương pháp 3 (One-Class SVM)** | Giải thích trực quan cách Autoencoder phát hiện lỗi qua sai số tái tạo (`topk_mean`) và cách One-Class SVM bao bọc miền đặc trưng normal. |
| **5:00 – 7:30** | **3. Hoàng** | **Phương pháp 2 (PatchCore) & Tinh chỉnh `I-02`** | Giải thích vì sao đặc trưng cục bộ cấp độ patch (`layer2+layer3`) + nén Coreset k-Center Greedy giúp bắt được cả vết xước nhỏ mà vẫn chạy nhanh. |
| **7:30 – 10:00** | **4. Kiên** | **Kết quả Thực nghiệm, Phân tích Lỗi & Độ nhạy ít mẫu (`I-01`)** | Trình bày bảng so sánh 3 methods; chỉ ra sự suy giảm khi giảm số mẫu normal xuống 10 mẫu và bài toán tối ưu chi phí báo động giả. |
| **10:00 – 12:30** | **5. Huy** | **Định vị vùng lỗi (Pixel AUROC vs. PRO) & Live Demo Gradio** | Giải thích ý nghĩa chỉ số `PRO`; chiếu Gallery 4 cột và **mở trực tiếp Demo Gradio trên Hugging Face Spaces** bấm thử 1–2 ảnh. |
| **12:30 – 15:00** | **6. Khánh** | **Hạn chế, Bài học Kỹ thuật & Kết luận** | Tổng kết khi nào nên dùng mỗi phương pháp trong thực tế công nghiệp, nêu thẳng thắn các ca lỗi khó (`screw`, `transistor`) và mở đầu phiên Q&A. |

---

## Phần 3: Bộ câu hỏi phản biện dự kiến từ giảng viên (Q&A Checklist)

1. **Hỏi:** *Tại sao các em chọn ngưỡng ở phân vị thứ 95 (`P95`) trên tập Validation? Điều này dẫn tới hệ quả gì trên tập Test?*
   - **Trả lời:** Vì trong bài toán One-Class, tập `val` chỉ chứa ảnh normal nên không thể tối ưu F1 trên `val`. Chọn `P95` đồng nghĩa với việc chấp nhận tỷ lệ báo động giả kỳ vọng trên phân phối normal là $\approx 5\%$. Trong mục `I-01`, nhóm đã mở rộng phân tích trên dải phân vị từ `P80` đến `P99.5` gắn với tỷ lệ chi phí $C_{\text{FN}}/C_{\text{FP}}$ để nhà máy chọn ngưỡng phù hợp.
2. **Hỏi:** *Tại sao phải tính thêm chỉ số `PRO` trong khi đã có `Pixel AUROC`?*
   - **Trả lời:** Trên cùng một category, một vết bẩn lớn có thể chiếm 5.000 pixels trong khi một vết xước nhỏ chỉ chiếm 50 pixels. `Pixel AUROC` gộp chung mọi pixel nên mô hình đoán đúng vết lỗi lớn mà bỏ sót hoàn toàn vết xước nhỏ vẫn đạt `Pixel AUROC > 0.95`. Ngược lại, `PRO` tính độ phủ trên từng vùng liên thông rồi lấy trung bình đều giữa các vùng, nên phản ánh trung thực khả năng bắt lỗi nhỏ.
3. **Hỏi:** *Khi số lượng ảnh normal giảm xuống còn 10 hoặc 25 ảnh (`I-01`), phương pháp nào sụt giảm mạnh nhất và tại sao?*
   - **Trả lời:** Autoencoder sụt giảm mạnh hơn vì phải huấn luyện các tầng tích chập từ đầu để học tái tạo ảnh nên dễ bị overfitting khi chỉ có 10–25 ảnh. Ngược lại, PatchCore tận dụng bộ trích xuất đặc trưng đã tiền huấn luyện trên ImageNet (không cần train lại trọng số mạng) và so khớp trực tiếp ở cấp độ patch cục bộ (10 ảnh vẫn cung cấp hàng nghìn patch normal), nên giữ độ ổn định cao hơn nhiều.
