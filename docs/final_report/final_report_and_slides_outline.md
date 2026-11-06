# Final report (6 to 8 pages) and 15-minute talk: outline (F-02, F-03)

Khung nội dung. Số liệu và nhận định chỉ đưa vào sau khi có kết quả đầy đủ trong `results/metrics/` và `results/figures/`.

## 1. Report structure

1. Abstract (150 đến 200 từ): bối cảnh thiếu mẫu lỗi, giao thức so sánh 3 phương pháp trên 5 category, kết quả chính (nêu số từ bảng).
2. Introduction: vì sao phát hiện lỗi công nghiệp dùng phương pháp one-class; nêu đúng những gì nhóm đã làm.
3. Dataset and protocol: 5 category, tiền xử lý 224x224, chia train/val/test, quy tắc ngưỡng P95 trên val (`docs/protocol_note.md`).
4. Methods: Autoencoder, PatchCore, One-class (SVM), đúng theo `configs/` và `src/methods/`.
5. Results: bảng tổng thể và theo category; bảng pixel AUROC và PRO; gallery 4 cột `Input | GT | AE | PatchCore` gồm cả ca thành công lẫn thất bại.
6. Extensions: độ nhạy theo số ảnh normal (I-01), đánh đổi báo động giả (mô tả, không chọn ngưỡng bằng test), một cải tiến nhỏ đã khai báo trước (I-02).
7. Limitations and conclusion: hạn chế nêu theo bằng chứng (category khó, tập val nhỏ, ngưỡng cố định, giới hạn của so sánh).

## 2. Talk (15 minutes, 6 speakers)

| Thời lượng | Nội dung | Người trình bày |
|---|---|---|
| 0:00 đến 2:30 | Bài toán, dữ liệu, giao thức, nguyên tắc không dùng test để chọn ngưỡng | chờ chốt |
| 2:30 đến 5:00 | Autoencoder và One-class | chờ chốt |
| 5:00 đến 7:30 | PatchCore | chờ chốt |
| 7:30 đến 10:00 | Kết quả, phân tích lỗi, độ nhạy theo số mẫu | chờ chốt |
| 10:00 đến 12:30 | Định vị (pixel AUROC, PRO) và demo trực tiếp | chờ chốt |
| 12:30 đến 15:00 | Hạn chế, bài học, kết luận | chờ chốt |

Demo phải chạy trên checkpoint thật (`python demo/app.py`), không dùng ảnh hay số liệu giả.

## 3. Q&A checklist

Câu hỏi nên chuẩn bị. Phần trả lời chỉ viết sau khi có số liệu; các điểm kỹ thuật dưới đây là định nghĩa, không phải kết quả.

1. Vì sao chọn ngưỡng ở P95 của val? Val chỉ có ảnh normal nên không tối ưu được F1 trên val; P95 chấp nhận khoảng 5% báo động giả
   trên phân phối normal của val. Điền thêm: báo động giả thực tế trên test (cột `false_alarm_rate`) và bảng đánh đổi ở I-01.
2. Vì sao cần PRO khi đã có pixel AUROC? Pixel AUROC gộp mọi pixel nên vùng lỗi lớn lấn át vùng lỗi nhỏ; PRO tính độ phủ theo từng
   vùng liên thông rồi lấy trung bình đều giữa các vùng. Điền: một ví dụ từ dữ liệu của nhóm nếu có.
3. Khi giảm số ảnh normal, phương pháp nào giảm nhiều nhất? Chỉ trả lời bằng đường cong từ I-01.
4. Vì sao one-class không có bản đồ lỗi? Đặc trưng được gộp toàn cục (global pooling) nên mất thông tin vị trí.
5. Test có bị dùng để chọn gì không? Không: checkpoint chọn theo val loss, ngưỡng theo val, hyperparameter cố định trong config từ trước.
   Dẫn chứng: `docs/evaluation_schema.md` (hàm `percentile_threshold` từ chối split khác val).
