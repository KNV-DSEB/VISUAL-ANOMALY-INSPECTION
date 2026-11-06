# Proposal draft: Topic 2, Automated Visual Quality Inspection with Few Defect Samples (P-01, P-02)

Bản nháp để cả nhóm review, giới hạn 2 trang khi nộp. Nội dung phải khớp code và config trong repo; chỉ mô tả những gì đã làm.

## 1. Problem and objectives (P-01)

Trong sản xuất công nghiệp, ảnh sản phẩm đạt chuẩn rất nhiều nhưng ảnh lỗi hiếm và hình thái lỗi mới có thể xuất hiện bất ngờ.
Vì vậy bài toán được đặt dưới dạng phát hiện bất thường (one-class): chỉ huấn luyện trên ảnh normal.

Mục tiêu đo được:
1. So sánh công bằng 3 hướng tiếp cận trên MVTec AD, cùng category, tiền xử lý, quy tắc chọn ngưỡng và code metric:
   Autoencoder (tái tạo), PatchCore (bộ nhớ đặc trưng patch), One-class (đặc trưng pretrained + One-Class SVM).
2. Đo định vị lỗi mức pixel (pixel AUROC, PRO) cho các phương pháp tạo được bản đồ bất thường.
3. Mở rộng: độ nhạy khi giảm số ảnh normal dùng để huấn luyện, và phân tích đánh đổi báo động giả.

## 2. Data and protocol (P-01)

- MVTec AD, 5 category: `bottle`, `carpet`, `leather`, `screw`, `transistor` (1.267 ảnh train normal, 584 ảnh test).
- Tiền xử lý chung: resize 256, center-crop 224, chuẩn hóa ImageNet; mask resize nearest.
- `train` = 80% `train/good`; `val` = 20% (seed 42, chỉ ảnh normal) dùng cho chọn checkpoint và ngưỡng; `test` chính thức chỉ để báo cáo.
- Chi tiết: `docs/protocol_note.md`, `configs/protocol.yaml`.

## 3. Methods (P-01)

1. Autoencoder tích chập nhỏ (encoder 4 tầng stride 2, latent 128 kênh 14x14, decoder upsample bilinear + conv). Loss MSE + 0.5 L1.
   Bản đồ lỗi tái tạo được làm mượt Gaussian; điểm ảnh là trung bình 2% pixel lỗi lớn nhất. Checkpoint chọn theo val loss.
2. PatchCore: WideResNet50-2 pretrained (trọng số IMAGENET1K_V1, ghim theo tên), đặc trưng `layer2` + `layer3`, gộp lân cận 3x3, chiếu ngẫu nhiên
   cố định seed xuống 384 chiều, bộ nhớ nén bằng k-center greedy coreset (tối đa 10.000 patch), điểm k-NN có hiệu chỉnh mật độ.
3. One-class: đặc trưng ResNet pretrained (trung bình toàn cục `layer2` + `layer3`, chuẩn hóa L2 và standardize) + One-Class SVM RBF,
   `nu` và `gamma` cố định từ trước (`configs/oneclass.yaml`). Chỉ có điểm mức ảnh.

## 4. Evaluation plan (P-02)

- Mức ảnh (cả 3 phương pháp): image AUROC; precision, recall, F1, false-alarm rate tại ngưỡng P95 của `val`.
- Mức pixel (AE và PatchCore): pixel AUROC và PRO (tích phân FPR trong [0, 0.3]).
- Báo cáo tổng thể và theo từng category; thời gian suy luận và kích thước checkpoint.
- Mở rộng tách khỏi phần lõi: sample-count sensitivity, false-alarm trade-off (mô tả, không chọn ngưỡng bằng test), một cải tiến nhỏ cho
  phương pháp tốt nhất được khai báo trước.

## 5. Team and timeline (P-02)

Phân công theo từng task: xem bảng timeline. *(Chờ chốt người phụ trách; điền bảng ngắn 6 dòng vào đây khi đã duyệt.)*

| Thành viên | Vai trò | Task |
|---|---|---|
| Khánh | Lead, tích hợp | chờ chốt |
| Vũ | Dữ liệu, Autoencoder | chờ chốt |
| Hoàng | Pipeline dữ liệu, PatchCore | chờ chốt |
| Phong | One-class, tái lập | chờ chốt |
| Kiên | Đánh giá | chờ chốt |
| Huy | Định vị, demo | chờ chốt |
