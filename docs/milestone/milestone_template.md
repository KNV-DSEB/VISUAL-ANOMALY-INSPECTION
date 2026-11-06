# Milestone report template (MS-01, MS-02): baseline results, what broke, revised plan

Khung để điền sau khi chạy đầy đủ trên Colab. Mọi số liệu lấy từ file trong `results/metrics/`; không điền số từ trí nhớ và
không viết kết luận trước khi có bảng. Giới hạn 3 trang.

## 1. Baseline results (MS-01)

Nguồn: `results/metrics/comparison.csv` (sinh bởi `python scripts/evaluate.py`). Trung bình macro trên 5 category; bảng theo category
nằm trong cùng file.

| Method | Image AUROC | Precision | Recall | F1 | False-alarm rate | Pixel AUROC | PRO | ms/ảnh | Checkpoint (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | | | | | | | | | |
| PatchCore | | | | | | | | | |
| One-class | | | n/a | n/a | | n/a | n/a | | |

(Thời gian suy luận và kích thước lấy từ `meta` của file dự đoán.) Pixel AUROC và PRO chỉ có cho Autoencoder và PatchCore.

Theo category: bảng 5 dòng x 3 method cho image AUROC và (AE, PatchCore) pixel AUROC, PRO. Nêu category khó nhất của từng method
dựa trên bảng, không dựa trên ví dụ chọn tay.

## 2. What broke (MS-01)

Chỉ ghi vấn đề đã thực sự xảy ra khi chạy, mỗi vấn đề một mục:

| Vấn đề (hiện tượng đo được) | Bằng chứng (file, log, hình) | Nguyên nhân | Cách xử lý và kết quả sau xử lý |
|---|---|---|---|
| | | | |

Mục cần có: lỗi pipeline, category khó, báo động giả/bỏ lọt (từ `scripts/analyze_errors.py`), giới hạn của từng method.

## 3. Revised plan (MS-02)

Bảng task còn lại từ tuần 10 đến tuần 15, lấy từ bảng timeline (task, người phụ trách, hạn, đầu ra). Ưu tiên deliverable của môn;
không mở thêm phạm vi lớn.

| Tuần | Task | Người phụ trách | Đầu ra |
|---|---|---|---|
| | | | |
