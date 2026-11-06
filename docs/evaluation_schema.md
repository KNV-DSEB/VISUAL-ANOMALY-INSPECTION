# Evaluation schema (E-01)

Một implementation duy nhất (`src/evaluation/`) cho cả 3 method: Autoencoder, PatchCore, One-class.

## 1. File prediction mà mỗi method phải xuất

Mỗi `(method, category, split)` là một file `.npz`:

```
results/predictions/<method>/<category>_<split>.npz      # split in {val, test}
```

| Key | Shape | Bắt buộc | Ý nghĩa |
|---|---|---|---|
| `scores` | (N,) | có | Điểm bất thường mức ảnh. **Cao hơn = bất thường hơn.** Phải hữu hạn (không NaN/inf). |
| `labels` | (N,) | có | 0 = good, 1 = defective. |
| `paths` | (N,) | có | Đường dẫn tương đối của ảnh (không dùng đường dẫn cá nhân tuyệt đối). |
| `defect_types` | (N,) | không | Loại lỗi (dùng cho E-03). |
| `anomaly_maps` | (N,H,W) | không* | Heatmap sau preprocessing (224x224). Cao hơn = bất thường hơn. |
| `gt_masks` | (N,H,W) | không* | Mask 0/1, resize bằng nearest. |
| `meta` | scalar JSON | không | Config path, checkpoint id, thời gian inference... |

\* `anomaly_maps` và `gt_masks` phải đi cùng nhau. Method không có heatmap (ví dụ One-class mức ảnh) bỏ qua, pixel metrics sẽ là NaN.

Cách dùng trong code của các method:

```python
from src.evaluation import Predictions
Predictions(method="patchcore", category="bottle", split="test",
            scores=..., labels=..., paths=..., anomaly_maps=..., gt_masks=...).save("results/predictions")
```

`val` = 20% `train/good` (seed 42), chỉ chứa ảnh normal. `test` = tập test chính thức của MVTec AD.

## 2. Quy tắc threshold

- Threshold **per-category** (điểm số giữa các category không so sánh được với nhau).
- `threshold = percentile(scores của val normal, P)`, mặc định P95 (`configs/evaluation.yaml`).
- `percentile_threshold()` từ chối nếu split không phải `val` hoặc `val` có ảnh defect, nên không thể vô tình fit trên test.
- Dự đoán: `score >= threshold` là anomalous.
- Metric phụ thuộc threshold (precision, recall, F1, false-alarm rate) chỉ dùng threshold này. AUROC / pixel AUROC / PRO không phụ thuộc threshold.

## 3. Metric

| Metric | Mức | Định nghĩa |
|---|---|---|
| `image_auroc` | ảnh | AUROC của `scores` theo `labels` (test). |
| `precision`, `recall`, `f1` | ảnh | Tại threshold từ val. Mẫu số bằng 0 thì trả 0. |
| `false_alarm_rate` | ảnh | FP / (FP + TN) trên test, phục vụ phần mở rộng false-alarm cost. |
| `pixel_auroc` | pixel | AUROC trên mọi pixel của mọi ảnh test trong category. Có thể subsample (`pixel_auroc_max_pixels`). |
| `pro` | pixel | Per-Region Overlap, tích phân FPR trong [0, 0.3], chuẩn hóa về [0, 1]. FPR tính trên pixel của ảnh normal; vùng = thành phần liên thông 8-neighbour. |

`tp, fp, fn, tn` cũng được ghi để truy vết.

## 4. Output

`python scripts/evaluate.py --config configs/evaluation.yaml`

- `results/metrics/<method>.csv|json`: một dòng mỗi category + dòng `overall`.
- `results/metrics/comparison.csv`: gộp cả method.
- `overall` = macro average qua category (mỗi category trọng số như nhau); các cột đếm được cộng dồn; `threshold` để NaN.
- Thiếu file prediction thì báo lỗi (không bỏ qua im lặng), để E-02 không thiếu run.

## 5. Kiểm thử

`python -m pytest tests/test_evaluation.py` gồm các case có đáp án biết trước: AUROC hoàn hảo/ngược/hòa/tính tay, P/R/F1 tính tay, threshold từ chối test/defect, pixel AUROC và PRO hoàn hảo/ngược/hằng số, PRO phạt vùng nhỏ bị bỏ sót (pixel AUROC thì gần như không), round-trip lưu/đọc, validate schema.
