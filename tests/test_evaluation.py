"""Sanity checks for the evaluation harness using small cases with known answers."""
import numpy as np
import pytest

from src.evaluation import (
    Predictions,
    evaluate_category,
    evaluate_method,
    image_auroc,
    percentile_threshold,
    pixel_auroc,
    pro_score,
    threshold_metrics,
)


# ------------------------------------------------------------ image level
def test_image_auroc_perfect_inverted_and_tie():
    y = [0, 0, 1, 1]
    assert image_auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert image_auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert image_auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(image_auroc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_image_auroc_hand_computed():
    # positives 0.8, 0.4 ; negatives 0.6, 0.2 -> 3 of 4 pairs ordered correctly
    assert image_auroc([1, 1, 0, 0], [0.8, 0.4, 0.6, 0.2]) == pytest.approx(0.75)


def test_threshold_metrics_hand_computed():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.4, 0.9, 0.5, 0.3, 0.7]
    m = threshold_metrics(labels, scores, threshold=0.5)  # pred>=0.5
    # pred = 0,0,1,0,1,1,0,1 -> tp=3 fp=1 fn=1 tn=3
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_threshold_metrics_no_positive_predictions():
    m = threshold_metrics([0, 1], [0.1, 0.2], threshold=10.0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ------------------------------------------------------------- thresholds
def _val(scores, labels=None, split="val"):
    n = len(scores)
    return Predictions("m", "c", split, scores, labels if labels is not None else [0] * n,
                       [f"p{i}" for i in range(n)])


def test_percentile_threshold_matches_numpy():
    s = np.arange(100, dtype=float)
    assert percentile_threshold(_val(s), 95) == pytest.approx(np.percentile(s, 95))


def test_threshold_refuses_test_split_or_defects():
    with pytest.raises(ValueError):
        percentile_threshold(_val([0.1, 0.2], split="test"))
    with pytest.raises(ValueError):
        percentile_threshold(_val([0.1, 0.9], labels=[0, 1]))


# ------------------------------------------------------------ pixel level
def _toy_maps(perfect=True):
    gt = np.zeros((4, 16, 16), dtype=np.uint8)
    gt[2, 2:6, 2:6] = 1  # defect A
    gt[3, 8:12, 8:12] = 1  # defect B
    rng = np.random.default_rng(0)
    maps = rng.uniform(0.0, 0.3, gt.shape)
    if perfect:
        maps[gt > 0] = rng.uniform(0.7, 1.0, (gt > 0).sum())
    else:
        maps = 1.0 - maps  # inverted-ish
        maps[gt > 0] = rng.uniform(0.0, 0.2, (gt > 0).sum())
    return maps, gt


def test_pixel_auroc_and_pro_perfect():
    maps, gt = _toy_maps(True)
    assert pixel_auroc(maps, gt) == 1.0
    assert pro_score(maps, gt) == pytest.approx(1.0, abs=1e-6)


def test_pixel_auroc_and_pro_inverted():
    maps, gt = _toy_maps(False)
    assert pixel_auroc(maps, gt) == 0.0
    assert pro_score(maps, gt) == pytest.approx(0.0, abs=1e-6)


def test_pro_constant_map_equals_half_of_fpr_limit_free_area():
    # constant map -> at any threshold FPR is 0 or 1, overlap is 0 or 1.
    # Within FPR <= 0.3 only the "nothing flagged" point is reachable => PRO = 0.
    gt = np.zeros((2, 8, 8), dtype=np.uint8)
    gt[1, 2:4, 2:4] = 1
    maps = np.full(gt.shape, 0.5)
    assert pro_score(maps, gt) == pytest.approx(0.0, abs=1e-6)


def test_pro_weights_small_regions_equally():
    # big region fully found, small region fully missed -> PRO ~ 0.5
    gt = np.zeros((2, 32, 32), dtype=np.uint8)
    gt[1, 0:10, 0:10] = 1  # big (100 px)
    gt[1, 20:22, 20:22] = 1  # small (4 px)
    maps = np.zeros(gt.shape)
    maps[1, 0:10, 0:10] = 1.0
    pro = pro_score(maps, gt)
    assert pro == pytest.approx(0.5, abs=0.02)
    # pixel AUROC barely notices the missed small region
    assert pixel_auroc(maps, gt) > 0.95


def test_pro_counts_false_alarms_inside_defective_images():
    # The defect is found, but only together with a large "halo" of false alarms in the same image.
    # FPR counts every non-defect pixel of every image, so the halo is penalised (PRO well below 1);
    # measuring FPR on anomaly-free images only would ignore it and give PRO = 1.
    gt = np.zeros((2, 16, 16), dtype=np.uint8)
    gt[1, 6:10, 6:10] = 1
    maps = np.zeros(gt.shape)
    maps[1, gt[1] > 0] = 1.0
    halo = np.zeros((16, 16), dtype=bool)
    halo[:8, :15] = True
    halo &= gt[1] == 0
    maps[1, halo] = 1.0
    assert 0.0 < pro_score(maps, gt) < 0.7


def test_pixel_metrics_nan_without_defects():
    gt = np.zeros((2, 4, 4), dtype=np.uint8)
    maps = np.random.default_rng(0).uniform(size=gt.shape)
    assert np.isnan(pixel_auroc(maps, gt))
    assert np.isnan(pro_score(maps, gt))


# ------------------------------------------------------ harness / schema
def _make_pair(method="m", category="bottle", with_maps=True):
    val = Predictions(method, category, "val", [0.1, 0.2, 0.3, 0.4, 0.5],
                      [0] * 5, [f"v{i}" for i in range(5)])
    maps, gt = _toy_maps(True)
    labels = (gt.reshape(4, -1).any(axis=1)).astype(int)
    scores = maps.reshape(4, -1).max(axis=1)
    test = Predictions(method, category, "test", scores, labels, [f"t{i}" for i in range(4)],
                       anomaly_maps=maps if with_maps else None,
                       gt_masks=gt if with_maps else None)
    return val, test


def test_evaluate_category_uses_val_threshold():
    val, test = _make_pair()
    row = evaluate_category(val, test, {"threshold_percentile": 95})
    assert row["threshold"] == pytest.approx(np.percentile(val.scores, 95))
    assert row["image_auroc"] == 1.0
    assert row["pixel_auroc"] == 1.0
    assert (row["n_good"], row["n_defect"]) == (2, 2)
    # both defect images score ~0.7-1.0 >= 0.48 -> recall 1; good images ~<0.3 -> no false alarm
    assert row["recall"] == 1.0 and row["false_alarm_rate"] == 0.0


def test_evaluate_category_without_maps_gives_nan_pixel():
    val, test = _make_pair(with_maps=False)
    row = evaluate_category(val, test)
    assert np.isnan(row["pixel_auroc"]) and np.isnan(row["pro"])


def test_evaluate_category_rejects_swapped_splits():
    val, test = _make_pair()
    with pytest.raises(ValueError):
        evaluate_category(test, val)


def test_roundtrip_and_overall(tmp_path):
    for cat in ("bottle", "carpet"):
        v, t = _make_pair(category=cat)
        v.save(tmp_path)
        t.save(tmp_path)
    rows = evaluate_method(tmp_path, "m", ["bottle", "carpet"], {})
    assert [r["category"] for r in rows] == ["bottle", "carpet", "overall"]
    assert rows[-1]["image_auroc"] == pytest.approx(1.0)
    assert rows[-1]["n_test"] == 8


def test_missing_prediction_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_method(tmp_path, "m", ["bottle"], {})


def test_schema_validation():
    with pytest.raises(ValueError):
        Predictions("m", "c", "test", [0.1, np.nan], [0, 1], ["a", "b"])
    with pytest.raises(ValueError):
        Predictions("m", "c", "test", [0.1, 0.2], [0, 2], ["a", "b"])
    with pytest.raises(ValueError):
        Predictions("m", "c", "train", [0.1], [0], ["a"])
    with pytest.raises(ValueError):  # maps without masks
        Predictions("m", "c", "test", [0.1], [0], ["a"], anomaly_maps=np.zeros((1, 4, 4)))
