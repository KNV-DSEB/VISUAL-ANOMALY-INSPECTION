"""Error analysis and false-alarm cost on hand-checkable predictions."""
import numpy as np
import pytest

from src.evaluation import (
    Predictions,
    analyze_false_alarm_cost,
    breakdown_by_defect_type,
    find_error_cases,
    percentile_threshold,
)


def _preds():
    # val scores 0.10..0.50 -> P95 threshold = 0.48
    val = Predictions("m", "c", "val", scores=np.linspace(0.1, 0.5, 20), labels=np.zeros(20),
                      paths=[f"v{i}" for i in range(20)])
    good = np.linspace(0.1, 0.55, 10)  # 0.50 and 0.55 are above 0.48 -> 2 false alarms
    defects = np.array([0.6, 0.7, 0.8, 0.85, 0.9, 0.62, 0.72, 0.95, 0.2, 0.3])  # last two are missed
    types = ["crack"] * 5 + ["hole"] * 5
    test = Predictions("m", "c", "test", scores=np.r_[good, defects], labels=np.r_[np.zeros(10), np.ones(10)],
                       paths=[f"t{i}" for i in range(20)], defect_types=["good"] * 10 + types)
    return val, test


def test_threshold_is_p95_of_val():
    val, _ = _preds()
    assert percentile_threshold(val, 95.0) == pytest.approx(0.48)


def test_false_alarm_cost_hand_computed():
    val, test = _preds()
    df = analyze_false_alarm_cost(val, test, cost_ratios=(1.0, 5.0), percentiles=(95.0,))
    r = df.iloc[0]
    assert (r.tp, r.fp, r.fn, r.tn) == (8, 2, 2, 8)
    assert r.fpr == pytest.approx(0.2) and r.recall == pytest.approx(0.8) and r.precision == pytest.approx(0.8)
    assert r.cost_ratio_5 == pytest.approx((2 + 5 * 2) / 20) and r.cost_ratio_1 == pytest.approx(4 / 20)


def test_higher_percentile_never_raises_false_alarms():
    val, test = _preds()
    df = analyze_false_alarm_cost(val, test, percentiles=(80.0, 90.0, 95.0, 99.0))
    assert df["fpr"].is_monotonic_decreasing or (df["fpr"].diff().dropna() <= 0).all()
    assert (df["threshold"].diff().dropna() >= 0).all()


def test_breakdown_by_defect_type():
    val, test = _preds()
    b = breakdown_by_defect_type(val, test).set_index("defect_type")
    assert b.loc["crack", "flag_rate"] == pytest.approx(1.0)
    assert b.loc["hole", "flag_rate"] == pytest.approx(3 / 5)
    assert b.loc["good", "flag_rate"] == pytest.approx(0.2) and b.loc["good", "role"] == "normal_false_alarm"


def test_find_error_cases_picks_most_confident_errors():
    val, test = _preds()
    e = find_error_cases(val, test, top_k=1).set_index("outcome")
    assert e.loc["FP", "score"] == pytest.approx(0.55)  # most confident false alarm
    assert e.loc["FN", "score"] == pytest.approx(0.2)   # most clearly missed defect
    assert set(e.index) == {"FP", "FN", "TP", "TN"}


def test_analysis_rejects_mismatched_inputs():
    val, test = _preds()
    with pytest.raises(ValueError, match="same method"):
        analyze_false_alarm_cost(val, Predictions("other", "c", "test", test.scores, test.labels, test.paths))
    with pytest.raises(ValueError, match="test split"):
        analyze_false_alarm_cost(val, val)
