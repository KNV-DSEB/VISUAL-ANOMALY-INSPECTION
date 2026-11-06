"""Experiment scripts end to end on a tiny synthetic dataset."""
import pytest

from scripts.analyze_errors import analyze
from scripts.generate_galleries import build_galleries, choose_examples
from scripts.run_sensitivity import plot_sensitivity_curve, run_cost_analysis, run_sensitivity
from src.evaluation import Predictions, evaluate_method
from src.methods.runner import run_method
from tests.test_methods import CFGS

METHODS = ["autoencoder", "patchcore", "oneclass"]


def test_scripts_run_on_saved_predictions(toy):
    protocol, tmp = toy
    pred = tmp / "pred"
    for m in METHODS:
        run_method(m, CFGS[m], protocol, pred_dir=pred, device="cpu")

    rows = evaluate_method(pred, "patchcore", ["toy"], {"threshold_percentile": 95.0})
    assert rows[-1]["category"] == "overall"

    (gallery,) = build_galleries(pred, tmp / "figs", protocol)
    assert gallery.exists() and gallery.stat().st_size > 0

    breakdown, errors = analyze(pred, tmp / "metrics", protocol, METHODS, top_k=2)
    assert set(breakdown["method"]) == set(METHODS) and set(errors["method"]) == set(METHODS)
    assert (tmp / "metrics" / "error_cases.csv").exists()

    cost = run_cost_analysis(pred, protocol, {"false_alarm_cost": {"cost_ratios": [1, 5]}}, METHODS, tmp / "figs")
    assert {"cost_ratio_1", "cost_ratio_5"} <= set(cost.columns)
    assert set(cost["category"]) == {"toy", "mean"}
    assert (tmp / "figs" / "false_alarm_cost_patchcore.png").exists()


def test_gallery_heatmaps_share_one_scale():
    import numpy as np

    from src.utils import normalize_anomaly_map

    quiet = np.random.default_rng(0).uniform(0.0, 1.0, (8, 8))  # a normal image: low scores everywhere
    assert normalize_anomaly_map(quiet).max() == 1.0  # on its own it is stretched to look hot
    assert normalize_anomaly_map(quiet, vmin=0.0, vmax=10.0).max() <= 0.1  # on the shared scale it stays cool


def test_gallery_examples_follow_the_fixed_rule(toy):
    protocol, tmp = toy
    run_method("patchcore", CFGS["patchcore"], protocol, pred_dir=tmp / "pred", device="cpu")
    test = Predictions.load(tmp / "pred", "patchcore", "toy", "test")
    idx = choose_examples(test)
    labels = [int(test.labels[i]) for i in idx]
    assert labels[:3] == [1, 1, 1] and labels[3] == 0  # best, worst, median defect, then worst normal
    assert test.scores[idx[0]] >= test.scores[idx[2]] >= test.scores[idx[1]]
    assert test.scores[idx[3]] == max(test.scores[test.labels == 0])


def test_sensitivity_caps_only_the_train_split(toy):
    protocol, tmp = toy
    df = run_sensitivity(protocol, {"sensitivity": {"sample_counts": [3, None]}}, tmp / "sens",
                         ["patchcore"], None, "cpu", method_cfgs={"patchcore": CFGS["patchcore"]})
    overall = df[df["category"] == "overall"].sort_values("n_normal_sort")
    assert overall["sample_label"].tolist() == ["3", "full"]
    assert overall["n_test"].tolist() == [6, 6]  # test set is identical for every sample count
    fig = plot_sensitivity_curve(df, tmp / "figs" / "sens.png")
    assert fig.exists()
