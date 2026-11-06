"""Inspection API and Gradio callbacks, run against real (tiny) trained checkpoints."""
from functools import partial

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

import demo.app as app  # noqa: E402
from scripts import predict_single as ps  # noqa: E402
from scripts.run_sensitivity import run_cost_analysis  # noqa: E402
from src.evaluation import evaluate_method  # noqa: E402
from src.evaluation.harness import write_table  # noqa: E402
from src.methods.runner import run_method  # noqa: E402
from tests.test_methods import CFGS  # noqa: E402

METHODS = ["autoencoder", "patchcore", "oneclass"]


@pytest.fixture
def trained(toy):
    """Tiny checkpoints for all three methods plus paths to a defective and a normal toy image."""
    protocol, tmp = toy
    for m in METHODS:
        run_method(m, CFGS[m], protocol, pred_dir=tmp / "pred", ckpt_dir=tmp / "ckpt", device="cpu")
    ps._MODEL_CACHE.clear()
    root = tmp / "mvtec" / "toy" / "test"
    return protocol, tmp, root / "crack" / "000.png", root / "good" / "000.png"


def test_inspect_uses_the_checkpoint_and_its_validation_threshold(trained):
    protocol, tmp, defect, _ = trained
    res = ps.inspect_single_image(defect, "toy", "patchcore", protocol, tmp / "ckpt", device="cpu")
    assert res["verdict"] in {"NORMAL", "DEFECT"} and not res["threshold_is_override"]
    assert res["threshold"] == res["val_threshold"]
    assert res["heatmap_rgb"].shape == (32, 32, 3) and res["supports_localization"]
    assert (res["verdict"] == "DEFECT") == (res["score"] >= res["val_threshold"])

    oc = ps.inspect_single_image(defect, "toy", "oneclass", protocol, tmp / "ckpt", device="cpu")
    assert oc["heatmap_rgb"] is None and not oc["supports_localization"]


def test_manual_threshold_override_and_normal_has_no_contours(trained):
    protocol, tmp, defect, _ = trained
    high = ps.inspect_single_image(defect, "toy", "patchcore", protocol, tmp / "ckpt",
                                   custom_threshold=1e9, device="cpu")
    assert high["verdict"] == "NORMAL" and high["threshold_is_override"]
    np.testing.assert_array_equal(high["contour_rgb"], high["preprocessed_rgb"])
    assert high["binary_mask_rgb"].sum() == 0
    low = ps.inspect_single_image(defect, "toy", "patchcore", protocol, tmp / "ckpt",
                                  custom_threshold=-1e9, device="cpu")
    assert low["verdict"] == "DEFECT" and low["binary_mask_rgb"].sum() > 0


def test_errors_are_explicit(trained):
    protocol, tmp, defect, _ = trained
    kw = dict(protocol=protocol, checkpoint_dir=tmp / "ckpt", device="cpu")
    with pytest.raises(ValueError, match="Unsupported category"):
        ps.inspect_single_image(defect, "spaceship", "patchcore", **kw)
    with pytest.raises(ValueError, match="Unsupported method"):
        ps.inspect_single_image(defect, "toy", "yolo", **kw)
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        ps.inspect_single_image(tmp / "missing.png", "toy", "patchcore", **kw)
    with pytest.raises(ValueError, match="too small"):
        ps.inspect_single_image(np.zeros((8, 8, 3), np.uint8), "toy", "patchcore", **kw)
    with pytest.raises(FileNotFoundError, match="Train one with"):  # no fallback model
        ps.inspect_single_image(defect, "toy", "patchcore", protocol, tmp / "no_checkpoints", device="cpu")


def test_compare_reports_missing_checkpoints_instead_of_faking(toy):
    protocol, tmp = toy
    run_method("patchcore", CFGS["patchcore"], protocol, pred_dir=tmp / "p", ckpt_dir=tmp / "c", device="cpu")
    ps._MODEL_CACHE.clear()
    img = tmp / "mvtec" / "toy" / "test" / "crack" / "000.png"
    df, outputs = ps.compare_all_methods_on_image(img, "toy", protocol, tmp / "c")
    assert list(df["Method"]) == METHODS and set(outputs) == {"patchcore"}
    assert df.set_index("Method").loc["autoencoder", "Verdict"] == "no checkpoint"


def test_gradio_callbacks(trained, monkeypatch):
    protocol, tmp, defect, _ = trained
    monkeypatch.setattr(app, "inspect_single_image",
                        partial(ps.inspect_single_image, protocol=protocol, checkpoint_dir=tmp / "ckpt", device="cpu"))
    monkeypatch.setattr(app, "compare_all_methods_on_image",
                        partial(ps.compare_all_methods_on_image, protocol=protocol, checkpoint_dir=tmp / "ckpt"))
    image = Image.open(defect)

    banner, contour, heatmap, mask = app.run_tab1_inspection(image, "toy", "patchcore", False, 0.5, 0.45, 0.92)
    assert ("PASS" in banner or "REJECT" in banner) and heatmap is not None
    assert "validation:" in banner
    banner, *_ = app.run_tab1_inspection(image, "toy", "patchcore", True, 0.5, 0.45, 0.92)
    assert "manual override" in banner

    empty, c, h, m = app.run_tab1_inspection(None, "toy", "patchcore", False, 0.5, 0.45, 0.92)
    assert "upload" in empty and c is None
    table, in_vis, ae_vis, pc_vis = app.run_tab2_arena(image, "toy", 0.45)
    assert len(table) == 3 and in_vis is not None and ae_vis is not None and pc_vis is not None
    with pytest.raises(gr.Error):
        app.run_tab2_arena(None, "toy", 0.45)


def test_missing_checkpoint_is_shown_as_an_error_not_a_verdict(toy, monkeypatch):
    protocol, tmp = toy
    ps._MODEL_CACHE.clear()
    monkeypatch.setattr(app, "inspect_single_image",
                        partial(ps.inspect_single_image, protocol=protocol, checkpoint_dir=tmp / "none", device="cpu"))
    image = Image.open(tmp / "mvtec" / "toy" / "test" / "crack" / "000.png")
    banner, contour, heatmap, mask = app.run_tab1_inspection(image, "toy", "patchcore", False, 0.5, 0.45, 0.92)
    assert "No trained checkpoint" in banner and "REJECT" not in banner and "PASS" not in banner
    assert contour is None and heatmap is None and mask is None


def test_result_tables_come_from_saved_metrics(trained, monkeypatch):
    protocol, tmp, *_ = trained
    metrics, figures = tmp / "metrics", tmp / "figures"
    monkeypatch.setattr(app, "METRICS_DIR", metrics)
    monkeypatch.setattr(app, "FIGURES_DIR", figures)

    overall, pixel = app.load_comparison_tables()  # nothing generated yet
    assert "chưa có kết quả" in overall.iloc[0, 0] and "evaluate.py" in overall.iloc[0, 0]
    table, fig = app.load_cost_view("patchcore")
    assert "chưa có kết quả" in table.iloc[0, 0] and fig is None

    rows = [r for m in METHODS for r in evaluate_method(tmp / "pred", m, ["toy"], {"threshold_percentile": 95.0})]
    write_table(rows, metrics / "comparison.csv")
    cost = run_cost_analysis(tmp / "pred", protocol, {"false_alarm_cost": {"cost_ratios": [1, 5]}}, METHODS, figures)
    cost.to_csv(metrics / "false_alarm_cost.csv", index=False)

    overall, pixel = app.load_comparison_tables()
    assert set(overall["method"]) == set(METHODS)
    assert set(pixel["method"]) == {"autoencoder", "patchcore"}  # image-level one-class has no pixel metrics
    table, fig = app.load_cost_view("patchcore")
    assert set(table["method"]) == {"patchcore"} and fig is not None


def test_app_builds():
    assert isinstance(app.build_demo_app(), gr.Blocks)
