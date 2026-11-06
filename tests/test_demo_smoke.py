"""Unit and smoke tests for Chặng 4: Single-image CLI & Gradio 3-Tab Demo UI (DM-01)."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from demo.app import (
    build_demo_app,
    run_tab1_inspection,
    run_tab2_arena,
    run_tab3_cost_simulator,
)
from scripts.predict_single import compare_all_methods_on_image, inspect_single_image


def test_inspect_single_image_and_error_handling(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    rgb = rng.integers(50, 200, size=(256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "sample.png"
    Image.fromarray(rgb).save(img_path)

    res_pc = inspect_single_image(
        image_input=img_path,
        category="bottle",
        method_name="patchcore",
        artifacts_dir=tmp_path / "empty_artifacts",
        allow_demo_fallback=True,
    )
    assert res_pc["verdict"] in {"NORMAL", "DEFECT"}
    assert res_pc["heatmap_rgb"] is not None
    assert res_pc["contour_rgb"].shape == (224, 224, 3)

    res_oc = inspect_single_image(
        image_input=rgb,
        category="carpet",
        method_name="one_class",
        artifacts_dir=tmp_path / "empty_artifacts",
        allow_demo_fallback=True,
    )
    assert res_oc["heatmap_rgb"] is None
    assert not res_oc["supports_localization"]

    # Error handling checks (DM-01 acceptance criteria)
    with pytest.raises(ValueError, match="Unsupported category"):
        inspect_single_image(rgb, category="spaceship", allow_demo_fallback=True)

    with pytest.raises(ValueError, match="Unsupported method"):
        inspect_single_image(rgb, category="bottle", method_name="yolo", allow_demo_fallback=True)

    with pytest.raises(FileNotFoundError):
        inspect_single_image(tmp_path / "missing.png", category="bottle", allow_demo_fallback=True)

    with pytest.raises(ValueError, match="too small"):
        inspect_single_image(np.zeros((8, 8, 3), dtype=np.uint8), category="bottle", allow_demo_fallback=True)


def test_gradio_callbacks_and_app_builder() -> None:
    rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
    rgb[80:140, 80:140] = 250

    banner, contour, heatmap, mask = run_tab1_inspection(
        image=rgb,
        category="bottle",
        method_name="patchcore",
        use_custom_threshold=False,
        custom_threshold=0.5,
        alpha=0.45,
        hotspot_quantile=0.92,
    )
    assert "PASS" in banner or "REJECT" in banner
    assert contour is not None and heatmap is not None and mask is not None

    # Empty image gracefully handled without crash
    empty_banner, c_none, h_none, m_none = run_tab1_inspection(
        None, "bottle", "patchcore", False, 0.5, 0.45, 0.92
    )
    assert "Vui lòng upload" in empty_banner
    assert c_none is None

    df_arena, in_vis, ae_vis, pc_vis = run_tab2_arena(rgb, "bottle", 0.45)
    assert len(df_arena) == 3
    assert in_vis is not None and ae_vis is not None and pc_vis is not None

    sim_md, sim_df, sim_plot = run_tab3_cost_simulator(2.0, 20.0)
    assert "Khuyến nghị Operating Point" in sim_md
    assert not sim_df.empty
    assert sim_plot is not None

    app = build_demo_app()
    assert isinstance(app, gr.Blocks)
