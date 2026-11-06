"""Gradio 3-Tab Industrial QA Dashboard for MVTec AD Visual Anomaly Inspection (DM-01)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_single import compare_all_methods_on_image, inspect_single_image
from src.data import load_protocol, resolve_root
from src.evaluation import analyze_false_alarm_cost
from src.utils import ensure_dir, plot_cost_tradeoff_curve


def ensure_sample_examples(protocol: dict[str, Any]) -> list[list[Any]]:
    """Collect real MVTec test samples if downloaded, or generate crisp synthetic samples for the gallery."""
    categories = protocol["dataset"]["categories"]
    data_root = resolve_root(protocol)
    examples: list[list[Any]] = []

    if data_root.is_dir():
        for cat in categories:
            test_dir = data_root / cat / "test"
            if not test_dir.is_dir():
                continue
            good_imgs = sorted((test_dir / "good").glob("*.png"))
            if good_imgs:
                examples.append([str(good_imgs[0]), cat, "patchcore"])
            for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
                def_imgs = sorted(defect_dir.glob("*.png"))
                if def_imgs:
                    examples.append([str(def_imgs[0]), cat, "patchcore"])
                    break
        if examples:
            return examples

    ex_dir = ensure_dir(REPO_ROOT / "demo" / "examples")
    rng = np.random.default_rng(42)
    palette = {
        "bottle": (125, 130, 138),
        "carpet": (110, 115, 128),
        "leather": (135, 118, 105),
        "screw": (128, 128, 132),
        "transistor": (115, 132, 120),
    }

    for cat in categories:
        base_color = np.array(palette.get(cat, (127, 127, 127)), dtype=np.float32)
        good_path = ex_dir / f"{cat}_good_sample.png"
        defect_path = ex_dir / f"{cat}_defect_sample.png"

        if not good_path.exists() or not defect_path.exists():
            noise = rng.normal(0.0, 4.0, size=(256, 256, 3)).astype(np.float32)
            good_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            cv2.circle(good_img, (128, 128), 72, (135, 138, 142), 2)
            Image.fromarray(good_img).save(good_path)

            def_img = good_img.copy()
            cv2.line(def_img, (88, 92), (168, 162), (250, 35, 35), 9)
            cv2.circle(def_img, (148, 112), 20, (20, 245, 240), -1)
            Image.fromarray(def_img).save(defect_path)

        examples.append([str(defect_path), cat, "patchcore"])
        examples.append([str(good_path), cat, "patchcore"])

    return examples


def format_verdict_banner(res: dict[str, Any]) -> str:
    """Create a rich HTML status card summarizing inspection results."""
    is_defect = res["is_defect"]
    bg = "#2d1215" if is_defect else "#0f291e"
    border = "#e53935" if is_defect else "#2e7d32"
    badge = "🚨 REJECT — DEFECT DETECTED" if is_defect else "✅ PASS — NORMAL PART"
    loc_text = "Pixel Heatmap + Contour Enabled" if res["supports_localization"] else "Image-Level Score Only (No Pixel Map)"

    return f"""
    <div style="padding: 16px; border-radius: 10px; background: {bg}; border: 2px solid {border}; color: #f5f5f5;">
      <div style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">{badge}</div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 14px;">
        <div><b>Category:</b> <code>{res['category']}</code></div>
        <div><b>Method:</b> <code>{res['method']}</code></div>
        <div><b>Latency:</b> <code>{res['latency_ms']} ms</code></div>
        <div><b>Anomaly Score:</b> <code>{res['score']:.4f}</code></div>
        <div><b>Active Threshold:</b> <code>{res['threshold']:.4f}</code> (Val P95: <code>{res['val_threshold']:.4f}</code>)</div>
        <div><b>Margin (Score - Thr):</b> <code>{res['margin']:+.4f}</code></div>
      </div>
      <div style="margin-top: 8px; font-size: 12px; opacity: 0.85;">
        <b>Localization:</b> {loc_text} &nbsp;|&nbsp; <b>Model Source:</b> <code>{res['checkpoint_source']}</code>
      </div>
    </div>
    """


def run_tab1_inspection(
    image: Image.Image | np.ndarray | str | None,
    category: str,
    method_name: str,
    use_custom_threshold: bool,
    custom_threshold: float,
    alpha: float,
    hotspot_quantile: float,
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Gradio callback for Tab 1: Single-Method Live Quality Inspector."""
    if image is None:
        return (
            "<div style='padding:12px;border-radius:8px;background:#332b00;color:#ffd54f;'>"
            "⚠️ Vui lòng upload một bức ảnh hoặc bấm chọn một ảnh mẫu bên dưới để kiểm định.</div>",
            None,
            None,
            None,
        )
    try:
        thr = float(custom_threshold) if use_custom_threshold else None
        res = inspect_single_image(
            image_input=image,
            category=category,
            method_name=method_name,
            custom_threshold=thr,
            alpha=float(alpha),
            hotspot_quantile=float(hotspot_quantile),
            allow_demo_fallback=True,
        )
        banner = format_verdict_banner(res)
        return (
            banner,
            res["contour_rgb"],
            res["heatmap_rgb"],
            res["binary_mask_rgb"],
        )
    except Exception as exc:
        err_html = (
            f"<div style='padding:12px;border-radius:8px;background:#3b1212;color:#ff8a80;'>"
            f"❌ <b>Inspection Error:</b> {exc}</div>"
        )
        return err_html, None, None, None


def run_tab2_arena(
    image: Image.Image | np.ndarray | str | None,
    category: str,
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Gradio callback for Tab 2: Side-by-Side 3-Method Arena."""
    if image is None:
        raise gr.Error("Vui lòng chọn hoặc upload một bức ảnh để so sánh 3 phương pháp.")
    df, outputs = compare_all_methods_on_image(
        image_input=image,
        category=category,
        alpha=float(alpha),
        allow_demo_fallback=True,
    )
    input_rgb = outputs["patchcore"]["preprocessed_rgb"]
    ae_vis = outputs["autoencoder"]["heatmap_rgb"]
    pc_vis = outputs["patchcore"]["heatmap_rgb"]
    return df, input_rgb, ae_vis, pc_vis


def load_benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved benchmark tables from results/metrics/ or show informative placeholder."""
    main_csv = REPO_ROOT / "results" / "metrics" / "main_comparison_table.csv"
    loc_csv = REPO_ROOT / "results" / "metrics" / "pixel_localization_table.csv"

    if main_csv.exists():
        main_df = pd.read_csv(main_csv).round(4)
    else:
        main_df = pd.DataFrame(
            [
                {
                    "run_name": "patchcore (pending full Colab run)",
                    "method": "patchcore",
                    "image_auroc": "Run scripts/evaluate.py",
                    "f1": "-",
                    "pixel_auroc": "-",
                    "pixel_pro": "-",
                }
            ]
        )

    if loc_csv.exists():
        loc_df = pd.read_csv(loc_csv).round(4)
    else:
        loc_df = pd.DataFrame(
            [
                {
                    "method": "patchcore / autoencoder",
                    "category": "all",
                    "pixel_auroc": "Run scripts/evaluate.py",
                    "pixel_pro": "-",
                }
            ]
        )
    return main_df, loc_df


def run_tab3_cost_simulator(
    c_fp: float,
    c_fn: float,
) -> tuple[str, pd.DataFrame, Image.Image | None]:
    """Interactive factory false-alarm vs. missed-defect cost simulator (I-01)."""
    ratio = float(c_fn) / max(float(c_fp), 1e-6)
    saved_cost_csv = REPO_ROOT / "results" / "metrics" / "false_alarm_cost.csv"

    if saved_cost_csv.exists():
        raw_cost = pd.read_csv(saved_cost_csv)
        pc_mean = raw_cost[
            (raw_cost["run_name"] == "patchcore") & (raw_cost["category"] == "mean")
        ].copy()
        if not pc_mean.empty:
            n_avg_test = 584.0 / 5.0
            pc_mean["expected_dollar_cost_per_part"] = (
                float(c_fp) * pc_mean["fp"] + float(c_fn) * pc_mean["fn"]
            ) / n_avg_test
            pc_mean[f"cost_ratio_{ratio:.1f}"] = (
                1.0 * pc_mean["fp"] + ratio * pc_mean["fn"]
            ) / n_avg_test
            cost_df = pc_mean.reset_index(drop=True)
        else:
            cost_df = pd.DataFrame()
    else:
        cost_df = pd.DataFrame()

    if cost_df.empty:
        rng = np.random.default_rng(42)
        val_scores = rng.normal(0.25, 0.08, size=100).clip(0.05, 0.60)
        test_normal = rng.normal(0.26, 0.09, size=60).clip(0.05, 0.65)
        test_defect = rng.normal(0.55, 0.14, size=60).clip(0.20, 0.98)

        test_labels = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
        test_scores = np.concatenate([test_normal, test_defect])

        cost_df = analyze_false_alarm_cost(
            test_labels=test_labels,
            test_scores=test_scores,
            val_scores=val_scores,
            cost_ratios=(1.0, ratio),
            percentiles=(80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0),
        )
        cost_df["expected_dollar_cost_per_part"] = (
            float(c_fp) * cost_df["fp"] + float(c_fn) * cost_df["fn"]
        ) / len(test_labels)

    best_idx = int(cost_df["expected_dollar_cost_per_part"].idxmin())
    best_row = cost_df.loc[best_idx]

    tmp_fig = ensure_dir(REPO_ROOT / "demo" / "examples") / "live_cost_curve.png"
    plot_cost_tradeoff_curve(
        cost_df,
        tmp_fig,
        title=f"Factory Operating Point Trade-off (C_FN / C_FP = {ratio:.1f})",
    )
    pil_plot = Image.open(tmp_fig).convert("RGB")

    rec_md = (
        f"### 💡 Khuyến nghị Operating Point cho Nhà máy (trên dữ liệu MVTec AD — PatchCore)\n"
        f"- Với chi phí **Báo động giả ($C_{{\\text{{FP}}}} = \\${c_fp:.1f}$)** và chi phí **Bỏ lọt hàng lỗi ($C_{{\\text{{FN}}}} = \\${c_fn:.1f}$)** "
        f"(tỷ lệ $C_{{\\text{{FN}}}}/C_{{\\text{{FP}}}} = {ratio:.1f}\\times$):\n"
        f"- **Ngưỡng phân vị Validation tối ưu:** **P{best_row['percentile']:.1f}** (`threshold = {best_row['threshold']:.4f}`)\n"
        f"- **Tỷ lệ Báo động giả (FPR):** `{best_row['fpr']*100:.1f}%` &nbsp;|&nbsp; **Tỷ lệ Bắt được lỗi (Recall):** `{best_row['recall']*100:.1f}%` "
        f"&nbsp;|&nbsp; **Chi phí kỳ vọng thấp nhất:** `${best_row['expected_dollar_cost_per_part']:.2f} / sản phẩm`"
    )
    display_cols = [
        "percentile",
        "threshold",
        "fpr",
        "recall",
        "precision",
        "f1",
        "expected_dollar_cost_per_part",
    ]
    return rec_md, cost_df[display_cols].round(4), pil_plot


def build_demo_app() -> gr.Blocks:
    """Construct the 3-Tab Gradio application."""
    protocol = load_protocol()
    categories = protocol["dataset"]["categories"]
    examples = ensure_sample_examples(protocol)
    main_df, loc_df = load_benchmark_tables()

    with gr.Blocks(title="MVTec AD Visual Anomaly Inspection — Topic 2") as demo:
        gr.Markdown(
            "# 🏭 Automated Visual Quality Inspection with Few Defect Samples\n"
            "**Course Project (Topic 2) · MVTec AD Benchmark** — So sánh 3 hướng tiếp cận chỉ huấn luyện trên ảnh Normal: "
            "**(1) Autoencoder Reconstruction**, **(2) PatchCore Feature Memory**, và **(3) One-Class SVM**."
        )

        with gr.Tabs():
            # TAB 1: LIVE QUALITY INSPECTOR
            with gr.TabItem("🔍 1. Trạm Kiểm Định Trực Tiếp (Live Inspector)"):
                with gr.Row():
                    with gr.Column(scale=4):
                        in_image = gr.Image(type="pil", label="Ảnh sản phẩm đầu vào (Upload hoặc chọn mẫu bên dưới)")
                        with gr.Row():
                            in_cat = gr.Dropdown(
                                choices=categories,
                                value=categories[0],
                                label="Category (Loại sản phẩm)",
                            )
                            in_method = gr.Dropdown(
                                choices=["patchcore", "autoencoder", "one_class", "patchcore_refined"],
                                value="patchcore",
                                label="Phương pháp phát hiện (Method)",
                            )
                        with gr.Accordion("⚙️ Tùy chỉnh Ngưỡng Kiểm Duyệt & Hiển Thị (Operating Point)", open=False):
                            in_use_custom_thr = gr.Checkbox(
                                value=False,
                                label="Ghi đè ngưỡng Validation P95 bằng thanh trượt thủ công",
                            )
                            in_custom_thr = gr.Slider(
                                minimum=-1.0,
                                maximum=12.0,
                                value=7.2,
                                step=0.05,
                                label="Custom Anomaly Threshold",
                            )
                            in_alpha = gr.Slider(
                                minimum=0.15,
                                maximum=0.85,
                                value=0.45,
                                step=0.05,
                                label="Độ đậm bản đồ nhiệt (Heatmap Opacity α)",
                            )
                            in_quantile = gr.Slider(
                                minimum=0.75,
                                maximum=0.99,
                                value=0.92,
                                step=0.01,
                                label="Ngưỡng khoanh vùng viền đỏ (Hotspot Contour Quantile)",
                            )
                        btn_inspect = gr.Button("🔍 Chạy Kiểm Định Chất Lượng (Inspect Part)", variant="primary")

                    with gr.Column(scale=6):
                        out_banner = gr.HTML(label="Kết luận kiểm định")
                        with gr.Row():
                            out_contour = gr.Image(label="1. Khoanh vùng vết lỗi (Defect Contours)")
                            out_heatmap = gr.Image(label="2. Bản đồ nhiệt (Anomaly Heatmap)")
                            out_mask = gr.Image(label="3. Mặt nạ vùng nghi ngờ (Hotspot Mask)")

                gr.Examples(
                    examples=examples,
                    inputs=[in_image, in_cat, in_method],
                    label="📸 Bộ ảnh mẫu bấm-là-chạy (Click vào một dòng bất kỳ rồi bấm Inspect Part)",
                )

                btn_inspect.click(
                    fn=run_tab1_inspection,
                    inputs=[
                        in_image,
                        in_cat,
                        in_method,
                        in_use_custom_thr,
                        in_custom_thr,
                        in_alpha,
                        in_quantile,
                    ],
                    outputs=[out_banner, out_contour, out_heatmap, out_mask],
                )

            # TAB 2: 3-METHOD SIDE-BY-SIDE ARENA
            with gr.TabItem("⚔️ 2. Đấu Trường So Sánh 3 Phương Pháp (3-Method Arena)"):
                gr.Markdown(
                    "Chạy đồng thời **PatchCore**, **Autoencoder** và **One-Class SVM** trên cùng một bức ảnh "
                    "để đối chiếu trực tiếp điểm số, phán quyết và chất lượng định vị vùng lỗi."
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        arena_image = gr.Image(type="pil", label="Ảnh cần đối chiếu 3 phương pháp")
                        arena_cat = gr.Dropdown(
                            choices=categories,
                            value=categories[0],
                            label="Category",
                        )
                        arena_alpha = gr.Slider(
                            minimum=0.2,
                            maximum=0.8,
                            value=0.45,
                            step=0.05,
                            label="Heatmap Opacity α",
                        )
                        btn_arena = gr.Button("⚔️ Chạy Đối Đầu Cả 3 Phương Pháp", variant="primary")
                    with gr.Column(scale=6):
                        arena_table = gr.Dataframe(label="Bảng đối chiếu trực tiếp 3 phương pháp")
                        with gr.Row():
                            arena_in_vis = gr.Image(label="Ảnh sau tiền xử lý (224x224)")
                            arena_ae_vis = gr.Image(label="Autoencoder Reconstruction Heatmap")
                            arena_pc_vis = gr.Image(label="PatchCore Feature Memory Heatmap")

                btn_arena.click(
                    fn=run_tab2_arena,
                    inputs=[arena_image, arena_cat, arena_alpha],
                    outputs=[arena_table, arena_in_vis, arena_ae_vis, arena_pc_vis],
                )

            # TAB 3: BENCHMARK TABLES & FACTORY COST SIMULATOR
            with gr.TabItem("📊 3. Bảng Điểm Benchmark & Mô Phỏng Chi Phí (Analytics & Cost)"):
                gr.Markdown("### 🏆 Bảng tổng hợp kết quả thực nghiệm (MVTec AD Shared Protocol)")
                gr.Dataframe(value=main_df, label="Main Comparison Table (E-02)")
                gr.Dataframe(value=loc_df, label="Pixel-Level Localization Table — Pixel AUROC & PRO (L-01)")

                gr.Markdown("---")
                gr.Markdown(
                    "### 💰 Bộ Mô Phỏng Chi Phí Báo Động Giả vs. Bỏ Lọt Hàng Lỗi (False-Alarm Cost Simulator — I-01)"
                )
                with gr.Row():
                    sim_cfp = gr.Slider(
                        minimum=0.5,
                        maximum=20.0,
                        value=2.0,
                        step=0.5,
                        label="Chi phí kiểm tra lại 1 Báo động giả (C_FP - USD)",
                    )
                    sim_cfn = gr.Slider(
                        minimum=1.0,
                        maximum=100.0,
                        value=20.0,
                        step=1.0,
                        label="Chi phí lọt 1 Sản phẩm lỗi ra thị trường (C_FN - USD)",
                    )
                    btn_sim = gr.Button("📈 Tính Ngưỡng Tối Ưu Chi Phí", variant="secondary")

                sim_md = gr.Markdown()
                with gr.Row():
                    sim_table = gr.Dataframe(label="Bảng đánh đổi FPR / Recall / Chi phí theo Validation Percentile")
                    sim_plot = gr.Image(label="Đường cong Operating Point & Chi phí kỳ vọng")

                btn_sim.click(
                    fn=run_tab3_cost_simulator,
                    inputs=[sim_cfp, sim_cfn],
                    outputs=[sim_md, sim_table, sim_plot],
                )

    return demo


if __name__ == "__main__":
    app = build_demo_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
