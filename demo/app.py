"""Gradio demo: upload an image, pick category and method, get score, verdict and heatmap (DM-01).

    python demo/app.py

Everything shown comes from trained checkpoints (checkpoints/<method>/<category>/) and saved
evaluation tables (results/metrics/). Nothing is simulated: a method without a checkpoint is
reported as such, and tables that have not been generated yet are shown empty with the command to
produce them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_single import METHODS, compare_all_methods_on_image, inspect_single_image  # noqa: E402
from src.data import load_protocol, resolve_root  # noqa: E402

METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"


def collect_examples(protocol: dict[str, Any]) -> list[list[str]]:
    """One normal and one defective real test image per category, if MVTec AD is downloaded."""
    root = resolve_root(protocol)
    examples: list[list[str]] = []
    for cat in protocol["dataset"]["categories"]:
        test_dir = root / cat / "test"
        if not test_dir.is_dir():
            continue
        good = sorted((test_dir / "good").glob("*.png"))
        if good:
            examples.append([str(good[0]), cat, "patchcore"])
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
            imgs = sorted(defect_dir.glob("*.png"))
            if imgs:
                examples.append([str(imgs[0]), cat, "patchcore"])
                break
    return examples


def _notice(text: str, kind: str = "warn") -> str:
    colors = {"warn": ("#332b00", "#ffd54f"), "error": ("#3b1212", "#ff8a80")}
    bg, fg = colors[kind]
    return f"<div style='padding:12px;border-radius:8px;background:{bg};color:{fg};'>{text}</div>"


def format_verdict_banner(res: dict[str, Any]) -> str:
    """HTML card with the verdict, score and the threshold actually used."""
    is_defect = res["is_defect"]
    bg, border = ("#2d1215", "#e53935") if is_defect else ("#0f291e", "#2e7d32")
    badge = "REJECT: defect detected" if is_defect else "PASS: normal part"
    source = "manual override" if res["threshold_is_override"] else f"validation: {res['val_threshold']:.4f}"
    loc = "pixel heatmap" if res["supports_localization"] else "image-level score only (no pixel map)"
    return f"""
    <div style="padding:16px;border-radius:10px;background:{bg};border:2px solid {border};color:#f5f5f5;">
      <div style="font-size:20px;font-weight:700;margin-bottom:8px;">{badge}</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:14px;">
        <div><b>Category:</b> <code>{res['category']}</code></div>
        <div><b>Method:</b> <code>{res['method']}</code></div>
        <div><b>Latency:</b> <code>{res['latency_ms']} ms</code></div>
        <div><b>Anomaly score:</b> <code>{res['score']:.4f}</code></div>
        <div><b>Threshold:</b> <code>{res['threshold']:.4f}</code> ({source})</div>
        <div><b>Margin:</b> <code>{res['margin']:+.4f}</code></div>
      </div>
      <div style="margin-top:8px;font-size:12px;opacity:0.85;"><b>Localization:</b> {loc}</div>
    </div>"""


def run_tab1_inspection(image, category: str, method_name: str, use_custom_threshold: bool,
                        custom_threshold: float, alpha: float, hotspot_quantile: float):
    """Tab 1: one method on one image."""
    if image is None:
        return _notice("Hãy upload một ảnh hoặc chọn một ảnh mẫu để kiểm định."), None, None, None
    try:
        res = inspect_single_image(
            image, category, method_name,
            custom_threshold=float(custom_threshold) if use_custom_threshold else None,
            alpha=float(alpha), hotspot_quantile=float(hotspot_quantile))
    except Exception as exc:  # shown to the user, e.g. missing checkpoint with the command to train it
        return _notice(f"<b>Lỗi kiểm định:</b> {exc}", "error"), None, None, None
    return format_verdict_banner(res), res["contour_rgb"], res["heatmap_rgb"], res["binary_mask_rgb"]


def run_tab2_arena(image, category: str, alpha: float):
    """Tab 2: every method on the same image."""
    if image is None:
        raise gr.Error("Hãy chọn hoặc upload một ảnh để so sánh các phương pháp.")
    df, outputs = compare_all_methods_on_image(image, category, alpha=float(alpha))
    any_res = next(iter(outputs.values()), None)
    return (df, any_res["preprocessed_rgb"] if any_res else None,
            outputs["autoencoder"]["heatmap_rgb"] if "autoencoder" in outputs else None,
            outputs["patchcore"]["heatmap_rgb"] if "patchcore" in outputs else None)


def _missing(cmd: str) -> pd.DataFrame:
    return pd.DataFrame([{"status": f"chưa có kết quả, hãy chạy: {cmd}"}])


def load_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall comparison and per-category pixel metrics from results/metrics/comparison.csv."""
    path = METRICS_DIR / "comparison.csv"
    if not path.exists():
        return _missing("python scripts/evaluate.py"), _missing("python scripts/evaluate.py")
    df = pd.read_csv(path)
    keep = ["method", "category", "image_auroc", "precision", "recall", "f1", "false_alarm_rate",
            "pixel_auroc", "pro"]
    overall = df[df["category"] == "overall"][[c for c in keep if c != "category"]].round(4)
    pixel = df[df["pixel_auroc"].notna() & (df["category"] != "overall")][
        ["method", "category", "pixel_auroc", "pro"]].round(4)
    return overall, pixel


def load_cost_view(method: str) -> tuple[pd.DataFrame, str | None]:
    """False-alarm table (mean over categories) and figure for one method, if generated."""
    path = METRICS_DIR / "false_alarm_cost.csv"
    if not path.exists():
        return _missing("python scripts/run_sensitivity.py --cost-only"), None
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["category"] == "mean")].drop(columns=["category"]).round(4)
    fig = FIGURES_DIR / f"false_alarm_cost_{method}.png"
    return df, str(fig) if fig.exists() else None


def build_demo_app() -> gr.Blocks:
    protocol = load_protocol()
    categories = protocol["dataset"]["categories"]
    examples = collect_examples(protocol)
    overall_df, pixel_df = load_comparison_tables()
    cost_df, cost_fig = load_cost_view(METHODS[0])

    with gr.Blocks(title="MVTec AD visual anomaly inspection") as demo:
        gr.Markdown(
            "# Automated visual quality inspection with few defect samples\n"
            "Ba phương pháp chỉ huấn luyện trên ảnh normal: Autoencoder, PatchCore, One-class SVM."
        )
        with gr.Tabs():
            with gr.TabItem("1. Kiểm định một ảnh"):
                with gr.Row():
                    with gr.Column(scale=4):
                        in_image = gr.Image(type="pil", label="Ảnh sản phẩm")
                        with gr.Row():
                            in_cat = gr.Dropdown(categories, value=categories[0], label="Category")
                            in_method = gr.Dropdown(list(METHODS), value="patchcore", label="Method")
                        with gr.Accordion("Tùy chọn", open=False):
                            in_use_thr = gr.Checkbox(False, label="Ghi đè ngưỡng validation bằng giá trị thủ công")
                            in_thr = gr.Slider(-1.0, 5.0, value=0.5, step=0.01,
                                               label="Ngưỡng thủ công (thang điểm khác nhau giữa các method)")
                            in_alpha = gr.Slider(0.15, 0.85, value=0.45, step=0.05, label="Độ đậm heatmap")
                            in_quant = gr.Slider(0.75, 0.99, value=0.92, step=0.01,
                                                 label="Phân vị khoanh vùng viền đỏ")
                        btn = gr.Button("Kiểm định", variant="primary")
                    with gr.Column(scale=6):
                        out_banner = gr.HTML()
                        with gr.Row():
                            out_contour = gr.Image(label="Khoanh vùng lỗi")
                            out_heatmap = gr.Image(label="Heatmap")
                            out_mask = gr.Image(label="Mặt nạ vùng nghi ngờ")
                if examples:
                    gr.Examples(examples=examples, inputs=[in_image, in_cat, in_method],
                                label="Ảnh mẫu từ tập test MVTec AD (chọn một dòng rồi bấm Kiểm định)")
                else:
                    gr.Markdown("Chưa có ảnh mẫu: tải dữ liệu bằng `python scripts/download_mvtec.py`.")
                btn.click(run_tab1_inspection,
                          [in_image, in_cat, in_method, in_use_thr, in_thr, in_alpha, in_quant],
                          [out_banner, out_contour, out_heatmap, out_mask])

            with gr.TabItem("2. So sánh các phương pháp"):
                with gr.Row():
                    with gr.Column(scale=4):
                        arena_image = gr.Image(type="pil", label="Ảnh cần so sánh")
                        arena_cat = gr.Dropdown(categories, value=categories[0], label="Category")
                        arena_alpha = gr.Slider(0.2, 0.8, value=0.45, step=0.05, label="Độ đậm heatmap")
                        btn_arena = gr.Button("So sánh", variant="primary")
                    with gr.Column(scale=6):
                        arena_table = gr.Dataframe(label="Kết quả từng phương pháp")
                        with gr.Row():
                            arena_in = gr.Image(label="Ảnh sau tiền xử lý")
                            arena_ae = gr.Image(label="Autoencoder")
                            arena_pc = gr.Image(label="PatchCore")
                btn_arena.click(run_tab2_arena, [arena_image, arena_cat, arena_alpha],
                                [arena_table, arena_in, arena_ae, arena_pc])

            with gr.TabItem("3. Kết quả đã đánh giá"):
                gr.Markdown("Bảng đọc từ `results/metrics/` (sinh bởi `scripts/evaluate.py`), "
                            "trung bình theo category; ngưỡng lấy từ tập validation.")
                gr.Dataframe(value=overall_df, label="So sánh tổng thể")
                gr.Dataframe(value=pixel_df, label="Định vị mức pixel (pixel AUROC, PRO)")
                gr.Markdown("### Điểm vận hành và báo động giả\n"
                            "Mỗi dòng là một phân vị của điểm normal trên validation, áp lên tập test. "
                            "Bảng chỉ mô tả sự đánh đổi; không dùng nó để chọn ngưỡng cuối bằng cách "
                            "tối ưu trên tập test.")
                cost_method = gr.Dropdown(list(METHODS), value=METHODS[0], label="Method")
                cost_table = gr.Dataframe(value=cost_df)
                cost_image = gr.Image(value=cost_fig, label="Đường cong đánh đổi")
                cost_method.change(load_cost_view, cost_method, [cost_table, cost_image])
    return demo


if __name__ == "__main__":
    build_demo_app().launch(server_name="127.0.0.1", server_port=7860)
