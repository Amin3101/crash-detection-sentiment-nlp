"""CrashLens: evidence-first dashboard for Nexar collision experiments.

Run with `streamlit run app.py`. No predictions are fabricated or inferred
without the original checkpoint and preprocessing pipeline.
"""
from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from analytics import PredictionValidationError, evaluate, validate_predictions
from log_explorer import read_log_map, track_steps, report_section, markdown_sections

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
TEAL = "#5CD8D0"
BLUE = "#78A6F8"
ORANGE = "#F3B568"
RED = "#F27B7B"
MUTED = "#98AABD"
PAPER = "#E8EFF7"
CARD = "#142334"

st.set_page_config(
    page_title="CrashLens | Video Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans','Segoe UI',sans-serif;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 80% 0%,#132b3b 0%,#0a131f 38%,#08111b 100%);}
[data-testid="stSidebar"] {background:#0c1a28;border-right:1px solid #203347;}
.block-container {padding-top:2rem; max-width:1450px; padding-bottom:4rem;}
[data-testid="stMetric"] {background:#132336;border:1px solid #263c4d;border-radius:15px;padding:12px 15px;}
[data-testid="stMetricValue"] {font-weight:800;color:#E8EFF7;font-size:1.6rem;}
[data-testid="stMetricLabel"] {color:#98aabd;}
div[data-testid="stHorizontalBlock"] {align-items:stretch;}
.hero {background:linear-gradient(110deg, rgba(46,108,128,.31), rgba(20,35,52,.9)); border:1px solid #305065;
  border-radius:20px; padding:27px 29px; margin-bottom:18px;}
.hero-top {font-size:11px;letter-spacing:.17em;text-transform:uppercase;font-weight:700;color:#74e6da;}
.hero h1 {font-size:2.25rem;font-weight:800;letter-spacing:-.055em;margin:10px 0 3px; color:#F3F8FE;}
.hero p {color:#afc1d0;font-size:.96rem;max-width:850px;margin:0;line-height:1.6;}
.kpi-row {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:17px 0 21px;}
.kpi {background:linear-gradient(160deg, #152a3c,#11202e);border:1px solid #2a4253;border-radius:16px;padding:19px 20px;}
.kpi-label {font-size:11px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#9cabbc;}
.kpi-number {font-size:2rem;font-weight:800;letter-spacing:-.06em;color:#edf8fc;margin:7px 0 4px;}
.kpi-sub {font-size:12px;color:#94aaba;}
.section-title {font-size:1.17rem;font-weight:750;letter-spacing:-.02em;margin:12px 0;color:#e9f2f9;}
.insight {border-left:3px solid #5cd8d0;padding:12px 17px;background:#12303b;border-radius:0 12px 12px 0;margin:15px 0;color:#bad8de;line-height:1.65;font-size:14px;}
.pill {display:inline-block;border-radius:100px;padding:4px 10px;font-size:11px;font-weight:700;margin-right:7px;letter-spacing:.04em;}
.pill.ok {background:#19423d;color:#7aebe0;}.pill.warning {background:#493424;color:#f6be84;}
.pill.neutral {background:#213953;color:#a6cbf0;}
.note {font-size:12px;color:#9cafbe;line-height:1.6;}
[data-testid="stDataFrame"] {border:1px solid #253a4b;border-radius:12px;}
@media(max-width:1000px) {.kpi-row {grid-template-columns: repeat(2,minmax(0,1fr));}}
@media(max-width:590px) {.kpi-row {grid-template-columns: 1fr;} .hero h1{font-size:1.65rem;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def format_metric(x: float | None, decimals: int = 3) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x):.{decimals}f}"


def hero(eyebrow: str, heading: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-top">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(heading)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-number">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(subtitle)}</div></div>'
        for label, value, subtitle in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{html.escape(text)}</div>', unsafe_allow_html=True)


def chart_style(fig: go.Figure, height: int = 340, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=14, r=16, t=20, b=22),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PAPER, family="DM Sans, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="#26384B", zeroline=False),
        yaxis=dict(gridcolor="#26384B", zeroline=False, title=ytitle),
        hoverlabel=dict(bgcolor="#20364B", font_color=PAPER),
    )
    return fig


def plot_training(run: str, metric: str) -> go.Figure:
    frame = read_csv("training.csv").query("run == @run")
    labels = {
        "loss": ["train_loss", "val_loss"],
        "f1": ["train_f1", "val_f1"],
        "auc": ["val_auc"],
    }[metric]
    fig = go.Figure()
    for key in labels:
        fig.add_trace(go.Scatter(
            x=frame.epoch, y=frame[key], name=key.replace("_", " ").title(),
            mode="lines+markers", line=dict(color=TEAL if "val" in key else BLUE, width=3),
            marker=dict(size=7), hovertemplate="Epoch %{x}<br>Value %{y:.4f}<extra>%{fullData.name}</extra>"
        ))
    fig.update_xaxes(title="Epoch", dtick=1)
    fig.update_yaxes(tickformat=".2f")
    return chart_style(fig, height=330)


def cm_chart(cm: np.ndarray, title: str = "Confusion matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: no", "Pred: collision"], y=["True: no", "True: collision"],
        colorscale=[[0, "#163345"], [0.45, "#276578"], [1, "#50BBAF"]],
        showscale=False, text=cm, texttemplate="%{text}",
        textfont=dict(size=24, color="#f7ffff"), hovertemplate="%{y} / %{x}<br>%{z} samples<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=title, font=dict(size=15)), xaxis_side="bottom")
    return chart_style(fig, 310)


def loaded_predictions() -> pd.DataFrame | None:
    return st.session_state.get("uploaded_predictions")


def selected_predictions() -> tuple[pd.DataFrame | None, str | None]:
    raw = loaded_predictions()
    if raw is None:
        return None, None
    if "split" in raw.columns:
        possible = [v for v in ["validation", "val", "test", "train"] if v in raw["split"].unique()]
        split = st.selectbox("Dataset split", possible, key="analysis_split")
        return raw[raw.split == split].copy(), split
    st.caption("CSV does not contain a split column; evaluation partition is unknown.")
    return raw.copy(), None


def threshold_widget(split: str | None, key: str) -> float:
    if split == "test":
        sandbox = st.checkbox("Allow exploratory threshold changes on test labels (not a valid final evaluation)", value=False, key=f"sandbox_{key}")
        if not sandbox:
            st.caption("Threshold locked to 0.50 to avoid test-set threshold tuning.")
            return 0.5
        st.warning("Exploratory only. Choose a threshold on validation data and then evaluate test once.")
    elif split is None:
        st.warning("Split is unknown; threshold changes are exploratory only.")
    return st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05, key=f"threshold_{key}")


def metric_strip(m: dict) -> None:
    vals = [
        ("Samples", f"{m['samples']:,}", "Imported sample scores"),
        ("Macro F1", format_metric(m["macro_f1"]), "Computed from probabilities"),
        ("ROC-AUC", format_metric(m["roc_auc"]), "Threshold independent"),
        ("Accuracy", f"{m['accuracy']:.1%}", "At selected threshold"),
    ]
    stat_cards(vals)


def source_note(names: list[str]) -> None:
    with st.expander("Sources and evaluation boundaries"):
        for source in names:
            st.code(source, language=None)
        st.caption("Metric entries are transcribed from these local notebooks/reports. Run-level predictions, source videos and model weights are not included in the original ZIP.")


def overview() -> None:
    hero("CrashLens / Project intelligence", "Collision intelligence, with evidence.",
         "A two-model video inference console and a traceable research archive. Detection and pre-event anticipation are evaluated separately.")
    stat_cards([
        ("Full-video ROC-AUC", "0.944", "Detection · 225 test videos"),
        ("Full-video Macro-F1", "0.889", "Detection · final.ipynb"),
        ("Pre-event ROC-AUC", "0.785", "Anticipation · clean run 1"),
        ("Pre-event Macro-F1", "0.729", "Anticipation · 170 test clips"),
    ])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Experimental performance</div>', unsafe_allow_html=True)
        task = st.radio("Protocol", ["Anticipation", "Detection"], horizontal=True)
        runs = read_csv("runs.csv")
        sub = runs[runs.task == task].copy()
        sub["short"] = sub.name.str.replace("Clean pre-event MViT · ", "", regex=False).str.replace("Full-video JPG frames", "Full video", regex=False).str.replace("Event-centered MViT + MLP", "Event centered", regex=False)
        fig = go.Figure()
        for name, col in [("Macro F1", TEAL), ("ROC-AUC", BLUE)]:
            metric = "macro_f1" if name == "Macro F1" else "roc_auc"
            fig.add_trace(go.Bar(x=sub["short"], y=sub[metric], name=name, marker_color=col,
                text=[f"{v:.3f}" for v in sub[metric]], textposition="outside"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
        st.plotly_chart(chart_style(fig, 335), use_container_width=True, config={"displayModeBar":False})
        st.caption("Only runs within the selected task are plotted. Different datasets and model configurations still limit controlled comparisons.")
    with right:
        st.markdown('<div class="section-title">Clean run · training trajectory</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_training("Clean pre-event MViT · run 1", "loss"), use_container_width=True, config={"displayModeBar":False})
        st.caption("Train and validation loss from log.md §7 · best validation-loss epoch: 4.")
    insight("Project story: timing metadata alone reached AUC 0.700 on the original dataset. After deduplication, balancing and group-aware validation, the tested non-visual baseline reached 0.486 (near chance).")
    st.markdown('<div class="section-title">What this dashboard can demonstrate</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("**01 · Live video inference**")
        st.caption("Run either trained checkpoint from Run Two Models; predictions require genuine weights.")
    with c2:
        st.markdown("**02 · Experiment graph & log explorer**")
        st.caption("Follow the separate experiment tracks, parameters, metrics, and source sections.")
    with c3:
        st.markdown("**03 · Failure analysis**")
        st.caption("View confusion matrices and drill into errors once score CSVs are available.")
    source_note(["final.ipynb — cell 10 (225-video full-clip test)", "log.md — §§2,6–8 (clean pre-event test and leakage)"])


def model_lab() -> None:
    hero("Experiments / 02", "Model laboratory", "Compare archived training runs without mixing validation and test results. Every plotted point comes from a report or notebook output.")
    training = read_csv("training.csv")
    run = st.selectbox("Training run", training.run.unique().tolist(), index=1)
    view = st.segmented_control("Metric", options=["loss", "f1", "auc"], default="loss", format_func=lambda v: {"loss":"Loss", "f1":"Macro F1", "auc":"ROC-AUC"}[v])
    st.plotly_chart(plot_training(run, view), use_container_width=True, config={"displayModeBar":False})
    st.caption("Source: log.md §§1,7. Curves for full-video JPG experiment are not reproduced here because their complete epoch-by-epoch series has not been curated.")
    st.markdown('<div class="section-title">Independent test evaluations</div>', unsafe_allow_html=True)
    runs = read_csv("runs.csv")
    task_choice = st.radio("Task type", ["Anticipation", "Detection"], horizontal=True, key="model_task")
    sub = runs[runs.task == task_choice][["name", "samples", "macro_f1", "roc_auc", "accuracy", "source", "notes"]]
    st.dataframe(sub, hide_index=True, use_container_width=True)
    st.caption("Do not treat the highest score across the two task types as a fair winner. They differ in timing, available frames and test splits.")
    st.markdown('<div class="section-title">Ablation study · original temporal dataset</div>', unsafe_allow_html=True)
    abl = read_csv("ablation.csv").sort_values("validation_macro_f1")
    fig = go.Figure(go.Bar(
        x=abl.validation_macro_f1, y=abl.condition, orientation="h",
        marker_color=[TEAL if x == "Normal video" else BLUE for x in abl.condition],
        text=[f"{v:.3f}" for v in abl.validation_macro_f1], textposition="outside"
    ))
    fig.update_xaxes(range=[0, 0.82], title="Validation Macro F1")
    st.plotly_chart(chart_style(fig, 330), use_container_width=True, config={"displayModeBar":False})
    insight("Static-frame performance falls sharply (F1 0.488 vs 0.707). Shuffled-time clips remain relatively strong (0.667), suggesting that learned motion cues are not strongly sensitive to temporal order in this ablation.")
    source_note(["log.md — §3 (ablation), §7 (training), trailing clean-runs summary", "nexar_collision_model_experiments.md — §8 (event-centered checkpoint test)"])


def data_audit() -> None:
    hero("Data quality / 03", "Data integrity audit", "Track why metadata shortcuts inflated early evaluations and how group-aware preprocessing changed the experimental evidence.")
    stat_cards([
        ("Original non-visual AUC", "0.700", "Timing + metadata"),
        ("Clean non-visual AUC", "0.486", "Timing + duration"),
        ("Duplicate video IDs removed", "52", "1364 → 1312 rows"),
        ("Final balanced clips", "1,128", "564 per class"),
    ])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Leakage-fix progression</div>', unsafe_allow_html=True)
        data = read_csv("leakage.csv")
        colors = [RED, ORANGE, ORANGE, TEAL]
        fig = go.Figure(go.Bar(x=data.stage, y=data.auc, marker_color=colors,
            text=data.auc.map(lambda x: f"{x:.3f}"), textposition="outside"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#9DB2C1", annotation_text="Chance 0.50", annotation_position="bottom right")
        fig.update_yaxes(range=[0, 0.9], title="Non-visual ROC-AUC")
        st.plotly_chart(chart_style(fig, 335), use_container_width=True, config={"displayModeBar":False})
    with right:
        st.markdown('<div class="section-title">Baseline details</div>', unsafe_allow_html=True)
        baselines = read_csv("nonvisual.csv")
        st.dataframe(baselines.rename(columns={"feature":"Feature", "original_auc":"Original AUC", "clean_auc":"Clean AUC", "clean_std":"Clean CV ±"}), hide_index=True, use_container_width=True)
        st.caption("Cells without data are intentionally left blank; these results were not reported for every setting.")
    st.markdown('<div class="section-title">How the dataset was repaired</div>', unsafe_allow_html=True)
    phases = [
        ("01", "Find the shortcut", "Start-time metadata alone achieved ROC-AUC 0.653. Compare label distributions using KS statistics and non-visual classifiers."),
        ("02", "Reject failed matching", "Greedy matching to time_after_clip left start_time uncontrolled and distorted negative examples."),
        ("03", "Pool and match candidates", "Build 30 candidate placements per negative video, then sample from the joint timing distribution."),
        ("04", "Deduplicate and group", "Remove 52 repeated video IDs, balance to 564 per class and use StratifiedGroupKFold(groups=video_id)."),
    ]
    for number, title, detail in phases:
        with st.container(border=True):
            st.markdown(f"**{number}  {title}**")
            st.caption(detail)
    insight("Clean AUC ≈ 0.5 supports that the specific timing and duration shortcuts tested are no longer predictive under the recorded CV protocol. It does not prove the absence of every possible source of leakage.")
    source_note(["log.md — §§2,4–6,10", "test.ipynb — cell 5 (group-aware CV, counts, KS tests)"])


def error_lab() -> None:
    hero("Failure analysis / 04", "Where does the model fail?", "Start with a verified aggregate confusion matrix; unlock sample-level error analysis by importing real predictions.")
    with st.expander("Optional · Import per-video test scores for detailed error inspection"):
        uploaded_csv = st.file_uploader("Predictions CSV (video_id, label, probability)", type=["csv"], key="error_prediction_csv")
        if uploaded_csv is not None:
            try:
                raw = pd.read_csv(BytesIO(uploaded_csv.getvalue()))
                st.session_state["uploaded_predictions"] = validate_predictions(raw)
                st.success(f"{len(raw):,} recorded sample predictions loaded")
            except (PredictionValidationError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.session_state.pop("uploaded_predictions", None)
                st.error(f"Invalid CSV: {exc}")
        else:
            st.session_state.pop("uploaded_predictions", None)
        st.caption("These are archived predictions, not live inference. The supplied ZIP did not include per-sample scores.")
        st.download_button("Download CSV schema", (DATA / "predictions_template.csv").read_bytes(), "predictions_template.csv", "text/csv")
    df, split = selected_predictions()
    if df is None:
        st.markdown('<span class="pill neutral">Published aggregate evaluation</span>', unsafe_allow_html=True)
        st.caption("Full-video JPG experiment: 225 test videos · final.ipynb cell 10 · threshold implied by argmax (equivalent to 0.50 for binary softmax).")
        cm = np.array([[97, 16], [9, 103]])
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.plotly_chart(cm_chart(cm), use_container_width=True, config={"displayModeBar":False})
        with right:
            stat_cards([
                ("True negatives", "97", "Correct non-collision"),
                ("False positives", "16", "False collision alarms"),
                ("False negatives", "9", "Missed collisions"),
                ("True positives", "103", "Correct collision detection"),
            ])
            st.info("Individual false-positive and false-negative videos cannot be identified from an aggregate confusion matrix. Import a prediction CSV to inspect specific examples.")
        return
    if df.empty:
        st.warning("No rows in the selected split.")
        return
    threshold = threshold_widget(split, "errors")
    m = evaluate(df, threshold)
    metric_strip(m)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.plotly_chart(cm_chart(m["cm"], title="Imported per-video scores"), use_container_width=True, config={"displayModeBar":False})
    with right:
        st.markdown('<div class="section-title">Error explorer</div>', unsafe_allow_html=True)
        errors = df.copy()
        errors["prediction"] = m["predictions"]
        errors["outcome"] = np.select([
            (errors.label == 0) & (errors.prediction == 0),
            (errors.label == 0) & (errors.prediction == 1),
            (errors.label == 1) & (errors.prediction == 0),
        ], ["TN", "FP", "FN"], default="TP")
        chosen = st.multiselect("Outcomes", ["FP", "FN", "TP", "TN"], default=["FP", "FN"])
        focus = errors[errors.outcome.isin(chosen)]
        st.caption(f"{len(focus)} selected samples")
        st.dataframe(focus[["video_id", "label", "probability", "prediction", "outcome"]], hide_index=True, use_container_width=True)
        st.download_button("Export filtered samples", focus.to_csv(index=False).encode("utf-8"), file_name="crashlens_error_cases.csv", mime="text/csv")
        if not focus.empty:
            video_choice = st.selectbox("Inspect a selected case", focus.video_id.astype(str).tolist())
            record = focus.loc[focus.video_id == video_choice].iloc[0]
            path = safe_video_path(record.get("video_file"))
            if path:
                st.video(str(path))
            else:
                st.caption("No matching local video file in assets/videos/.")
    if len(df.label.unique()) == 2:
        st.markdown('<div class="section-title">Ranking curves</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        fpr, tpr, _ = roc_curve(df.label, df.probability)
        pr, recall, _ = precision_recall_curve(df.label, df.probability)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC", line=dict(color=TEAL, width=3)))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(color=MUTED, dash="dash")))
        roc_fig.update_xaxes(title="False positive rate", range=[0, 1]);roc_fig.update_yaxes(title="True positive rate", range=[0,1])
        pr_fig = go.Figure(go.Scatter(x=recall, y=pr, name="Precision–recall", line=dict(color=BLUE, width=3)))
        pr_fig.update_xaxes(title="Recall", range=[0,1]);pr_fig.update_yaxes(title="Precision", range=[0,1])
        with c1: st.plotly_chart(chart_style(roc_fig, 300), use_container_width=True, config={"displayModeBar":False})
        with c2: st.plotly_chart(chart_style(pr_fig, 300), use_container_width=True, config={"displayModeBar":False})
        st.caption("ROC-AUC and average precision depend on the full probability ranking, not on the decision threshold.")
    source_note(["Imported prediction CSV (user-supplied labels and softmax class-1 probabilities)"])



# ——— V2: dual-model inference and complete experiment history ———
def _checkpoint_path(task: str) -> Path:
    return BASE / "models" / ("detection.pth" if task == "Detection" else "anticipation.pth")


def _checkpoint_badge(task: str) -> tuple[str, bool]:
    checkpoint = _checkpoint_path(task)
    if checkpoint.is_file():
        return f"Checkpoint found · {checkpoint.stat().st_size / 1024**2:,.1f} MB", True
    return "Checkpoint missing · no inference available yet", False


@st.cache_resource(show_spinner="Loading verified checkpoint …", max_entries=2)
def _cached_model(checkpoint: str, mtime: int, device: str):
    # Inference dependencies are optional so the research dashboard keeps working.
    from model_runtime import load_checkpoint
    return load_checkpoint(checkpoint, device)


def _choose_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def inference_console() -> None:
    import hashlib
    import tempfile
    from contextlib import ExitStack

    hero("CrashLens / Two production paths", "Run the two video models.",
         "Upload one dashcam video. Detection scores the entire video; Anticipation evaluates pre-event, four-second windows. The two outputs answer different questions.")
    d, a = st.columns(2, gap="medium")
    with d:
        with st.container(border=True):
            st.markdown("### 01 · Collision Detection")
            st.caption("Full-video classification · 16 frames over the complete clip · final.ipynb")
            st.metric("Recorded test ROC-AUC", "0.944")
            status, available = _checkpoint_badge("Detection")
            st.success(status) if available else st.warning(status)
    with a:
        with st.container(border=True):
            st.markdown("### 02 · Collision Anticipation")
            st.caption("Pre-event temporal windows · 16 frames / 4 seconds · main.ipynb")
            st.metric("Reported B validation ROC-AUC", "0.749")
            status, available = _checkpoint_badge("Anticipation")
            st.success(status) if available else st.warning(status)
            st.caption("Experiment B only documented to epoch 8; this is not an independent test score.")

    st.info("Checkpoint files were not included in the original project ZIP. Place the two *distinct*, architecture-matched files in models/detection.pth and models/anticipation.pth. The interface will not invent predictions if they are missing.")
    task_mode = st.radio("Which model(s) should process the video?", ["Detection", "Anticipation", "Both"], horizontal=True)
    selected = ["Detection", "Anticipation"] if task_mode == "Both" else [task_mode]
    uploaded = st.file_uploader("Upload a dashcam video (MP4, MOV, WebM)", type=["mp4", "mov", "webm", "m4v"], key="inference_upload")
    settings = st.columns([1, 1, 1], gap="medium")
    with settings[0]:
        threshold_d = st.slider("Detection threshold", 0.05, 0.95, 0.50, 0.05)
    with settings[1]:
        threshold_a = st.slider("Anticipation threshold", 0.05, 0.95, 0.50, 0.05)
    with settings[2]:
        device_choice = st.selectbox("Compute device", ["Auto", "CPU"])
    use_event = False
    event_at = 0.0
    stride = 1.0
    max_windows = 8
    if "Anticipation" in selected:
        st.markdown("**Anticipation sampling**")
        event_mode = st.radio("Window strategy", ["Sliding windows (unknown event time)", "Known event timestamp (4s window ending 1s before event)"], horizontal=True)
        use_event = event_mode.startswith("Known")
        if use_event:
            event_at = st.number_input("Known collision event time (seconds)", min_value=5.0, value=8.0, step=0.5)
            st.caption("Uses [event_time − 5s, event_time − 1s); the event time is supplied by the user, not predicted by the model.")
        else:
            stride = st.slider("Window stride (seconds)", 0.5, 4.0, 1.0, 0.5)
            max_windows = st.slider("Max windows to score (from beginning)", 1, 30, 8)
            st.caption("Scores the footage seen within each 4-second window. This is exploratory risk scoring, not validated time-to-collision or guaranteed advance warning.")

    if uploaded is not None:
        st.video(uploaded)
        st.caption(f"File: {uploaded.name} · {uploaded.size/1024**2:.1f} MB")
    deps_ready = True
    if uploaded is not None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as ex:
            st.error(f"Inference libraries missing: {ex}. Install with: python -m pip install torch torchvision opencv-python-headless Pillow")
            deps_ready = False
    ready = all(_checkpoint_path(name).is_file() for name in selected)
    key = None
    if uploaded is not None:
        # Hash the upload so a prior video's score is never displayed for a new file.
        key = hashlib.sha256(uploaded.getvalue()).hexdigest() + ":" + task_mode + ":" + str((threshold_d, threshold_a, use_event, event_at, stride, max_windows, device_choice))
        for name in selected:
            p = _checkpoint_path(name)
            key += f":{p.stat().st_mtime_ns if p.exists() else 'missing'}"
    if st.button("▶ Run real model inference", type="primary", disabled=(uploaded is None or not ready or not deps_ready), use_container_width=True):
        from model_runtime import get_video_info, sample_video, scan_windows, predict_probability
        device = _choose_device() if device_choice == "Auto" else "cpu"
        try:
            # Temporary upload is deleted immediately after inference.
            with tempfile.TemporaryDirectory(prefix="crashlens_") as td:
                suffix=Path(uploaded.name).suffix.lower()
                video_path=Path(td)/("input"+suffix)
                video_path.write_bytes(uploaded.getvalue())
                info=get_video_info(video_path)
                result_rows=[]
                for task in selected:
                    path=_checkpoint_path(task)
                    with st.spinner(f"Running {task} using {device.upper()} …"):
                        model, metadata=_cached_model(str(path),path.stat().st_mtime_ns,device)
                        if task == "Detection":
                            arr=sample_video(video_path,"detection")
                            prob=predict_probability(model,arr,device)
                            result_rows.append(dict(model="Detection",start_s=0.0,end_s=round(info['duration'],2),collision_probability=prob,threshold=threshold_d,decision=int(prob>=threshold_d),sampling="Full video"))
                        elif use_event:
                            start=float(event_at)-5.0
                            if start<0 or event_at-1>info["duration"]:
                                raise ValueError(f"Event time must be between 5.0s and {info['duration']+1:.1f}s to obtain the complete pre-event 4s window.")
                            arr=sample_video(video_path,"anticipation",start=start)
                            prob=predict_probability(model,arr,device)
                            result_rows.append(dict(model="Anticipation",start_s=round(start,2),end_s=round(event_at-1,2),collision_probability=prob,threshold=threshold_a,decision=int(prob>=threshold_a),sampling="Known event pre-window"))
                        else:
                            windows=scan_windows(video_path,stride=float(stride))
                            if not windows:
                                raise ValueError(f"Video needs at least 4s for the anticipation path (video={info['duration']:.2f}s).")
                            run_windows=windows[:int(max_windows)]
                            progress=st.progress(0.0,text=f"Anticipation: 0/{len(run_windows)}")
                            for ix,(start,end) in enumerate(run_windows):
                                arr=sample_video(video_path,"anticipation",start=start)
                                prob=predict_probability(model,arr,device)
                                result_rows.append(dict(model="Anticipation",start_s=start,end_s=end,collision_probability=prob,threshold=threshold_a,decision=int(prob>=threshold_a),sampling="Sliding exploratory"))
                                progress.progress((ix+1)/len(run_windows),text=f"Anticipation: {ix+1}/{len(run_windows)}")
                            progress.empty()
                            if len(windows)>len(run_windows):
                                st.info(f"Only the first {len(run_windows)} of {len(windows)} available windows were scored. Increase the window limit for broader coverage.")
                st.session_state["inference_outcome"]={"key":key,"rows":result_rows,"duration":info['duration'],"device":device}
        except (ValueError,RuntimeError,ImportError,FileNotFoundError,OSError,MemoryError) as err:
            st.session_state.pop("inference_outcome",None)
            st.error(f"Inference stopped: {err}")
            st.caption("Common causes: checkpoint not compatible with notebook architecture, missing CUDA memory, unsupported video codec or wrong window range. A failure does not produce a placeholder score.")
    result = st.session_state.get("inference_outcome")
    if result is not None and result["key"]==key:
        scored=pd.DataFrame(result["rows"])
        st.markdown('<div class="section-title">Real checkpoint predictions</div>',unsafe_allow_html=True)
        st.caption(f"Processed using {result['device'].upper()} · video duration {result['duration']:.2f}s · class 1 softmax probability · threshold is user-selected")
        for model_name in scored.model.unique():
            subset=scored[scored.model==model_name]
            st.markdown(f"**{model_name}**")
            if model_name=="Detection":
                r=subset.iloc[0]
                st.metric("Probability of collision (observed full video)",f"{r.collision_probability:.1%}")
                st.info("Predicted label: "+("Collision" if r.decision else "No collision"))
            else:
                fig=go.Figure(go.Scatter(x=subset.end_s,y=subset.collision_probability,mode="lines+markers",line=dict(color=TEAL,width=3),name="P(class 1)"))
                fig.add_hline(y=float(subset.iloc[0].threshold),line_dash="dash",line_color=ORANGE)
                fig.update_yaxes(range=[0,1],title="Model score")
                fig.update_xaxes(title="End of observed window (s)")
                st.plotly_chart(chart_style(fig,310),use_container_width=True,config={"displayModeBar":False})
                st.caption("Window ends shown on x-axis. This is not validated prediction lead time; thresholds and calibration depend on checkpoint identity.")
        st.dataframe(scored,hide_index=True,use_container_width=True)
        st.download_button("Download inference results (CSV)",scored.to_csv(index=False).encode("utf-8"),"crashlens_inference.csv","text/csv")
    elif result is not None and key is not None:
        st.caption("Video or model settings changed. Run inference again to see predictions for these exact inputs.")
    with st.expander("Checkpoint architecture, reproducibility and deployment notes"):
        st.markdown("The implemented `NexarCollisionModel` matches `final.ipynb` cell 7 and `main.ipynb` cell 8 (MViT-v2-S, temporal Transformer, 3 CNN kernels). Files using a different backbone or classification head will be rejected with strict loading. `models/` stays local and no score is generated without weights. Uploads are processed in a temporary directory and removed after inference. Scanning an arbitrary video cannot establish that the model predicted an event before it occurred without event-time annotations and independent evaluation.")
        st.code("models/detection.pth\nmodels/anticipation.pth",language=None)
        st.caption("Do not copy the single best_mvit_collision.pth into both locations: that filename was reused for different runs and its true identity must be checked.")


def experiment_history() -> None:
    hero("Research / all recorded runs", "Training history, without missing attempts.",
         "Seven documented training curves across detection, early pre-event sampling and alert-aware anticipation. Compare epochs, read original logs, and inspect settings.")
    catalog=read_csv("experiment_catalog.csv")
    epochs=read_csv("experiment_history.csv")
    count=read_csv("project_timeline.csv")
    stat_cards([
        ("Recorded experiments",str(len(catalog)),"With settings and source"),
        ("Logged epochs",str(len(epochs)),"Actual validation/train records"),
        ("Journey milestones",str(len(count)),"Including failed attempts"),
        ("Archived reports","3","All kept in sources/")
    ])
    selected_task=st.radio("Task",["Detection","Anticipation"],horizontal=True,key="hist_task")
    subset=catalog[catalog.task==selected_task]
    selected=st.selectbox("Experiment",subset.run.tolist(),key="history_run")
    info=subset[subset.run==selected].iloc[0]
    st.markdown(f"**{info['status']}** · {info['evaluation']} · {info['source']}")
    st.caption(info['notes'])
    cards=[
        ("Best val ROC-AUC",format_metric(info['best_val_auc']),"As recorded in report"),
        ("Val Macro-F1",format_metric(info['val_macro_f1']),"May be threshold-optimized"),
        ("Test ROC-AUC",format_metric(info['test_auc']),"— if independent test absent"),
        ("Test Macro-F1",format_metric(info['test_macro_f1']),"— if independent test absent"),
    ]
    stat_cards(cards)
    cur=epochs[epochs.run==selected].sort_values("epoch")
    metric=st.segmented_control("Training metric", options=["loss","f1","auc","accuracy"],default="loss",key="history_metric")
    metric_groups={"loss":["train_loss","val_loss"],"f1":["train_f1","val_f1"],"auc":["val_auc"],"accuracy":["train_acc","val_acc"]}
    fig=go.Figure()
    for metric_col in metric_groups[metric]:
        if cur[metric_col].notna().any():
            fig.add_trace(go.Scatter(x=cur.epoch,y=cur[metric_col],mode="lines+markers",name=metric_col.replace("_"," ").title(),line=dict(width=3,color=BLUE if metric_col.startswith("train") else TEAL)))
    fig.update_xaxes(title="Epoch",dtick=1)
    fig.update_yaxes(title=metric.title())
    st.plotly_chart(chart_style(fig,370),use_container_width=True,config={"displayModeBar":False})
    st.caption(f"Training points: {len(cur)} · Frame sampling: {info['sampling']}")
    with st.expander("Raw epoch-by-epoch metrics"):
        st.dataframe(cur.drop(columns=["source","phase","task","split","run"]),hide_index=True,use_container_width=True)
        st.download_button("Download this history",cur.to_csv(index=False).encode("utf-8"),"experiment_epochs.csv","text/csv")
    st.markdown('<div class="section-title">Full experiment registry</div>',unsafe_allow_html=True)
    st.dataframe(subset[["run","status","epochs_reported","best_val_auc","val_macro_f1","test_auc","test_macro_f1","evaluation","source"]],hide_index=True,use_container_width=True)
    st.caption("Different tasks and test protocols must not be ranked as identical experiments. Experiment B has 8/15 epochs in the supplied archive; threshold-tuned validation F1 is not independent test F1.")


def experiment_graph(mapping: dict, selected_track: str) -> go.Figure:
    """Four independent research lanes; arrows show within-report progress only."""
    labels={
        "initial":"Original", "shortcut":"Shortcut", "ablation":"Ablation",
        "greedy":"Greedy failed", "pool":"Global pool", "grouped":"Group CV",
        "retrain":"Retrain", "clean_test":"Test",
        "linear":"Linear", "mlp":"MLP", "checkpoint":"Checkpoint test",
        "labels":"Targets", "A":"Attention A", "B":"Transformer B", "full_video":"Full-video",
    }
    fig=go.Figure()
    colors={"Recorded":TEAL,"Partial":ORANGE,"Failed attempt":RED}
    for row,track in enumerate(mapping["tracks"]):
        steps=track_steps(mapping,track["id"])
        y=3-row
        selected=track["id"]==selected_track
        n=len(steps)
        xx=np.linspace(0.5,14.3,n) if n>1 else np.array([7.4])
        if n>1:
            for i in range(n-1):
                fig.add_annotation(x=float(xx[i+1]-.21),y=y,ax=float(xx[i]+.20),ay=y,
                    xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=2,
                    arrowwidth=2 if selected else 1,arrowsize=.9,
                    arrowcolor="#527c93" if selected else "#374657")
        fig.add_trace(go.Scatter(
            x=xx,y=[y]*n,mode="markers+text",showlegend=False,
            name=track["title"],text=[labels.get(step["id"],step["title"]) for step in steps],
            textposition="bottom center",textfont=dict(size=11 if selected else 10,color=PAPER if selected else MUTED),
            marker=dict(size=[25 if selected else 19]*n,color=[colors[st['status']] for st in steps],
                opacity=1 if selected else .43, line=dict(color="#e2f8fa" if selected else "#374657",width=1.5 if selected else .7)),
            customdata=[[st['title'],st['short'],st['source'],st['status']] for st in steps],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Source: %{customdata[2]}<br>Status: %{customdata[3]}<extra></extra>"
        ))
        fig.add_annotation(x=-.55,y=y,text=track["title"].split('·',1)[-1].strip(),showarrow=False,
            xanchor='right',font=dict(size=12,color=PAPER if selected else MUTED))
    fig.update_xaxes(visible=False,range=[-1.35,15.2],fixedrange=True)
    fig.update_yaxes(visible=False,range=[-.62,3.6],fixedrange=True)
    fig.update_layout(hovermode="closest",dragmode=False)
    return chart_style(fig,590)


def _stage_history(run_name: str) -> None:
    history=read_csv("experiment_history.csv")
    view=history[history.run==run_name].sort_values('epoch')
    if view.empty:
        return
    st.markdown('<div class="section-title">Epoch-by-epoch record</div>',unsafe_allow_html=True)
    metric=st.radio('Metric to inspect', ['Validation ROC-AUC','Macro-F1','Loss'], horizontal=True,key='stage_training_metric')
    columns={'Validation ROC-AUC':['val_auc'],'Macro-F1':['train_f1','val_f1'],'Loss':['train_loss','val_loss']}[metric]
    fig=go.Figure()
    for field in columns:
        if view[field].notna().any():
            fig.add_trace(go.Scatter(x=view.epoch,y=view[field],mode='lines+markers',
                name=field.replace('_',' ').title(),line=dict(width=3,color=TEAL if field.startswith('val') else BLUE),
                hovertemplate='Epoch %{x}: %{y:.4f}<extra>%{fullData.name}</extra>'))
    fig.update_xaxes(title='Epoch',dtick=1)
    fig.update_yaxes(title=metric)
    st.plotly_chart(chart_style(fig,310),use_container_width=True,config={'displayModeBar':False})
    with st.expander(f'All {len(view)} logged epoch records'):
        st.dataframe(view[['epoch','train_loss','val_loss','train_f1','val_f1','val_auc','train_acc','val_acc']],
            hide_index=True,use_container_width=True)


def project_journey() -> None:
    hero('Research archive / reproducible evidence', 'Experiment graph & log explorer',
         'Understand what changed, which settings were recorded, the outcome of each attempt, and where the exact evidence appears in the original reports.')
    mapping=read_log_map()
    catalog=read_csv('experiment_catalog.csv')
    epochs=read_csv('experiment_history.csv')
    stat_cards([
        ('Research tracks','4','3 Markdown reports + final notebook'),
        ('Mapped decisions',str(len(mapping['steps'])),'Including failed and partial work'),
        ('Epoch-tracked runs',str(len(catalog)),'7 documented model experiments'),
        ('Recorded epochs',str(len(epochs)),'No fabricated history'),
    ])
    options={t['title']:t for t in mapping['tracks']}
    pick=st.selectbox('Choose a research track / source report',list(options),key='journey_track')
    track=options[pick]
    st.markdown('<div class="section-title">Research decision graph</div>',unsafe_allow_html=True)
    st.plotly_chart(experiment_graph(mapping,track['id']),use_container_width=True,config={'displayModeBar':False})
    st.caption('Arrows connect decisions within each track only; separate lanes are NOT a single chronological pipeline. Hover over a node to read its recorded outcome. Green = documented, amber = partial, red = failed attempt.')
    st.info(f"**Selected source:** `{track['source']}` — {track['description']}\n\n**Evaluation boundary:** {track['scope']}")
    steps=track_steps(mapping,track['id'])
    st.markdown('<div class="section-title">What was logged at each step?</div>',unsafe_allow_html=True)
    table=pd.DataFrame([{'Step':i+1,'Decision':s['title'],'Result':s['short'],
        'Status':s['status'],'Source section':f"§{s['section']}" if s['section'] is not None else 'Notebook'}
        for i,s in enumerate(steps)])
    st.dataframe(table,hide_index=True,use_container_width=True)
    id_map={f"{i+1:02d} · {s['title']}":s for i,s in enumerate(steps)}
    chosen=st.selectbox('Inspect one decision and its evidence',list(id_map),key=f"journey_step_{track['id']}")
    step=id_map[chosen]
    st.markdown(f"### {step['title']}")
    status_color={'Recorded':'ok','Partial':'warning','Failed attempt':'warning'}[step['status']]
    st.markdown(f'<span class="pill {status_color}">{html.escape(step["status"])}</span> &nbsp; <code>{html.escape(step["source"])}</code>',unsafe_allow_html=True)
    a,b=st.columns([1,1.04],gap='large')
    with a:
        with st.container(border=True):
            st.markdown('**What changed / what was tried**')
            st.write(step['change'])
            st.markdown('**Recorded settings**')
            st.dataframe(pd.DataFrame([{'Parameter':k,'Logged value':v} for k,v in step['settings'].items()]),
                hide_index=True,use_container_width=True)
    with b:
        with st.container(border=True):
            st.markdown('**Reported outcome**')
            st.write(step['result'])
            if step['metrics']:
                st.dataframe(pd.DataFrame(step['metrics']).rename(columns={'label':'Metric','value':'Recorded value','scope':'Evaluation / context'}),
                    hide_index=True,use_container_width=True)
            st.markdown('**Interpretation**')
            st.write(step['interpretation'])
            if step['caveat']:
                st.warning(step['caveat'])
    if step['run']:
        _stage_history(step['run'])
    with st.expander('Read the exact original source for this decision'):
        raw=report_section(step['source'],step['section'])
        if raw:
            st.markdown(raw)
        else:
            st.info('This result is in a Jupyter notebook, rather than a Markdown report. Download the original notebook to inspect the logged output and code.')
        sourcefile=BASE/'sources'/step['source']
        st.download_button('Download source file',sourcefile.read_bytes(),file_name=step['source'],
            mime='text/markdown' if step['source'].endswith('.md') else 'application/x-ipynb+json',key='download_stage_source')
    st.markdown('<div class="section-title">All tracked model runs in this source</div>',unsafe_allow_html=True)
    relevant=catalog[catalog.source==track['source']]
    if relevant.empty:
        st.caption('No individual training-run catalog rows in this source.')
    else:
        cols=['run','status','epochs_reported','best_val_auc','val_macro_f1','test_auc','test_macro_f1','evaluation']
        st.dataframe(relevant[cols],hide_index=True,use_container_width=True)
        st.caption('Validation and test values are distinct. The alert-aware A/B run has validation only; B has only 8 of 15 planned epochs in the archive.')
    st.markdown('<div class="section-title">Browse any section of the original log</div>',unsafe_allow_html=True)
    path=BASE/'sources'/track['source']
    if path.suffix=='.md':
        doc=path.read_text(encoding='utf-8')
        sections=markdown_sections(doc)
        sec=st.selectbox('Original report sections',range(len(sections)),format_func=lambda i:sections[i][0],key=f"journey_source_section_{track['id']}")
        with st.expander('Show report section verbatim'):
            st.markdown(sections[sec][1])
        st.download_button('Download full original report',doc.encode('utf-8'),file_name=track['source'],mime='text/markdown',key='download_full_log')
    else:
        st.caption('Full-video results are preserved in the source notebook, not a dedicated Markdown log.')
        st.download_button('Download final.ipynb',path.read_bytes(),file_name=track['source'],mime='application/x-ipynb+json',key='download_notebook')
    st.caption('Settings are shown only where recorded by the project. This graph documents research relationships, not comparative model superiority across different splits or protocols.')


with st.sidebar:
    st.markdown("## 🚦 CrashLens")
    st.caption("NEXAR · RESEARCH DASHBOARD")
    st.divider()
    st.markdown("**Presentation focus**")
    st.caption("Two trained models · experiment decisions · model evaluation")
    st.caption("Detailed score CSVs are optional and can be imported directly on Error Analysis.")
    st.divider()
    st.markdown('<span class="pill ok">Archived results</span>', unsafe_allow_html=True)
    st.caption("Archived evidence is always available. Inference requires genuine checkpoints.")
    st.divider()
    st.caption("Detection ≠ pre-event anticipation")

pages = [
    st.Page(overview, title="Overview", icon="📊", default=True),
    st.Page(inference_console, title="Run Two Models", icon="🎯"),
    st.Page(project_journey, title="Experiment Graph & Logs", icon="🧭"),
    st.Page(experiment_history, title="Training History", icon="📈"),
    st.Page(model_lab, title="Model Lab", icon="🧪"),
    st.Page(data_audit, title="Data Audit", icon="🛡️"),
    st.Page(error_lab, title="Error Analysis", icon="🔍"),
]
pg = st.navigation(pages)
pg.run()
