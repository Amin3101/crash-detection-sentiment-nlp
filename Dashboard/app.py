"""CrashLens: integrated Computer Vision and NLP presentation dashboard.

Run with `streamlit run app.py`. Archived research metrics remain separate from
live inference; no predictions are fabricated without a deployable checkpoint.
"""
from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import precision_recall_curve, roc_curve

from analytics import PredictionValidationError, evaluate, validate_predictions, safe_video_path
from log_explorer import read_log_map, track_steps, report_section, markdown_sections
from nlp_runtime import find_training_csv, train_deployable_classical, predict_reviews, load_classical_artifact

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
    page_title="CrashLens | CV + NLP Intelligence",
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


def model_lab(compact: bool = False) -> None:
    if compact:
        st.markdown("### Archived model comparisons & ablations")
    else:
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


def data_audit(compact: bool = False) -> None:
    if compact:
        st.markdown("### Data quality & leakage controls")
    else:
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


def error_lab(compact: bool = False) -> None:
    if compact:
        st.markdown("### Error analysis & per-video scores")
    else:
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


def experiment_history(compact: bool = False) -> None:
    if compact:
        st.markdown("### Epoch-by-epoch training history")
    else:
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


def project_journey(compact: bool = False) -> None:
    if compact:
        st.markdown("### Experiment graph & source logs")
    else:
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


# -----------------------------------------------------------------------------
# CrashLens v4: independently sourced CV architecture/EDA and NLP research pages.
# -----------------------------------------------------------------------------
def architecture_dot(kind: str) -> str:
    is_detection = kind == "Detection"
    mode = "16 frames evenly across full video" if is_detection else "16 frames from a 4 s pre-event window"
    src = "final.ipynb · cell 7" if is_detection else "main.ipynb · cell 8"
    dot = r'''
digraph model {
  graph [bgcolor="transparent", rankdir=TB, pad="0.15", nodesep="0.25", ranksep="0.35", splines=ortho];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, fontcolor="#eaf5fc", color="#39546a", fillcolor="#173044", penwidth=1.1, margin="0.15,0.10"];
  edge [color="#7b9db2", arrowsize=0.65, penwidth=1.4];
  source [label="INPUT_LABEL", fillcolor="#174b5a", color="#59c5c4"];
  tensor [label="RGB + resize 224 × 224\\nnormalize → [B, 3, 16, 224, 224]"];
  backbone [label="MViT-v2-S backbone\\nspatiotemporal transformer blocks", fillcolor="#234568", color="#7daaf4"];
  cls [label="CLS token\\n768-d"];
  patches [label="Patch tokens [B, T', H', W', 768]"];
  spatial [label="Spatial mean + max → concat\\nLinear + LayerNorm + GELU → 768-d"];
  trans [label="Temporal Transformer branch\\npositional embedding + 2 encoder layers\\n8 attention heads → attention pool", fillcolor="#204c59", color="#54cbbf"];
  cnn [label="Temporal CNN branches\\nkernels 3 / 5 / 7 → fuse 1×1\\nattention pool", fillcolor="#204c59", color="#54cbbf"];
  mean [label="Temporal mean / max\\n2 × 768-d"];
  fuse [label="Transformer + CNN features\\nconcat → Linear + Norm + GELU\\n768-d attention fusion", fillcolor="#3a3b64", color="#a6a1f5"];
  concat [label="Concat: CLS + fused + mean + max\\n4 × 768 = 3072-d"];
  head [label="LayerNorm → Linear 512 → 256 → 2\\nDropout + GELU", fillcolor="#564632", color="#f1b76c"];
  out [label="Softmax → P(no collision), P(collision)\\nOUTPUT_LABEL", fillcolor="#17483d", color="#65d6bb"];
  source -> tensor -> backbone;
  backbone -> cls;
  backbone -> patches;
  patches -> spatial;
  spatial -> trans;
  spatial -> cnn;
  spatial -> mean;
  trans -> fuse;
  cnn -> fuse;
  cls -> concat;
  mean -> concat;
  fuse -> concat;
  concat -> head -> out;
  {rank=same; trans; cnn; mean}
}
'''
    return (dot.replace(r"\\n", r"\n").replace("INPUT_LABEL",mode)
            .replace("OUTPUT_LABEL", "Collision within observed video" if is_detection else "Risk score for sampled pre-event clip"))


def cv_architecture() -> None:
    hero("Computer vision / network design", "Two model architecture trees",
         "Structures traced from final.ipynb and main.ipynb. The branches look similar, but the temporal inputs and evaluation protocols are different.")
    tab_detection, tab_anticipation, tab_difference = st.tabs([
        "01 · Detection tree", "02 · Anticipation tree", "03 · Compare protocols"
    ])
    with tab_detection:
        st.caption("Source: `final.ipynb`, class NexarCollisionModel · full-video input")
        st.graphviz_chart(architecture_dot("Detection"), use_container_width=True)
        st.info("**Input:** 16 uniformly selected frames across the full video. Frames after the accident can be included, so this is event detection, not a pre-crash forecast.")
        png=BASE/'assets'/'architecture_detection.png'
        if png.exists():
            st.download_button("Download Detection architecture (PNG)",png.read_bytes(),file_name=png.name,mime='image/png',key='download_detection_tree')
    with tab_anticipation:
        st.caption("Source: `main.ipynb`, class NexarCollisionModel · four-second temporal input")
        st.graphviz_chart(architecture_dot("Anticipation"), use_container_width=True)
        st.info("**Input:** 16 frames from the chosen four-second pre-event window. An experiment requiring a known event timestamp is not equivalent to unconstrained real-time forecasting.")
        png=BASE/'assets'/'architecture_anticipation.png'
        if png.exists():
            st.download_button("Download Anticipation architecture (PNG)",png.read_bytes(),file_name=png.name,mime='image/png',key='download_anticipation_tree')
    with tab_difference:
        st.dataframe(pd.DataFrame([
          {"Feature":"Source architecture", "Detection":"final.ipynb · cell 7", "Anticipation":"main.ipynb · cell 8"},
          {"Feature":"Temporal input", "Detection":"16 frames across entire video", "Anticipation":"16 frames from pre-event 4s clip"},
          {"Feature":"Post-collision frames", "Detection":"May be included", "Anticipation":"Not in a correctly pre-event sampled clip"},
          {"Feature":"Backbone & head", "Detection":"MViT-v2-S + dual temporal branch", "Anticipation":"MViT-v2-S + dual temporal branch"},
          {"Feature":"Reported test AUC", "Detection":"0.944 · full-video run", "Anticipation":"0.785 · EARLIER clean-data model, not main.ipynb Model B"},
          {"Feature":"New main.ipynb Model B", "Detection":"Not applicable", "Anticipation":"Partial validation run; no independent test"},
        ]),hide_index=True,use_container_width=True)
        st.caption("The same layer layout has two distinct training/evaluation problems. Weight files must be checked individually.")
    st.markdown('<div class="section-title">Shared building blocks</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a:
        st.markdown("**Backbone / spatial representation**")
        st.write("MViT-v2-S features; class token plus patch tokens. Spatial mean/max pooling projects patch features to 768 dimensions.")
    with b:
        st.markdown("**Two temporal paths**")
        st.write("A two-layer Transformer encoder (8 heads) and a multi-scale temporal CNN (kernels 3, 5, 7) with learned attention pooling.")
    with c:
        st.markdown("**Fusion / binary head**")
        st.write("The CLS token, fused temporal features, temporal mean and max are concatenated (3072 dimensions), then classified into 2 classes.")
    st.warning("**Checkpoint caveat:** identical architecture does not mean interchangeable trained weights. The archived 0.785 pre-event test AUC belongs to an earlier clean-data run; it is not proof that the later `main.ipynb` Model B has the same test score. That newer run has only partial validation logs.")
    with st.expander("Exact architecture sources and model-loading details"):
        st.write("Detection: `sources/final.ipynb`, cell 7; Anticipation: `sources/main.ipynb`, cell 8. Both define `MultiScaleTemporalPool` and `NexarCollisionModel` with the same main layer layout.")
        st.write("Implementation used for live video inference: `model_runtime.py`, with two separate checkpoint paths. Checkpoint identity and preprocessing must be verified before showcasing real inference.")


def cv_eda() -> None:
    hero("Computer vision / dataset profiling", "Nexar dataset EDA",
         "Original class and metadata exploration from the supplied nexar-EDA.ipynb. These plots describe the 1,500-row EDA sample, not the later cleaned training split.")
    stat_cards([
        ("Metadata rows", "1,500", "EDA dataframe"),
        ("No collision", "750", "Class 0"),
        ("Collision", "750", "Class 1"),
        ("Weather missing", "2", "Notebook output · cell 4"),
    ])
    st.markdown('<div class="section-title">Original EDA charts</div>',unsafe_allow_html=True)
    plots=[('Weather', 'weather.png'), ('Scene', 'scene.png'), ('Light conditions', 'light_conditions.png')]
    for label, filename in plots:
        with st.expander(label, expanded=label=='Weather'):
            st.image(str(BASE/'assets'/'eda'/filename), use_container_width=True,
                     caption=f"Original {label.lower()} distribution within class · nexar-EDA.ipynb, cell 7")
    st.markdown('<div class="section-title">Recorded missing values</div>',unsafe_allow_html=True)
    missing=pd.DataFrame([
        {"Column":"time_of_event", "Missing":750, "Share":0.5},
        {"Column":"time_of_alert", "Missing":750, "Share":0.5},
        {"Column":"weather", "Missing":2, "Share":2/1500},
        {"Column":"time_to_accident", "Missing":1500, "Share":1.0},
        {"Column":"scene", "Missing":0, "Share":0.0},
        {"Column":"light_conditions", "Missing":0, "Share":0.0},
    ])
    fig=go.Figure(go.Bar(x=missing['Column'],y=missing['Share']*100,
              marker_color=[RED if v>0.5 else TEAL for v in missing['Share']],
              text=missing['Missing'],textposition='outside'))
    fig.update_yaxes(range=[0,111],title='Missing values (%)');fig.update_xaxes(tickangle=-25)
    st.plotly_chart(chart_style(fig,330),use_container_width=True,config={'displayModeBar':False})
    insight("Event and alert timestamps are absent for the 750 negative cases. Missingness here can encode the target, so these fields must not be used naively as non-visual predictors. The time_to_accident column is entirely missing in this EDA output.")
    with st.expander("Original notebook / audit notes"):
        st.write("Data source: `nexar-EDA.ipynb`, cell 4 (shape, class counts, missingness) and cell 7 (three saved matplotlib figures). Download the original notebook for the full extraction code.")
        p=BASE/'sources'/'nexar-EDA.ipynb'
        st.download_button("Download original Nexar EDA notebook",p.read_bytes(),file_name=p.name,mime='application/x-ipynb+json',key='download_cv_eda')
    st.caption("These are notebook outputs, not reconstructed distributions. The later balanced/deduplicated data have a different sample count.")


def cv_overview_v5() -> None:
    hero(
        "Computer vision / presentation overview",
        "Crash detection & anticipation",
        "Two video-model tasks, their architectures and key test results. Detailed evidence remains available in the Research & Logs section."
    )
    stat_cards([
        ("Detection · Test AUC", "0.944", "Full-video · 225 test videos"),
        ("Detection · Test Macro-F1", "0.889", "Full-video · final.ipynb"),
        ("Pre-event · Test AUC", "0.785", "Earlier clean-data run · log.md"),
        ("Pre-event · Test Macro-F1", "0.729", "Earlier clean-data run · 170 clips"),
    ])
    st.caption("The pre-event test metrics belong to the earlier clean-data run, **not** the later Model B in `main.ipynb`. Different evaluation protocols; these figures are not a head-to-head comparison.")
    st.markdown('<div class="section-title">Two model architecture trees</div>', unsafe_allow_html=True)
    st.write("Both use the MViT-v2-S backbone and dual temporal branches. The temporal sampling protocol is what changes their intended use.")
    detection_tab, anticipation_tab, differences_tab = st.tabs(["🎯 Detection", "⏱️ Anticipation", "Compare input protocols"])
    with detection_tab:
        a,b=st.columns([1.2,0.8],gap="large")
        with a:
            pic=BASE/'assets'/'architecture_detection.png'
            st.image(str(pic),use_container_width=True,caption="Source: final.ipynb · NexarCollisionModel")
            st.download_button("Save Detection architecture PNG",pic.read_bytes(),file_name=pic.name,mime='image/png',key='overview_tree_detection')
        with b:
            st.markdown("#### Detection · observed video")
            st.write("Samples **16 frames across the full video** and classifies whether a collision is visible in those frames.")
            st.markdown("**Shared model layers**")
            st.write("MViT-v2-S → spatial token pooling → temporal Transformer + multi-kernel CNN → attention fusion → binary classification head.")
            st.info("This path may see post-collision frames, so it must not be presented as advance accident forecasting.")
    with anticipation_tab:
        a,b=st.columns([1.2,0.8],gap="large")
        with a:
            pic=BASE/'assets'/'architecture_anticipation.png'
            st.image(str(pic),use_container_width=True,caption="Source: main.ipynb · NexarCollisionModel")
            st.download_button("Save Anticipation architecture PNG",pic.read_bytes(),file_name=pic.name,mime='image/png',key='overview_tree_anticipation')
        with b:
            st.markdown("#### Anticipation · pre-event clip")
            st.write("Samples **16 frames from a four-second input window** selected before the annotated event, or from a rolling window at inference time.")
            st.markdown("**Shared model layers**")
            st.write("MViT-v2-S → spatial token pooling → temporal Transformer + multi-kernel CNN → attention fusion → binary classification head.")
            st.warning("The later `main.ipynb` Model B has partial validation history only; no independent held-out test metric was provided for that checkpoint.")
    with differences_tab:
        st.dataframe(pd.DataFrame([
            {"Dimension":"Input", "Detection":"16 frames from full video", "Anticipation":"16 frames from pre-event 4 s window"},
            {"Dimension":"Can include frames after impact?", "Detection":"Yes", "Anticipation":"No (when correctly sampled)"},
            {"Dimension":"Architecture", "Detection":"MViT-v2-S + dual temporal branches", "Anticipation":"MViT-v2-S + dual temporal branches"},
            {"Dimension":"Relevant notebook", "Detection":"final.ipynb", "Anticipation":"main.ipynb"},
            {"Dimension":"Reported test metric", "Detection":"Full-video AUC 0.944", "Anticipation":"Earlier clean pre-event AUC 0.785"},
        ]),hide_index=True,use_container_width=True)
        st.caption("Weights are **not interchangeable** simply because the architectures look similar. Verify the checkpoint and preprocessing of each training run.")
    st.markdown('<div class="section-title">Original Nexar dataset · at a glance</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("EDA records","1,500","Original notebook sample")
    b.metric("Collision / no collision","750 / 750","Equal class counts")
    c.metric("Weather missing","2","Original sample")
    with st.expander("Inspect EDA figures and missing-value audit",expanded=False):
        chosen=st.selectbox("Original notebook plot",["Weather","Scene","Light conditions"],key='overview_eda_chart')
        names={"Weather":"weather.png","Scene":"scene.png","Light conditions":"light_conditions.png"}
        st.image(str(BASE/'assets'/'eda'/names[chosen]),use_container_width=True,caption=f"Original Nexar EDA plot · {chosen.lower()} · nexar-EDA.ipynb")
        st.dataframe(pd.DataFrame([
            {"Field":"time_of_event", "Missing":750},
            {"Field":"time_of_alert", "Missing":750},
            {"Field":"weather", "Missing":2},
            {"Field":"time_to_accident", "Missing":1500},
            {"Field":"scene", "Missing":0},
            {"Field":"light_conditions", "Missing":0},
        ]),hide_index=True,use_container_width=True)
        st.warning("Event and alert timestamps are absent for all 750 negative cases. Missingness can expose the label. Original EDA and cleaned training split are distinct datasets.")
    with st.expander("Extra evaluation chart · recorded experimental performance",expanded=False):
        task=st.radio("Protocol",["Detection","Anticipation"],horizontal=True,key="overview_v5_task")
        runs=read_csv("runs.csv")
        sub=runs[runs.task==task]
        fig=go.Figure()
        for metric,color,label in [("macro_f1",TEAL,"Macro-F1"),("roc_auc",BLUE,"ROC-AUC")]:
            fig.add_trace(go.Bar(x=sub.name,y=sub[metric],name=label,marker_color=color,
                   text=[format_metric(x) for x in sub[metric]],textposition='outside'))
        fig.update_layout(barmode='group')
        fig.update_yaxes(range=[0,1.08])
        st.plotly_chart(chart_style(fig,340),use_container_width=True,config={'displayModeBar':False})
        st.caption("Recorded test runs only. Separate datasets, sampling choices and checkpoints limit direct comparisons.")
    insight("Presentation flow: introduce the two tasks here → demonstrate a verified checkpoint in Video Demo → show the decision graph and training evidence → discuss CV errors and limits.")


def cv_research_v5() -> None:
    hero("Computer vision / research evidence", "Experiment graph, training & data integrity",
         "One research page with four focused views. The original log mapping, every documented epoch and all ablations are retained.")
    selection=st.segmented_control("Research view",
        ["Experiment graph & logs","Training history","Ablations & model lab","Data leakage audit"],
        default="Experiment graph & logs",key='cv_research_view') or "Experiment graph & logs"
    if selection=="Experiment graph & logs":
        project_journey(compact=True)
    elif selection=="Training history":
        experiment_history(compact=True)
    elif selection=="Ablations & model lab":
        model_lab(compact=True)
    elif selection=="Data leakage audit":
        data_audit(compact=True)


def cv_results_v5() -> None:
    hero("Computer vision / measured results", "Evaluation & error analysis",
         "Show test-set results and failure modes without conflating full-video detection with pre-event anticipation.")
    with st.expander("Recorded independent test runs",expanded=False):
        runs=read_csv("runs.csv")
        st.dataframe(runs[["task","name","samples","macro_f1","roc_auc","accuracy","source"]],
                     hide_index=True,use_container_width=True)
        st.caption("Different tasks, time windows and evaluation sets; not a controlled same-dataset ranking.")
    error_lab(compact=True)



def _nlp_dashboard_html() -> str:
    """Load the team's original self-contained dashboard and apply CrashLens dark styling."""
    dashboard_path = BASE / "assets" / "nlp_team_dashboard.html"
    if not dashboard_path.is_file():
        raise FileNotFoundError(f"Team NLP dashboard is missing: {dashboard_path}")
    raw = dashboard_path.read_text(encoding="utf-8")
    theme = """
<style>
:root {
  color-scheme: dark;
  --bg:#0a1420;
  --card:#132638;
  --ink:#e8f1f8;
  --muted:#9cafbe;
  --line:#294257;
  --accent:#5cd8d0;
  --pos:#5cd8d0;
  --neg:#f27b7b;
  --neu:#78a6f8;
  --warn:#f3b568;
}
body { background:var(--bg); color:var(--ink); font-family:'DM Sans','Segoe UI',sans-serif; }
header { padding:20px 22px 8px; }
header h1 { color:#f3f8fe; letter-spacing:-.035em; }
header p { color:#afc1d0; }
nav { background:var(--bg); border-bottom-color:var(--line); }
nav button { background:#132638; border-color:#294257; color:#dbe8f2; }
nav button.on { background:#246c70; border-color:#5cd8d0; color:#f4ffff; }
main { max-width:1450px; padding-top:18px; }
.card,.kpi { background:linear-gradient(160deg,#152a3c,#11202e); border-color:#294257; color:var(--ink); }
.card h2,.story { color:var(--ink); }
.card .sub,.note,.kpi span,.ctl { color:var(--muted); }
th { color:#b2c2d0; }
td { color:#e5eef6; }
th,td { border-bottom-color:#294257; }
.ctl select,.ctl input[type=text] { background:#0c1b29; color:#e8f1f8; border-color:#38536a; }
.story { border-left-color:#5cd8d0; }
.tag { border-color:#38536a; color:#b8cbd9; }
</style>
"""
    return raw.replace("</head>", theme + "\n</head>")


def _nlp_model_status() -> tuple[Path, Path | None]:
    model_path = BASE / "models" / "nlp_classical.joblib"
    train_csv = find_training_csv()
    return model_path, train_csv


def _nlp_training_panel(model_path: Path, train_csv: Path | None) -> None:
    st.markdown("### Prepare a runnable classical model")
    if model_path.is_file():
        try:
            artifact = load_classical_artifact(model_path)
            metrics = artifact.get("validation_metrics", {})
            stat_cards([
                ("Checkpoint", "Ready", "Classical TF-IDF + Logistic Regression"),
                ("Training rows", f"{artifact.get('n_train', 0):,}", "Rows used for fitting"),
                ("Validation B Micro-F1", format_metric(metrics.get("validation_b_micro_f1")), "Natural validation distribution"),
                ("Validation B Macro-F1", format_metric(metrics.get("validation_b_macro_f1")), "Natural validation distribution"),
            ])
            st.caption(f"Local checkpoint: `{model_path.relative_to(BASE)}`")
        except Exception as exc:
            st.error(f"Checkpoint could not be loaded: {exc}")
    else:
        st.warning("No deployable NLP checkpoint is present yet.")
        st.caption(
            "The supplied team archive contains validation/test probability arrays and submission files, "
            "but it does not contain the fitted TF-IDF vectorizers, Logistic Regression estimator, "
            "or Transformer weights required to score new text."
        )

    if train_csv is None:
        st.info(
            "The raw labelled training file is also absent from the supplied ZIP. "
            "Add the team's original `train_data.csv` to the path below before training."
        )
        st.code("sources/nlp_team/amazon-sentiment/data/train_data.csv", language=None)
        st.caption(
            "The training file is not copied into this release. The script uses the team's original "
            "columns: overall, reviewText, summary, reviewerID and asin."
        )
        return

    st.caption(f"Training data found: `{train_csv.relative_to(BASE) if BASE in train_csv.parents else train_csv}`")
    sample_size = st.select_slider(
        "Training sample size",
        options=[50_000, 100_000, 200_000],
        value=200_000,
        help="200,000 follows the sample size in the team's main classical run. Smaller options reduce memory and training time.",
        key="nlp_train_sample_size",
    )
    if st.button("Train / refresh classical NLP checkpoint", type="primary", use_container_width=True):
        try:
            with st.status("Training the deployable classical branch…", expanded=True) as status:
                result = train_deployable_classical(
                    train_csv=train_csv,
                    output_path=model_path,
                    n_train=sample_size,
                    log_callback=lambda message: status.write(message),
                )
                status.update(label="Classical NLP checkpoint saved.", state="complete", expanded=False)
            st.success(
                f"Checkpoint saved. Validation B micro-F1: "
                f"{result['metrics']['validation_b_micro_f1']:.4f}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.caption("Check the dataset columns, available memory and installed dependencies.")


def _nlp_single_review(model_path: Path) -> None:
    st.markdown("### Score one review")
    with st.form("nlp_single_review_form"):
        summary = st.text_input("Review summary (optional)")
        review_text = st.text_area("Review text", height=170, placeholder="Paste the customer's review here…")
        c1, c2 = st.columns(2)
        with c1:
            reviewer_id = st.text_input("Reviewer ID (optional)")
        with c2:
            asin = st.text_input("Product ASIN (optional)")
        submitted = st.form_submit_button("Run classical model", type="primary", use_container_width=True)

    if submitted:
        if not review_text.strip():
            st.error("Enter review text first.")
            return
        frame = pd.DataFrame([{
            "summary": summary,
            "reviewText": review_text,
            "reviewerID": reviewer_id or pd.NA,
            "asin": asin or pd.NA,
        }])
        try:
            result = predict_reviews(frame, model_path)
            probabilities = result["calibrated_probabilities"][0]
            predicted = int(np.argmax(probabilities) + 1)
            left, right = st.columns([0.75, 1.25], gap="large")
            with left:
                st.metric("Predicted rating", f"{predicted} / 5 stars")
                st.caption("Calibrated classical-model output. No label-shift correction is applied to a single review.")
            with right:
                fig = go.Figure(go.Bar(
                    x=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
                    y=probabilities,
                    marker_color=[RED, ORANGE, MUTED, BLUE, TEAL],
                    text=[f"{p:.1%}" for p in probabilities],
                    textposition="outside",
                ))
                fig.update_yaxes(range=[0, min(1.12, max(0.25, float(probabilities.max()) + 0.12))],
                                 title="Calibrated probability")
                st.plotly_chart(chart_style(fig, 320), use_container_width=True,
                                config={"displayModeBar": False})
            st.dataframe(pd.DataFrame({
                "Rating": [1, 2, 3, 4, 5],
                "Probability": probabilities,
            }).assign(Probability=lambda d: d["Probability"].map(lambda x: f"{x:.2%}")),
                hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def _nlp_batch_reviews(model_path: Path) -> None:
    st.markdown("### Score a CSV batch")
    uploaded = st.file_uploader(
        "Upload CSV with a `reviewText` column",
        type=["csv"],
        key="nlp_batch_upload",
        help="Optional columns: summary, reviewerID, asin.",
    )
    if uploaded is None:
        st.caption("The output includes the predicted rating and the probability of each rating from 1 to 5.")
        return
    import hashlib
    upload_bytes = uploaded.getvalue()
    upload_key = hashlib.sha256(upload_bytes).hexdigest()
    try:
        frame = pd.read_csv(BytesIO(upload_bytes))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if "reviewText" not in frame.columns:
        st.error("The uploaded CSV must contain a column named `reviewText`.")
        return
    st.caption(f"{len(frame):,} reviews loaded · {len(frame.columns)} columns")
    use_em = st.checkbox(
        "Apply exploratory EM label-shift correction to this batch",
        value=False,
        disabled=len(frame) < 1000,
        help="Only available for 1,000+ rows. This assumes the class-conditional text distribution stays stable; it is not a verified correction for arbitrary user data.",
        key="nlp_batch_em",
    )
    if len(frame) < 1000:
        st.caption("EM correction is disabled for small batches. Use calibrated probabilities only.")
    if st.button("Run NLP model on uploaded CSV", type="primary", use_container_width=True):
        try:
            with st.spinner("Scoring reviews…"):
                result = predict_reviews(frame, model_path, apply_label_shift=use_em)
            probs = result["calibrated_probabilities"]
            output = frame.copy()
            output["predicted_rating"] = np.argmax(probs, axis=1) + 1
            for i in range(5):
                output[f"rating_{i+1}_probability"] = probs[:, i]
            if use_em and result["corrected_probabilities"] is not None:
                corrected = result["corrected_probabilities"]
                output["predicted_rating_em_corrected"] = np.argmax(corrected, axis=1) + 1
                for i in range(5):
                    output[f"rating_{i+1}_probability_em_corrected"] = corrected[:, i]
                st.info(
                    "Estimated batch rating mix (EM): "
                    + " · ".join(f"{i+1}★ {p:.1%}" for i, p in enumerate(result["estimated_class_mix"]))
                )
            st.session_state["nlp_batch_result"] = output
            st.session_state["nlp_batch_key"] = upload_key + (":em" if use_em else ":calibrated")
        except Exception as exc:
            st.error(f"Batch inference failed: {exc}")

    output = st.session_state.get("nlp_batch_result")
    if output is not None and st.session_state.get("nlp_batch_key") == upload_key + (":em" if use_em else ":calibrated"):
        st.markdown('<div class="section-title">Prediction output</div>', unsafe_allow_html=True)
        st.dataframe(output.head(100), hide_index=True, use_container_width=True)
        st.caption(f"Showing the first {min(100, len(output)):,} of {len(output):,} scored reviews.")
        st.download_button(
            "Download predictions CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="nlp_batch_download",
        )


def nlp_workspace() -> None:
    hero(
        "NLP / Amazon electronics reviews",
        "Interactive research dashboard & model runner",
        "The team's original eight-tab dashboard is integrated below. The runner is a separate, local inference workflow and does not alter the archived evaluation results."
    )
    stat_cards([
        ("Training reviews", "838,944", "Amazon electronics · labelled"),
        ("Products", "90,715", "Unique products in training data"),
        ("Unlabelled test rows", "20,000", "No ground-truth test labels"),
        ("Ensemble micro-F1", "0.6124", "Validation simulation · estimated test mix"),
    ])
    view = st.segmented_control(
        "NLP workspace",
        ["Interactive research dashboard", "Run NLP model"],
        default="Interactive research dashboard",
        key="nlp_workspace_view",
    ) or "Interactive research dashboard"

    if view == "Interactive research dashboard":
        st.caption(
            "Original team dashboard · 8 internal tabs · all chart data is embedded in the HTML; "
            "no raw dataset or internet connection is required to view it."
        )
        try:
            components.html(_nlp_dashboard_html(), height=1750, scrolling=True)
        except Exception as exc:
            st.error(f"Could not load the embedded NLP dashboard: {exc}")
            st.caption("The original standalone file is available at `assets/nlp_team_dashboard.html`.")
    else:
        model_path, train_csv = _nlp_model_status()
        classical_ready = False
        if model_path.is_file():
            try:
                load_classical_artifact(model_path)
                classical_ready = True
            except Exception:
                classical_ready = False
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.metric("Classical checkpoint", "Ready" if classical_ready else "Not trained")
        with c2:
            st.metric("Transformer weights", "Not supplied")
        with c3:
            st.metric("Live ensemble", "Unavailable")
        st.warning(
            "This console runs the classical TF-IDF + Logistic Regression branch after its local checkpoint is trained. "
            "The final classical + Transformer ensemble shown in the research dashboard cannot run on new text yet, "
            "because the Transformer weights were not included in the team's ZIP."
        )
        _nlp_training_panel(model_path, train_csv)
        if model_path.is_file():
            st.divider()
            single_tab, batch_tab = st.tabs(["Single review", "CSV batch"])
            with single_tab:
                _nlp_single_review(model_path)
            with batch_tab:
                _nlp_batch_reviews(model_path)
        with st.expander("What is required for full ensemble inference?"):
            st.markdown(
                "- The deployable classical checkpoint is created by the training workflow above.\n"
                "- The Transformer branch needs its matching fine-tuned model weights, tokenizer files and exact base-model identifier.\n"
                "- The archive contains `classical_probs.npz` and `transformer_probs.npz`, which are validation/test probability arrays. They cannot score a new review by themselves.\n"
                "- Until the Transformer checkpoint is supplied and verified, do not label the live output as the final ensemble."
            )


with st.sidebar:
    st.markdown("## 🚦 CrashLens")
    st.caption("FINAL PRESENTATION EDITION · v6")
    st.divider()
    st.markdown("**Two independent workstreams**")
    st.caption("Computer Vision · Nexar video models")
    st.caption("NLP · Amazon review ratings 1–5")
    st.divider()
    st.markdown('<span class="pill ok">Research results</span>',unsafe_allow_html=True)
    st.caption("Video and text inference require local model checkpoints. No synthetic predictions are shown.")
    st.divider()
    st.caption("Two workstreams · Five pages · One integrated NLP workspace")

pages = {
    "01 · COMPUTER VISION / 4 PAGES": [
        st.Page(cv_overview_v5, title="CV Overview & Architecture", icon="📊", default=True),
        st.Page(inference_console, title="Video Demo · Two Models", icon="🎯"),
        st.Page(cv_research_v5, title="Experiments & Data Audit", icon="🧭"),
        st.Page(cv_results_v5, title="CV Evaluation & Errors", icon="🔍"),
    ],
    "02 · NLP / 1 PAGE": [
        st.Page(nlp_workspace, title="NLP Dashboard & Run Model", icon="📝"),
    ],
}
pg = st.navigation(pages)
pg.run()
