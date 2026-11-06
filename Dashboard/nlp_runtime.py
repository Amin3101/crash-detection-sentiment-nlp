"""Deployable NLP inference utilities for CrashLens.

The archived NLP dashboard contains validation/test probabilities, not a fitted
text model. This module can train a reproducible classical branch from the
original train_data.csv, save its vectorizers/classifier, and run new reviews.
It does not pretend to reproduce the missing Transformer checkpoint or ensemble.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
TEAM_PROJECT = BASE / "sources" / "nlp_team" / "amazon-sentiment"
if str(TEAM_PROJECT) not in sys.path:
    sys.path.insert(0, str(TEAM_PROJECT))

from src.text import clean_text  # noqa: E402

CLASSES = np.arange(1, 6, dtype=np.int64)
SEED = 42
SMOOTHING = 3.0


def find_training_csv() -> Path | None:
    """Find the original training CSV in the documented project locations."""
    candidates = [
        TEAM_PROJECT / "data" / "train_data.csv",
        BASE / "data" / "nlp" / "train_data.csv",
        BASE / "data" / "train_data.csv",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _encode(keys_pool, y_pool, keys_apply, global_mean: float, smoothing: float) -> np.ndarray:
    """Smoothed target encoding; unknown/missing keys fall back to the global mean."""
    pool_keys = pd.Series(keys_pool, dtype="string")
    apply_keys = pd.Series(keys_apply, dtype="string")
    frame = pd.DataFrame({"key": pool_keys, "y": np.asarray(y_pool, dtype=np.float64)})
    grouped = frame.dropna(subset=["key"]).groupby("key")["y"].agg(["sum", "count"])
    sums = apply_keys.map(grouped["sum"]).fillna(0.0).to_numpy(dtype=np.float64)
    counts = apply_keys.map(grouped["count"]).fillna(0.0).to_numpy(dtype=np.float64)
    return ((sums + smoothing * global_mean) / (counts + smoothing)).astype(np.float32)


def _build_target_map(keys, y) -> dict[str, tuple[float, int]]:
    frame = pd.DataFrame({"key": pd.Series(keys, dtype="string"), "y": np.asarray(y, dtype=np.float64)})
    grouped = frame.dropna(subset=["key"]).groupby("key")["y"].agg(["sum", "count"])
    return {str(k): (float(r["sum"]), int(r["count"])) for k, r in grouped.iterrows()}


def _lookup_target_features(values, mapping: dict[str, tuple[float, int]],
                            global_mean: float, smoothing: float) -> np.ndarray:
    out = np.full(len(values), global_mean, dtype=np.float32)
    for i, value in enumerate(pd.Series(values, dtype="string").tolist()):
        if pd.isna(value):
            continue
        record = mapping.get(str(value))
        if record is not None:
            total, count = record
            out[i] = (total + smoothing * global_mean) / (count + smoothing)
    return out


def _fit_bcts(probabilities: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    z = np.log(np.clip(probabilities, 1e-9, 1.0))

    def objective(params):
        temperature = np.exp(params[0])
        calibrated_logp = log_softmax(z / temperature + params[1:], axis=1)
        return -calibrated_logp[np.arange(len(y)), y - 1].mean()

    result = minimize(objective, np.zeros(6), method="L-BFGS-B")
    return float(np.exp(result.x[0])), result.x[1:].astype(np.float64)


def _apply_bcts(probabilities: np.ndarray, temperature: float, bias: np.ndarray) -> np.ndarray:
    return softmax(np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature + bias, axis=1)


def _em_mix(probabilities: np.ndarray, source_prior: np.ndarray, iterations: int = 300) -> np.ndarray:
    source = np.clip(np.asarray(source_prior, dtype=np.float64), 1e-9, 1.0)
    q = source.copy()
    for _ in range(iterations):
        weighted = probabilities * (q / source)
        row_sums = np.clip(weighted.sum(axis=1, keepdims=True), 1e-12, None)
        q = (weighted / row_sums).mean(axis=0)
    return q / q.sum()


def train_deployable_classical(
    train_csv: str | Path,
    output_path: str | Path,
    n_train: int = 200_000,
    validation_size: int = 40_000,
    max_features_review: int = 300_000,
    max_features_summary: int = 100_000,
    min_df: int = 3,
    log_callback: Callable[[str], Any] | None = None,
) -> dict:
    """Train and save the project's classical TF-IDF + history branch.

    The training recipe follows scripts/final_model.py. It stores a deployable
    artifact, which the original experiment script did not save.
    """
    def log(message: str):
        if log_callback:
            log_callback(message)

    train_csv = Path(train_csv)
    output_path = Path(output_path)
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training file not found: {train_csv}")

    log(f"Reading training data: {train_csv}")
    data = pd.read_csv(
        train_csv,
        usecols=["overall", "reviewText", "summary", "reviewerID", "asin"],
        dtype={"reviewerID": "string", "asin": "string"},
        low_memory=False,
    )
    data["overall"] = pd.to_numeric(data["overall"], errors="coerce")
    data = data.dropna(subset=["overall"]).copy()
    data["overall"] = data["overall"].astype("int8")
    data = data[data["overall"].isin(CLASSES)].reset_index(drop=True)
    if len(data) <= validation_size:
        raise ValueError(
            f"Need more than {validation_size:,} labelled rows; found {len(data):,}."
        )

    pool, val = train_test_split(
        data, test_size=validation_size, random_state=SEED, stratify=data["overall"]
    )
    val_a, val_b = train_test_split(
        val, test_size=0.5, random_state=SEED, stratify=val["overall"]
    )
    del data
    y_pool = pool["overall"].to_numpy(dtype=np.int64)
    global_mean = float(y_pool.mean())
    rng = np.random.default_rng(SEED)
    log(f"Pool: {len(pool):,} | Validation A: {len(val_a):,} | Validation B: {len(val_b):,}")

    # Out-of-fold reviewer/product target encodings, as in the original run.
    fold = rng.integers(0, 5, len(pool))
    oof = np.zeros((len(pool), 2), dtype=np.float32)
    columns = ["reviewerID", "asin"]
    for col_i, col in enumerate(columns):
        keys = pool[col].astype("string")
        for fold_id in range(5):
            train_mask, apply_mask = fold != fold_id, fold == fold_id
            oof[apply_mask, col_i] = _encode(
                keys[train_mask], y_pool[train_mask], keys[apply_mask],
                global_mean, SMOOTHING
            )
    target_maps = {
        col: _build_target_map(pool[col], y_pool)
        for col in columns
    }
    meta_a = np.column_stack([
        _encode(pool[col], y_pool, val_a[col], global_mean, SMOOTHING)
        for col in columns
    ]).astype(np.float32)
    meta_b = np.column_stack([
        _encode(pool[col], y_pool, val_b[col], global_mean, SMOOTHING)
        for col in columns
    ]).astype(np.float32)

    n_train = min(int(n_train), len(pool))
    selected = rng.choice(len(pool), n_train, replace=False)
    train = pool.iloc[selected]
    meta_train = oof[selected]
    y_train = train["overall"].to_numpy(dtype=np.int64)
    log(f"Fitting text vectorizers on {n_train:,} reviews...")

    review_train = train["reviewText"].fillna("").map(clean_text)
    summary_train = train["summary"].fillna("").map(clean_text)
    review_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_features=max_features_review,
        sublinear_tf=True, dtype=np.float32
    )
    summary_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_features=max_features_summary,
        sublinear_tf=True, dtype=np.float32
    )
    x_review = review_vectorizer.fit_transform(review_train)
    x_summary = summary_vectorizer.fit_transform(summary_train)
    x_train_text = sp.hstack([x_review, x_summary]).tocsr()

    scaler = StandardScaler().fit(meta_train)

    def add_meta(x_text, meta):
        scaled = scaler.transform(meta).astype(np.float32)
        return sp.hstack([x_text, sp.csr_matrix(scaled)]).tocsr()

    classifier = LogisticRegression(C=2, max_iter=300)
    classifier.fit(add_meta(x_train_text, meta_train), y_train)
    log("Classifier fitted. Scoring validation A/B to store calibration parameters...")

    def transform_text(frame):
        r = frame["reviewText"].fillna("").map(clean_text)
        s = frame["summary"].fillna("").map(clean_text)
        return sp.hstack([
            review_vectorizer.transform(r),
            summary_vectorizer.transform(s)
        ]).tocsr()

    pa = classifier.predict_proba(add_meta(transform_text(val_a), meta_a))
    pb = classifier.predict_proba(add_meta(transform_text(val_b), meta_b))
    ya = val_a["overall"].to_numpy(dtype=np.int64)
    yb = val_b["overall"].to_numpy(dtype=np.int64)
    temperature, bias = _fit_bcts(pa, ya)
    source_prior = _apply_bcts(pa, temperature, bias).mean(axis=0)
    pred_b = CLASSES[pb.argmax(axis=1)]
    metrics = {
        "validation_b_micro_f1": float(f1_score(yb, pred_b, average="micro")),
        "validation_b_macro_f1": float(f1_score(yb, pred_b, average="macro")),
        "validation_b_samples": int(len(yb)),
    }
    log(
        "Validation B (natural mix): "
        f"micro-F1={metrics['validation_b_micro_f1']:.4f}, "
        f"macro-F1={metrics['validation_b_macro_f1']:.4f}"
    )

    artifact = {
        "artifact_type": "crashlens_nlp_classical_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(n_train),
        "n_pool": int(len(pool)),
        "validation_size": int(validation_size),
        "max_features_review": int(max_features_review),
        "max_features_summary": int(max_features_summary),
        "min_df": int(min_df),
        "global_mean": global_mean,
        "smoothing": SMOOTHING,
        "target_maps": target_maps,
        "review_vectorizer": review_vectorizer,
        "summary_vectorizer": summary_vectorizer,
        "scaler": scaler,
        "classifier": classifier,
        "classes": CLASSES.copy(),
        "bcts_temperature": temperature,
        "bcts_bias": bias,
        "source_prior": source_prior,
        "validation_metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    log(f"Saved deployable classical checkpoint: {output_path}")
    return {"path": str(output_path), "metrics": metrics, "n_train": n_train}


def load_classical_artifact(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"NLP model checkpoint not found: {model_path}")
    artifact = joblib.load(model_path)
    if artifact.get("artifact_type") != "crashlens_nlp_classical_v1":
        raise ValueError("This is not a supported CrashLens NLP classical checkpoint.")
    return artifact


def predict_reviews(
    frame: pd.DataFrame,
    model_path: str | Path,
    apply_label_shift: bool = False,
) -> dict:
    """Predict ratings for new reviews using a locally trained classical artifact."""
    if "reviewText" not in frame.columns:
        raise ValueError("CSV must contain a 'reviewText' column.")
    if frame.empty:
        raise ValueError("The uploaded review table is empty.")
    if apply_label_shift and len(frame) < 1000:
        raise ValueError("EM label-shift correction is only exposed for batches of 1,000+ rows.")

    artifact = load_classical_artifact(model_path)
    work = frame.copy()
    review = work["reviewText"].fillna("").map(clean_text)
    if "summary" in work.columns:
        summary = work["summary"].fillna("").map(clean_text)
    else:
        summary = pd.Series([""] * len(work), index=work.index)

    x_text = sp.hstack([
        artifact["review_vectorizer"].transform(review),
        artifact["summary_vectorizer"].transform(summary),
    ]).tocsr()

    meta_cols = []
    for col in ["reviewerID", "asin"]:
        values = work[col] if col in work.columns else pd.Series([pd.NA] * len(work))
        meta_cols.append(_lookup_target_features(
            values,
            artifact["target_maps"][col],
            float(artifact["global_mean"]),
            float(artifact["smoothing"]),
        ))
    meta = np.column_stack(meta_cols).astype(np.float32)
    scaled = artifact["scaler"].transform(meta).astype(np.float32)
    x = sp.hstack([x_text, sp.csr_matrix(scaled)]).tocsr()

    raw = artifact["classifier"].predict_proba(x)
    calibrated = _apply_bcts(
        raw,
        float(artifact["bcts_temperature"]),
        np.asarray(artifact["bcts_bias"], dtype=np.float64),
    )
    result = {
        "raw_probabilities": raw,
        "calibrated_probabilities": calibrated,
        "corrected_probabilities": None,
        "estimated_class_mix": None,
    }
    if apply_label_shift:
        source = np.asarray(artifact["source_prior"], dtype=np.float64)
        estimated = _em_mix(calibrated, source)
        weighted = calibrated * (estimated / np.clip(source, 1e-9, 1.0))
        corrected = weighted / np.clip(weighted.sum(axis=1, keepdims=True), 1e-12, None)
        result["corrected_probabilities"] = corrected
        result["estimated_class_mix"] = estimated
    return result
