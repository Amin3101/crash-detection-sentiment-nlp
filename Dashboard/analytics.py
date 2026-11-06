"""Reusable, offline-testable analytics for CrashLens.

No model predictions are synthesized. A score CSV must come from an actual
checkpoint evaluation with input order aligned to the corresponding video IDs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

REQUIRED = {"video_id", "label", "probability"}
VIDEO_ROOT = Path(__file__).resolve().parent / "assets" / "videos"


class PredictionValidationError(ValueError):
    pass


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate real per-video scores; reject missing, malformed or duplicate IDs."""
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise PredictionValidationError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise PredictionValidationError("The CSV has no prediction rows.")
    df = frame.copy()
    df["video_id"] = df["video_id"].astype("string").str.strip()
    if df["video_id"].isna().any() or (df["video_id"].str.len() == 0).any():
        raise PredictionValidationError("video_id may not be empty.")
    if df["video_id"].duplicated().any():
        raise PredictionValidationError("video_id must be unique. For multiple clips per video, use a distinct clip ID as video_id.")
    for column in ["label", "probability"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise PredictionValidationError(f"{column} must contain only finite numbers.")
    if not df["label"].isin([0, 1]).all():
        raise PredictionValidationError("label must be 0 or 1.")
    if not df["probability"].between(0, 1).all():
        raise PredictionValidationError("probability must be in [0, 1].")
    df["label"] = df["label"].astype(int)
    if "split" in df:
        df["split"] = df["split"].astype("string").str.strip().str.lower()
        if not df["split"].isin(["train", "val", "validation", "test"]).all():
            raise PredictionValidationError("split must be train, val, validation or test.")
    if "video_file" not in df:
        df["video_file"] = ""
    return df.reset_index(drop=True)


def evaluate(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    y = df["label"].to_numpy(dtype=int)
    prob = df["probability"].to_numpy(dtype=float)
    pred = (prob >= threshold).astype(int)
    result = {
        "samples": len(df),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "cm": confusion_matrix(y, pred, labels=[0, 1]),
        "roc_auc": None,
        "average_precision": None,
        "predictions": pred,
    }
    if len(set(y.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(y, prob))
        result["average_precision"] = float(average_precision_score(y, prob))
    return result


def safe_video_path(relative_name: str | None) -> Path | None:
    """Serve only locally supplied assets/videos files, preventing path traversal."""
    if relative_name is None or not str(relative_name).strip():
        return None
    root = VIDEO_ROOT.resolve()
    candidate = (root / str(relative_name)).resolve()
    if candidate != root and root in candidate.parents and candidate.is_file():
        if candidate.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
            return candidate
    return None
