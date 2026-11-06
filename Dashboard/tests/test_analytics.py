import pandas as pd
import pytest
from analytics import PredictionValidationError, evaluate, safe_video_path, validate_predictions


def test_valid_metrics():
    df = validate_predictions(pd.DataFrame({
        "video_id": ["a", "b", "c", "d"],
        "label": [0, 0, 1, 1],
        "probability": [0.1, 0.7, 0.6, 0.9],
        "split": ["test"] * 4,
    }))
    m = evaluate(df, 0.5)
    assert m["cm"].tolist() == [[1, 1], [0, 2]]
    assert m["accuracy"] == .75
    assert m["roc_auc"] == 0.75


@pytest.mark.parametrize("bad", [
    {"video_id": ["a", "a"], "label": [0, 1], "probability": [.1, .9]},
    {"video_id": ["a"], "label": [2], "probability": [.1]},
    {"video_id": ["a"], "label": [1], "probability": [1.2]},
    {"video_id": ["a"], "label": [1], "probability": [float("nan")]},
])
def test_invalid_predictions(bad):
    with pytest.raises(PredictionValidationError):
        validate_predictions(pd.DataFrame(bad))


def test_video_path_rejects_traversal():
    assert safe_video_path("../../README.md") is None
    assert safe_video_path("missing.mp4") is None
