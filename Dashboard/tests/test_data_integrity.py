from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def test_source_tables_are_consistent():
    runs = pd.read_csv(ROOT / "data" / "runs.csv")
    assert runs.run_id.is_unique
    assert set(runs.task) == {"Anticipation", "Detection"}
    assert runs.set_index("run_id").loc["full_video_jpg", "samples"] == 225
    assert runs.set_index("run_id").loc["clean_1", "samples"] == 170
    assert (runs[["macro_f1", "roc_auc"]].notna().all(axis=1)).all()  # Accuracy not documented for every run
    assert runs[["macro_f1", "roc_auc", "accuracy"]].apply(lambda c: c.dropna().between(0, 1).all()).all()


def test_full_video_confusion_matches_notebook():
    y_true = [0] * 97 + [0] * 16 + [1] * 9 + [1] * 103
    y_pred = [0] * 97 + [1] * 16 + [0] * 9 + [1] * 103
    assert len(y_true) == 225
    assert round(accuracy_score(y_true, y_pred), 4) == .8889
    assert round(f1_score(y_true, y_pred, average="macro"), 4) == .8888


def test_curated_epochs():
    train = pd.read_csv(ROOT / "data" / "training.csv")
    assert len(train.query("run == 'Original temporal sample'")) == 9
    assert len(train.query("run == 'Clean pre-event MViT · run 1'")) == 7
