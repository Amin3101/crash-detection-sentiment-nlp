"""Copy the example call at bottom into an EVALUATED notebook cell.

DO NOT run this file as-is: it intentionally exports no fabricated rows.
Only export arrays after evaluating the correct checkpoint and loader.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def export_predictions(video_ids, test_targets, test_probs, split="test", output="real_predictions.csv"):
    """Persist real test output with explicit index-alignment validation."""
    ids = list(map(str, video_ids))
    labels = np.asarray(test_targets)
    probs = np.asarray(test_probs)
    assert len(ids) == len(labels) == len(probs), "Mismatched ID/label/probability lengths"
    assert len(ids) == len(set(ids)), "Duplicate video IDs; use unique clip IDs if sampling multiple clips"
    assert set(np.unique(labels)).issubset({0, 1}), "Labels should be binary"
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(), "Probabilities invalid"
    frame = pd.DataFrame({
        "video_id": ids,
        "label": labels.astype(int),
        "probability": probs.astype(float),
        "split": split,
        "video_file": "",
    })
    frame.to_csv(output, index=False)
    return Path(output)


# In final.ipynb AFTER the original evaluation:
# assert test_loader.shuffle is False  # This attribute is not defined on all DataLoaders!
# The notebook creates test_loader with shuffle=False, so test_df order is expected
# to match collected outputs. Verify IDs against the underlying test dataset.
# export_predictions(
#     video_ids=test_dataset.data["video_id"].tolist(),
#     test_targets=test_targets,
#     test_probs=test_probs,
#     split="test",
#     output="full_video_test_predictions.csv",
# )
