"""Project-wide configuration (paths, constants, random seed)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Put the CSV files in <project>/data (or point DATA_DIR to another folder).
_DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
_WORKSPACE_DATA_DIRS = PROJECT_ROOT.parents
if "DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["DATA_DIR"])
elif (_DEFAULT_DATA_DIR / "train_data.csv").exists():
    DATA_DIR = _DEFAULT_DATA_DIR
elif (workspace_data_dir := next(
    (parent for parent in _WORKSPACE_DATA_DIRS if (parent / "train_data.csv").exists()),
    None,
)) is not None:
    DATA_DIR = workspace_data_dir
else:
    DATA_DIR = _DEFAULT_DATA_DIR
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"

TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test_data.csv"
PRODUCTS_CSV = DATA_DIR / "title_brand.csv"

SEED = 42
RATING_LABELS = [1, 2, 3, 4, 5]

for _d in (OUTPUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
