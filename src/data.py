"""Data loading and cleaning helpers.

Design notes
------------
* ``asin`` is always read as *string*: some ASINs are 10-digit numbers with
  leading zeros (ISBN-10 style) which pandas would otherwise silently corrupt.
* ``vote`` is stored as text such as ``"1,489"`` or ``"3.0"``; it is parsed
  into a numeric column where a missing value means "no helpful votes" (0).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config

TRAIN_COLUMNS = [
    "overall", "vote", "verified", "reviewTime", "reviewerID", "asin",
    "style", "reviewerName", "reviewText", "summary", "unixReviewTime",
]


def _read_csv(path, usecols=None, nrows=None) -> pd.DataFrame:
    """Read a CSV keeping identifiers as strings (see module docstring)."""
    return pd.read_csv(
        path,
        usecols=usecols,
        nrows=nrows,
        dtype={"asin": "string", "reviewerID": "string", "vote": "string"},
        low_memory=False,
    )


def parse_votes(votes: pd.Series) -> pd.Series:
    """Convert raw ``vote`` strings (``'1,489'``, ``'3.0'``, NaN) to ints."""
    cleaned = votes.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype("int32")


def load_train(columns=None, nrows=None) -> pd.DataFrame:
    """Load the training set. Pass ``columns`` to save memory (8 GB laptops!)."""
    df = _read_csv(config.TRAIN_CSV, usecols=columns, nrows=nrows)
    if "vote" in df.columns:
        df["vote"] = parse_votes(df["vote"])
    if "overall" in df.columns:
        df["overall"] = df["overall"].astype("int8")
    return df


def load_test(columns=None) -> pd.DataFrame:
    """Load the unlabeled test set (row order matters for the submission file)."""
    df = _read_csv(config.TEST_CSV, usecols=columns)
    if "vote" in df.columns:
        df["vote"] = parse_votes(df["vote"])
    return df


# --------------------------------------------------------------------------- #
# Products (asin -> title / brand)
# --------------------------------------------------------------------------- #
_AMAZON_AUTHOR = re.compile(r"^Visit Amazon's (?P<name>.+?) Page$")
_BRAND_PREFIX = re.compile(r"^Brand:\s*", flags=re.IGNORECASE)
PLACEHOLDER_BRANDS = {"generic", "unknown", "n/a", "na", "none", "unbranded", ""}


def _normalise_brand(brand: str | float) -> str | float:
    if not isinstance(brand, str):
        return np.nan
    brand = html.unescape(brand).strip()
    brand = _BRAND_PREFIX.sub("", brand)
    match = _AMAZON_AUTHOR.match(brand)          # books: "Visit Amazon's <Author> Page"
    return match.group("name").strip() if match else brand


def load_products() -> pd.DataFrame:
    """Return one clean row per ``asin`` with ``title``, ``brand`` and ``brand_key``.

    Cleaning steps (all justified by inspecting ``title_brand.csv``):
      1. drop exact duplicate ASIN rows (≈30k; they never conflict),
      2. un-escape HTML entities such as ``&amp;`` (≈53k titles),
      3. strip Amazon's "Visit Amazon's X Page" wrapper used for authors,
      4. merge case variants (``Sony`` / ``SONY`` / ``sony``) through a
         case-insensitive ``brand_key`` and show the most frequent spelling,
      5. flag placeholder brands (``Generic`` ...) with ``brand_is_placeholder``.
    """
    prod = pd.read_csv(config.PRODUCTS_CSV, dtype="string")
    prod = prod.drop_duplicates("asin").reset_index(drop=True)
    prod["title"] = prod["title"].map(lambda t: html.unescape(t).strip() if isinstance(t, str) else t)
    prod["brand"] = prod["brand"].map(_normalise_brand)
    prod["brand_key"] = prod["brand"].str.lower().str.replace(r"\s+", " ", regex=True)

    # most frequent surface form per key => nice display name
    display = (
        prod.dropna(subset=["brand"]).groupby("brand_key")["brand"]
        .agg(lambda s: s.value_counts().index[0])
    )
    prod["brand"] = prod["brand_key"].map(display).astype("string")
    prod["brand_is_placeholder"] = prod["brand_key"].isin(PLACEHOLDER_BRANDS)
    return prod


if __name__ == "__main__":
    sample = load_train(nrows=5)
    print(f"Training CSV: {config.TRAIN_CSV}")
    print(f"Loaded sample: {len(sample)} rows x {len(sample.columns)} columns")
