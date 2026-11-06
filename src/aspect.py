"""Helpers for Part 2: satisfaction about one aspect (warranty / guarantee).

Idea
----
1. *Spelling variants*: words within a small edit distance of the target found in
   the real vocabulary (``warrenty``, ``gaurantee`` ...). Candidates are reviewed by hand,
   because edit distance also returns false friends (``warcraft``, ``granted``).
2. *Semantic neighbours*: word embeddings trained on the reviews themselves
   (PPMI + truncated SVD, the count-based equivalent of word2vec), so they also
   contain the typos and the jargon of this domain.
3. *Matching*: whole-word regular expression over the review text.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

_TOKEN = re.compile(r"[a-z]{2,}")


def count_vocabulary(texts: pd.Series) -> Counter:
    """Word counts (lower-case alphabetic tokens of length >= 2)."""
    counts: Counter = Counter()
    for s in texts.fillna("").str.lower():
        counts.update(_TOKEN.findall(s))
    return counts


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert / delete / substitute = 1)."""
    if abs(len(a) - len(b)) > 3:
        return 9
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def spelling_candidates(target: str, vocab: Counter, max_dist: int = 3, min_count: int = 3) -> pd.DataFrame:
    """Vocabulary words close to ``target`` (review the list by hand before using it)."""
    rows = [(w, n, edit_distance(target, w)) for w, n in vocab.items()
            if n >= min_count and len(w) >= 5 and abs(len(w) - len(target)) <= 2]
    out = pd.DataFrame(rows, columns=["word", "count", "edit_distance"])
    out = out[out["edit_distance"] <= max_dist]
    return out.sort_values(["edit_distance", "count"], ascending=[True, False]).reset_index(drop=True)


def build_embeddings(texts: pd.Series, vocab_words: list[str], window: int = 5,
                     dim: int = 150, seed: int = 0):
    """PPMI + SVD word vectors. Returns (unit-length matrix E, word -> row index)."""
    w2i = {w: i for i, w in enumerate(vocab_words)}
    size = len(vocab_words)
    rows, cols = [], []
    for s in texts.fillna("").str.lower():
        ids = np.fromiter((w2i.get(w, -1) for w in _TOKEN.findall(s)), dtype=np.int32)
        for k in range(1, window + 1):
            if len(ids) > k:
                a, b = ids[:-k], ids[k:]
                keep = (a >= 0) & (b >= 0)
                rows.append(a[keep]); cols.append(b[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    co = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(size, size)).tocsr()
    co = (co + co.T).tocoo()
    total = co.sum()
    row_sum = np.asarray(co.tocsr().sum(1)).ravel()
    ctx = row_sum ** 0.75
    ctx = ctx / ctx.sum()                      # smoothed context distribution (as in word2vec)
    pmi = np.log((co.data / total) / ((row_sum[co.row] / total) * ctx[co.col]) + 1e-12)
    keep = pmi > 0                             # positive PMI only
    ppmi = sp.csr_matrix((pmi[keep], (co.row[keep], co.col[keep])), shape=(size, size))
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(ppmi)
    return normalize(emb), w2i


def nearest(emb: np.ndarray, vocab_words: list[str], w2i: dict, word: str, k: int = 25) -> pd.DataFrame:
    sims = emb @ emb[w2i[word]]
    order = np.argsort(-sims)[1:k + 1]
    return pd.DataFrame({"word": [vocab_words[i] for i in order], "cosine": sims[order].round(2)})


def contains_any(texts: pd.Series, words) -> pd.Series:
    """True where the review contains at least one of ``words`` as a whole word."""
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    return texts.fillna("").str.lower().str.contains(pattern, regex=True)
