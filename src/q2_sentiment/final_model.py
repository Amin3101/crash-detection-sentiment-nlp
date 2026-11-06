"""Part 3 final model (v2): sentence-transformer embeddings (GPU) + TF-IDF + reviewer/product
history -> MLP classifier, then EM label-shift correction with bias-corrected temperature
scaling (Alexandari et al. 2020).

Changes vs. v1 (TF-IDF + LogisticRegression only):
  1. Added SBERT sentence embeddings (all-MiniLM-L6-v2, encoded on GPU) as extra features.
     This is the "pretrained embedding to catch synonyms/typos" idea from section 2 of the
     project text, reused here as a general semantic feature for sentiment too.
  2. Replaced LogisticRegression with a small MLP (MLPClassifier) + class_weight-style
     oversampling, which helps macro-F1 on the rare classes (2 and 4 stars).
  3. Kept the EM+BCTS label-shift correction from v1 unchanged -- it already worked well
     (see report_final.txt) and is not related to the embedding change.

Run from the project root:  python final_model.py [N_TRAIN]   (default 200000; use 300000 if
you have 8 GB+ free). Needs a GPU for the embedding step to be fast; falls back to CPU
automatically (just slower) if torch.cuda is not available.
Writes outputs/submission_2q.csv, submission_2q_uncorrected.csv, report_final.txt"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from src import config, text

N_TRAIN, SEED = (int(sys.argv[1]) if len(sys.argv) > 1 else 200_000), config.SEED
rng = np.random.default_rng(SEED); t0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG = []
def log(m):
    print(m, flush=True); LOG.append(m)

data = pd.read_csv(config.TRAIN_CSV, usecols=["overall","reviewText","summary","reviewerID","asin"], dtype={"asin": str})
pool, val = train_test_split(data, test_size=40_000, random_state=SEED, stratify=data["overall"]); del data
val_a, val_b = train_test_split(val, test_size=0.5, random_state=SEED, stratify=val["overall"])
test = pd.read_csv(config.TEST_CSV, usecols=["reviewText","summary","reviewerID","asin"], dtype={"asin": str})
GLOBAL = pool["overall"].mean(); M = 3.0
log(f"loaded {time.time()-t0:.0f}s | pool {len(pool):,} | device={DEVICE}")

# ---------------------------------------------------------------------------
# Target encoding for reviewer/product history (unchanged from v1)
# ---------------------------------------------------------------------------
def encode(keys_pool, y_pool, keys_apply):
    g = pd.DataFrame({"k": keys_pool, "y": y_pool}).groupby("k")["y"].agg(["sum", "count"])
    s = g["sum"].reindex(keys_apply).fillna(0).to_numpy(); n = g["count"].reindex(keys_apply).fillna(0).to_numpy()
    return (s + M * GLOBAL) / (n + M)

fold = rng.integers(0, 5, len(pool)); y_pool = pool["overall"].to_numpy()
oof = np.zeros((len(pool), 2))
for col_i, col in enumerate(["reviewerID", "asin"]):
    keys = pool[col].to_numpy()
    for f in range(5):
        tr_m, ap_m = fold != f, fold == f
        oof[ap_m, col_i] = encode(keys[tr_m], y_pool[tr_m], keys[ap_m])
def meta(df):
    return np.c_[[encode(pool[c].to_numpy(), y_pool, df[c].to_numpy()) for c in ["reviewerID", "asin"]]].T
meta_a, meta_b, meta_t = meta(val_a), meta(val_b), meta(test)
log(f"target encoding done {time.time()-t0:.0f}s")

pos = rng.choice(len(pool), N_TRAIN, replace=False); train = pool.iloc[pos]; meta_tr = oof[pos]; y_train = train["overall"].to_numpy()
prior_train = np.bincount(y_train, minlength=6)[1:] / len(y_train)
prep = lambda d: (d["reviewText"].fillna("").map(text.clean_text), d["summary"].fillna("").map(text.clean_text))

# ---------------------------------------------------------------------------
# TF-IDF features (unchanged from v1, just smaller vocab since SBERT now
# carries most of the semantic signal -- keeps this from being redundant)
# ---------------------------------------------------------------------------
vect = lambda n: TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=n, sublinear_tf=True, dtype=np.float32)
tr_r, tr_s = prep(train); v_r, v_s = vect(150_000), vect(50_000)
Xtr_tfidf = sp.hstack([v_r.fit_transform(tr_r), v_s.fit_transform(tr_s)]).tocsr()
def tf(d): r, s = prep(d); return sp.hstack([v_r.transform(r), v_s.transform(s)]).tocsr()
Xa_tfidf, Xb_tfidf, Xt_tfidf = tf(val_a), tf(val_b), tf(test)
log(f"tfidf features {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# NEW: sentence-transformer embeddings, encoded on GPU in batches.
# all-MiniLM-L6-v2 is small (384-dim) and fast but carries real semantic
# meaning (synonyms, negation, typos handled better than TF-IDF n-grams).
# ---------------------------------------------------------------------------
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
def embed(d, batch_size=256):
    texts = (d["reviewText"].fillna("") + " " + d["summary"].fillna("")).tolist()
    return sbert.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
Etr = embed(train); Ea, Eb, Et = embed(val_a), embed(val_b), embed(test)
log(f"sbert embeddings done {time.time()-t0:.0f}s")

sc = StandardScaler().fit(meta_tr)
def build(Xtfidf, E, m):
    return sp.hstack([Xtfidf, sp.csr_matrix(E), sp.csr_matrix(sc.transform(m).astype(np.float32))]).tocsr()
log(f"all features assembled {time.time()-t0:.0f}s")
ya, yb = val_a["overall"].to_numpy(), val_b["overall"].to_numpy()
classes = np.arange(1, 6)

# ---------------------------------------------------------------------------
# Label-shift correction (EM + BCTS, and BBSE) -- unchanged from v1
# ---------------------------------------------------------------------------
def bcts_fit(P, y):
    from scipy.optimize import minimize; from scipy.special import log_softmax
    Z = np.log(np.clip(P, 1e-9, 1))
    def nll(p):
        L = log_softmax(Z / np.exp(p[0]) + p[1:], axis=1); return -L[np.arange(len(y)), y - 1].mean()
    r = minimize(nll, np.zeros(6), method="L-BFGS-B"); T, b = np.exp(r.x[0]), r.x[1:]
    from scipy.special import softmax
    return lambda Q: softmax(np.log(np.clip(Q, 1e-9, 1)) / T + b, axis=1)
def em(Pc, src, iters=300):
    q = src.copy()
    for _ in range(iters):
        w = Pc * (q / src); w /= w.sum(1, keepdims=True); q = w.mean(0)
    return q
def bbse_fit(P, y):
    from scipy.optimize import nnls
    conf = confusion_matrix(y, classes[P.argmax(1)], labels=classes, normalize="true")
    def est(Q):
        mu = np.bincount(Q.argmax(1), minlength=5) / len(Q); q, _ = nnls(conf.T, mu); return q / q.sum()
    return est
f1s = lambda y, p: (f1_score(y, p, average="micro"), f1_score(y, p, average="macro"))
fmt = lambda v: " ".join(f"{x:5.3f}" for x in v)

# ---------------------------------------------------------------------------
# Classifier: MLP instead of LogisticRegression. class_weight is emulated by
# resampling the minority classes (2, 4) up to the size of the majority
# class, which is the main lever for raising macro-F1.
# ---------------------------------------------------------------------------
def balance_oversample(X, y, rng):
    counts = np.bincount(y, minlength=6)[1:]; target = counts.max()
    idx_parts = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        reps = rng.choice(idx_c, target, replace=True)
        idx_parts.append(reps)
    idx = np.concatenate(idx_parts); rng.shuffle(idx)
    return X[idx], y[idx]

results = {}
for name, A, B, T_ in [("tfidf + sbert + history", build(Xa_tfidf, Ea, meta_a), build(Xb_tfidf, Eb, meta_b), build(Xt_tfidf, Et, meta_t))]:
    Xfit = build(Xtr_tfidf, Etr, meta_tr)
    Xfit_bal, y_bal = balance_oversample(Xfit, y_train, rng)
    clf = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-4, max_iter=60,
                         early_stopping=True, random_state=SEED).fit(Xfit_bal, y_bal)
    Pa, Pb, Pt = clf.predict_proba(A), clf.predict_proba(B), clf.predict_proba(T_)
    results[name] = (Pa, Pb, Pt)
    log(f"\n===== {name} ({time.time()-t0:.0f}s) =====")
    mi, ma = f1s(np.r_[ya, yb], np.r_[classes[Pa.argmax(1)], classes[Pb.argmax(1)]])
    log(f"validation at training mix: micro {mi:.4f} macro {ma:.4f}")
    cal = bcts_fit(Pa, ya); Pac, Ptc = cal(Pa), cal(Pt); src = Pac.mean(0)
    bb = bbse_fit(Pa, ya)
    q_em, q_bb = em(Ptc, src), bb(Pt)
    log("real test mix  EM-BCTS: " + fmt(q_em) + " | BBSE: " + fmt(q_bb))
    for mname, mix in [("uniform", np.full(5, 0.2)), ("EM estimate", q_em), ("BBSE estimate", q_bb)]:
        w = (mix / prior_train)[yb - 1]; idx = rng.choice(len(yb), len(yb), replace=True, p=w / w.sum())
        ys, Ps = yb[idx], Pb[idx]; Pcs = cal(Ps)
        e_bb, e_em = bb(Ps), em(Pcs, src)
        true = np.bincount(ys, minlength=6)[1:] / len(ys)
        log(f"simulate mix={mname:13s} | mix error L1: BBSE {np.abs(e_bb-true).sum():.3f}, EM {np.abs(e_em-true).sum():.3f}")
        for lab, pred in [("uncorrected", classes[Ps.argmax(1)]),
                          ("BBSE-corrected", classes[(Ps * e_bb / prior_train).argmax(1)]),
                          ("EM+BCTS-corrected", classes[(Pcs * e_em / src).argmax(1)]),
                          ("oracle mix (cal.)", classes[(Pcs * true / src).argmax(1)])]:
            mi, ma = f1s(ys, pred); log(f"    {lab:20s} micro {mi:.4f} macro {ma:.4f}")
config.OUTPUT_DIR.mkdir(exist_ok=True)
pd.DataFrame({"predicted": classes[Ptc.argmax(1)]}).to_csv(config.OUTPUT_DIR / "submission_2q_uncorrected.csv", index=False)
pd.DataFrame({"predicted": classes[(Ptc * q_em / src).argmax(1)]}).to_csv(config.OUTPUT_DIR / "submission_2q.csv", index=False)
log("final submission mix: " + fmt(np.bincount(classes[(Ptc * q_em / src).argmax(1)], minlength=6)[1:] / len(Ptc)))
log(f"\ndone {time.time()-t0:.0f}s")
open(config.OUTPUT_DIR / "report_final.txt", "w").write("\n".join(LOG))
