"""Part 3 final model (v3, full GPU): SBERT sentence embeddings + TF-IDF + reviewer/product
history -> a small PyTorch MLP trained on GPU, then EM label-shift correction with
bias-corrected temperature scaling (Alexandari et al. 2020).

Changes vs. v2 (SBERT embeddings + sklearn MLPClassifier):
  - The classifier itself (not just the embedding step) now runs on GPU: a 2-layer
    PyTorch MLP trained with Adam + weighted cross-entropy (class weights handle the
    imbalance directly, which is cleaner than the oversampling trick used in v2 and
    avoids literally duplicating rows in memory).
  - TF-IDF is still built with sklearn (TF-IDF itself is a sparse linear-algebra op,
    not something GPUs help with) but is converted to a dense torch tensor in batches
    for the classifier.

Run from the project root:  python final_model_gpu.py [N_TRAIN]   (default 200000)
Requires: torch, sentence-transformers, scikit-learn, scipy, pandas, numpy.
Falls back to CPU automatically if no GPU is available (just slower, same results).
Writes outputs/submission_2q.csv, submission_2q_uncorrected.csv, report_final.txt"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from src import config, text

N_TRAIN, SEED = (int(sys.argv[1]) if len(sys.argv) > 1 else 200_000), config.SEED
rng = np.random.default_rng(SEED); torch.manual_seed(SEED); t0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = []
def log(m):
    print(m, flush=True); LOG.append(m)

data_df = pd.read_csv(config.TRAIN_CSV, usecols=["overall","reviewText","summary","reviewerID","asin"], dtype={"asin": str})
pool, val = train_test_split(data_df, test_size=40_000, random_state=SEED, stratify=data_df["overall"]); del data_df
val_a, val_b = train_test_split(val, test_size=0.5, random_state=SEED, stratify=val["overall"])
test = pd.read_csv(config.TEST_CSV, usecols=["reviewText","summary","reviewerID","asin"], dtype={"asin": str})
GLOBAL = pool["overall"].mean(); M = 3.0
log(f"loaded {time.time()-t0:.0f}s | pool {len(pool):,} | device={DEVICE}")

# ---------------------------------------------------------------------------
# Target encoding for reviewer/product history (unchanged from v1/v2)
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
# TF-IDF features (reduced vocab -- SBERT below carries most semantic signal)
# ---------------------------------------------------------------------------
vect = lambda n: TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=n, sublinear_tf=True, dtype=np.float32)
tr_r, tr_s = prep(train); v_r, v_s = vect(150_000), vect(50_000)
Xtr_tfidf = sp.hstack([v_r.fit_transform(tr_r), v_s.fit_transform(tr_s)]).tocsr()
def tf(d): r, s = prep(d); return sp.hstack([v_r.transform(r), v_s.transform(s)]).tocsr()
Xa_tfidf, Xb_tfidf, Xt_tfidf = tf(val_a), tf(val_b), tf(test)
log(f"tfidf features {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# SBERT sentence embeddings, encoded on GPU in batches (384-dim, normalized).
# ---------------------------------------------------------------------------
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
def embed(d, batch_size=256):
    texts = (d["reviewText"].fillna("") + " " + d["summary"].fillna("")).tolist()
    return sbert.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
Etr = embed(train); Ea, Eb, Et = embed(val_a), embed(val_b), embed(test)
log(f"sbert embeddings done {time.time()-t0:.0f}s")

# To keep the TF-IDF part from dominating GPU memory, project it down with a
# fixed random sparse projection (cheap, no fitting needed) before concatenating
# with the dense SBERT + history features. This keeps the final feature vector
# small enough to comfortably fit as a dense GPU tensor for the whole dataset.
TFIDF_PROJ_DIM = 512
proj_dim_in = Xtr_tfidf.shape[1]
proj_matrix = sp.random(proj_dim_in, TFIDF_PROJ_DIM, density=0.01, random_state=SEED,
                         data_rvs=lambda n: rng.normal(0, 1, n).astype(np.float32)).tocsr()
project = lambda X: np.asarray((X @ proj_matrix).todense(), dtype=np.float32)
Ptr, Pa_, Pb_, Pt_ = project(Xtr_tfidf), project(Xa_tfidf), project(Xb_tfidf), project(Xt_tfidf)
log(f"tfidf projected to {TFIDF_PROJ_DIM}d {time.time()-t0:.0f}s")

sc = StandardScaler().fit(meta_tr)
def build(P, E, m):
    return np.concatenate([P, E, sc.transform(m).astype(np.float32)], axis=1)
Xtr_feat = build(Ptr, Etr, meta_tr)
Xa_feat, Xb_feat, Xt_feat = build(Pa_, Ea, meta_a), build(Pb_, Eb, meta_b), build(Pt_, Et, meta_t)
log(f"all features assembled {time.time()-t0:.0f}s | dim={Xtr_feat.shape[1]}")
ya, yb = val_a["overall"].to_numpy(), val_b["overall"].to_numpy()
classes = np.arange(1, 6)

# ---------------------------------------------------------------------------
# PyTorch MLP classifier, trained on GPU with class-weighted cross-entropy
# (handles the 1/2/4-star minority classes directly, no row duplication).
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 64), n_classes=5, p_drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def to_loader(X, y=None, batch_size=512, shuffle=False):
    Xt = torch.from_numpy(X)
    if y is None:
        return DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    yt = torch.from_numpy((y - 1).astype(np.int64))
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

class_counts = np.bincount(y_train, minlength=6)[1:]
class_weights = torch.tensor((class_counts.sum() / (5 * class_counts)).astype(np.float32), device=DEVICE)

model = MLP(Xtr_feat.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
train_loader = to_loader(Xtr_feat, y_train, batch_size=512, shuffle=True)
val_loader_a = to_loader(Xa_feat, batch_size=1024)

best_state, best_macro, patience, bad_epochs = None, -1.0, 5, 0
MAX_EPOCHS = 40
for epoch in range(MAX_EPOCHS):
    model.train()
    for xb, yb_ in train_loader:
        xb, yb_ = xb.to(DEVICE), yb_.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb_)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = []
        for (xb,) in val_loader_a:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy() + 1)
        preds = np.concatenate(preds)
    macro = f1_score(ya, preds, average="macro")
    log(f"epoch {epoch+1:02d}  val_a macro F1 {macro:.4f}  ({time.time()-t0:.0f}s)")
    if macro > best_macro:
        best_macro, best_state, bad_epochs = macro, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            log(f"early stopping at epoch {epoch+1} (best val_a macro F1 {best_macro:.4f})")
            break
model.load_state_dict(best_state)
log(f"classifier trained {time.time()-t0:.0f}s")

@torch.no_grad()
def predict_proba(X):
    model.eval()
    out = []
    for (xb,) in to_loader(X, batch_size=1024):
        out.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out)

Pa, Pb, Pt = predict_proba(Xa_feat), predict_proba(Xb_feat), predict_proba(Xt_feat)

# ---------------------------------------------------------------------------
# Label-shift correction (EM + BCTS, and BBSE) -- unchanged from v1/v2
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

log(f"\n===== tfidf-proj + sbert + history (torch MLP, {DEVICE}) ({time.time()-t0:.0f}s) =====")
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
