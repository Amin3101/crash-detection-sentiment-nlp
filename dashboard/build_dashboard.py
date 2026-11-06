"""Builds dashboard/dashboard.html (one self-contained, interactive file) from the raw data and results/.

    python dashboard/build_dashboard.py          # run from the project root, data in data/
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score

import ensemble as ens
from src import config, data

RES = ROOT / "results"
train = data.load_train(columns=["overall", "vote", "reviewerID", "reviewerName", "asin", "reviewText", "unixReviewTime"])
products = data.load_products()
D = {}

# ---- meta, distribution, trend (Q1) ----
counts = train["overall"].value_counts().sort_index()
D["meta"] = {"n_reviews": int(len(train)), "n_products": int(train["asin"].nunique()), "mean_rating": float(train["overall"].mean())}
D["dist"] = {"share": (counts / len(train)).round(5).tolist()}
year = pd.to_datetime(train["unixReviewTime"], unit="s").dt.year
g = train.groupby(year)["overall"]
D["year"] = {"years": [str(y) for y in g.size().index], "n": g.size().tolist(),
             "five": (g.apply(lambda s: (s == 5).mean()) * 100).round(2).tolist(), "one": (g.apply(lambda s: (s == 1).mean()) * 100).round(2).tolist()}

# ---- words (Q2) ----
EXTRA = {"one", "would", "get", "got", "also", "use", "used", "using", "will", "can", "could", "just", "like", "product", "quot", "amp",
         "don", "didn", "doesn", "isn", "wasn", "ve", "ll", "re", "im", "br"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA)
sentiment = np.select([train["overall"] >= 4, train["overall"] == 3], ["positive", "neutral"], default="negative")
D["words"] = {}
for label in ["positive", "neutral", "negative"]:
    texts = train.loc[sentiment == label, "reviewText"].dropna().sample(60_000, random_state=config.SEED)
    vec = CountVectorizer(stop_words=STOP, token_pattern=r"\b[a-z]{2,}\b", min_df=5, max_features=2000)
    X = vec.fit_transform(texts); totals = np.asarray(X.sum(axis=0)).ravel()
    order = np.argsort(-totals)[:30]
    D["words"][label] = [[str(vec.get_feature_names_out()[i]), int(totals[i])] for i in order]

# ---- reviewers and votes (Q3) ----
by_rev = train.groupby("reviewerID").agg(name=("reviewerName", "first"), votes=("vote", "sum"), n=("overall", "size")).sort_values("votes", ascending=False)
D["votes"] = {"top": [{"rank": i + 1, "name": str(r.name_), "votes": int(r.votes), "n": int(r.n)} for i, r in enumerate(by_rev.head(10).rename(columns={"name": "name_"}).itertuples())]}
v = np.sort(train["vote"].to_numpy())[::-1]; cum = np.cumsum(v) / v.sum()
D["votes"]["curve"] = [[p, float(cum[max(int(len(v) * p / 100) - 1, 0)])] for p in [0.1, 0.5, 1, 2, 5, 10, 20, 50]]
D["votes"]["share_with_vote"] = float((v > 0).mean())

# ---- text length (Q4) ----
L = train["reviewText"].fillna("").str.len().to_numpy()
edges = list(range(300, 3500, 100))
share = [float(((L >= e - 100) & (L < e)).mean()) for e in edges]
D["length"] = {"edges": edges, "share": share, "cum": [float((L < e).mean()) for e in edges], "median": float(np.median(L)),
               "p99": float(np.quantile(L, 0.99)), "cum_at": {str(c): float((L <= c).mean()) for c in (500, 1000, 2000)}}

# ---- products and brands (Q5, Q6) ----
five = train.loc[train["overall"] == 5].groupby("asin").size().nlargest(10)
tot = train[train["asin"].isin(five.index)].groupby("asin")["overall"].agg(total="size", mean="mean")
prod = five.rename("five").to_frame().join(tot).reset_index().merge(products[["asin", "brand", "title"]], on="asin", how="left")
D["products"] = [{"brand": str(r.brand), "title": str(r.title)[:70], "five": int(r.five), "total": int(r.total), "share": float(r.five / r.total), "mean": float(r.mean)} for r in prod.itertuples()]
merged = train[["asin", "overall"]].merge(products[["asin", "brand"]], on="asin", how="left").dropna(subset=["brand"])
bs = merged.groupby("brand")["overall"].agg(n="size", mean="mean", std="std").nlargest(20, "n")
bs["ci"] = 1.96 * bs["std"] / np.sqrt(bs["n"])
D["brands"] = [{"brand": str(i), "n": int(r["n"]), "mean": float(r["mean"]), "ci": float(r["ci"])} for i, r in bs.iterrows()]
D["brand_axis"] = [float(np.floor((bs["mean"] - bs["ci"]).min() * 10) / 10), float(np.ceil((bs["mean"] + bs["ci"]).max() * 10) / 10)]

# ---- warranty (part 2) ----
w = pd.read_csv(RES / "warranty_satisfaction_by_product.csv")
rows = w[w["n_warranty_reviews"] >= 3]
D["warranty"] = {"n_reviews": int(w["n_warranty_reviews"].sum()), "mean": float((w["n_warranty_reviews"] * w["warranty_mean_overall"]).sum() / w["n_warranty_reviews"].sum()),
                 "overall_mean": float(train["overall"].mean()), "n_products_any": int(len(w)), "min_available": 3,
                 "rows": [{"brand": str(r.brand), "title": str(r.title)[:60], "n": int(r.n_warranty_reviews), "mean": float(r.warranty_mean_overall), "pm": float(r.product_mean_overall)} for r in rows.itertuples()]}

# ---- model (part 3) ----
names = ["classical", "transformer"]
P = {n: np.load(RES / "model_probabilities" / f"{n}_probs.npz") for n in names}
ya, yb = P["classical"]["ya"], P["classical"]["yb"]
Pa, Pb, Pt = [P[n]["Pa"] for n in names], [P[n]["Pb"] for n in names], [P[n]["Pt"] for n in names]
wts = ens.fit_weights(Pa, ya)
Ea, Eb, Et = ens.combine(Pa, wts), ens.combine(Pb, wts), ens.combine(Pt, wts)
rows_m = [ens.evaluate(n, Pa[i], Pb[i], Pt[i], ya, yb)[0] for i, n in enumerate(names)] + [ens.evaluate("ENSEMBLE", Ea, Eb, Et, ya, yb)[0]]
cal = ens.fit_bcts(Ea, ya); src = cal(Ea).mean(0); q_em = ens.em_mix(cal(Et), src)
conf = confusion_matrix(ya, Ea.argmax(1) + 1, labels=range(1, 6), normalize="true")
q_bbse, _ = nnls(conf.T, np.bincount(Et.argmax(1), minlength=5) / len(Et)); q_bbse /= q_bbse.sum()
truth = np.r_[ya, yb]
sets = {"classical": (Pa[0], Pb[0]), "transformer": (Pa[1], Pb[1]), "ensemble": (Ea, Eb)}
pred = {n: np.r_[a.argmax(1), b.argmax(1)] + 1 for n, (a, b) in sets.items()}
D["model"] = {
    "results": [{k: (float(x) if k != "model" else x) for k, x in r.items()} for r in rows_m],
    "weights": [[n, float(x)] for n, x in zip(names, wts)],
    "mix": {"train": (np.bincount(ya, minlength=6)[1:] / len(ya)).tolist(), "em": q_em.tolist(), "bbse": q_bbse.tolist()},
    "confusion": [{"model": n, "matrix": confusion_matrix(truth, pred[n], normalize="true").round(4).tolist()} for n in sets],
    "f1": {n: f1_score(truth, pred[n], average=None).round(4).tolist() for n in sets},
}

html = (ROOT / "dashboard" / "template.html").read_text().replace("/*DATA*/", json.dumps(D, separators=(",", ":"), ensure_ascii=False))
out = ROOT / "dashboard" / "dashboard.html"; out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
