"""Combine several models, calibrate, correct for the label shift of the test set, write the submission.

    python ensemble.py
    python ensemble.py classical_probs.npz transformer_probs.npz --out submission_2q.csv

Every .npz holds Pa, Pb (probabilities on validation A / B), Pt (test) and ya, yb (true labels).
Weights and calibration are fitted on validation A only; validation B is the honest check.
Needs only numpy, pandas, scipy and scikit-learn.
"""
import argparse
from pathlib import Path

import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score

CLASSES = np.arange(1, 6)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG = lambda P: np.log(np.clip(P, 1e-9, 1))

def nll(logp, y):
    return -logp[np.arange(len(y)), y - 1].mean()

def fit_weights(Ps, y):
    """Weights of a log-linear combination, fitted by minimising the log-loss on validation A."""
    def loss(w):
        return nll(log_softmax(sum(np.exp(wi) * LOG(P) for wi, P in zip(w, Ps)), axis=1), y)
    r = minimize(loss, np.log(np.full(len(Ps), 1.0 / len(Ps))), method="Nelder-Mead")
    return np.exp(r.x)

def combine(Ps, w):
    return softmax(sum(wi * LOG(P) for wi, P in zip(w, Ps)), axis=1)

def fit_bcts(P, y):
    """Bias-corrected temperature scaling (Alexandari et al. 2020)."""
    Z = LOG(P)
    r = minimize(lambda p: nll(log_softmax(Z / np.exp(p[0]) + p[1:], axis=1), y), np.zeros(6), method="L-BFGS-B")
    T, b = np.exp(r.x[0]), r.x[1:]
    return lambda Q: softmax(LOG(Q) / T + b, axis=1)

def em_mix(Pc, src, iters=300):
    q = src.copy()
    for _ in range(iters):
        w = Pc * (q / src); w /= w.sum(1, keepdims=True); q = w.mean(0)
    return q

def decide(Pc, q, src):
    return CLASSES[(Pc * (q / src)).argmax(1)]

def evaluate(name, Pa, Pb, Pt, ya, yb):
    cal = fit_bcts(Pa, ya); src = cal(Pa).mean(0); q_test = em_mix(cal(Pt), src)
    rows = {"model": name, "log-loss (val B)": nll(LOG(cal(Pb)), yb)}
    for mname, mix in [("uniform", np.full(5, 0.2)), ("test estimate", q_test)]:
        micro, macro = [], []
        for seed in range(5):
            rg = np.random.default_rng(seed)
            w = (mix / (np.bincount(yb, minlength=6)[1:] / len(yb)))[yb - 1]
            idx = rg.choice(len(yb), len(yb), replace=True, p=w / w.sum())
            Pcs = cal(Pb[idx]); pred = decide(Pcs, em_mix(Pcs, src), src)
            micro.append(f1_score(yb[idx], pred, average="micro")); macro.append(f1_score(yb[idx], pred, average="macro"))
        rows[f"micro ({mname} mix)"] = np.mean(micro); rows[f"macro ({mname} mix)"] = np.mean(macro)
    return rows, cal, src, q_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        help="model probability files (defaults to the saved classical and transformer predictions)",
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "submission_2q.csv"),
    )
    args = ap.parse_args()
    files = args.files or [
        str(PROJECT_ROOT / "results" / "model_probabilities" / "classical_probs.npz"),
        str(PROJECT_ROOT / "results" / "model_probabilities" / "transformer_probs.npz"),
    ]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        ap.error(
            "model probability file(s) not found: "
            + ", ".join(missing)
            + ". Provide existing .npz files as arguments."
        )
    data = [np.load(f) for f in files]
    ya, yb = data[0]["ya"], data[0]["yb"]
    assert all((d["ya"] == ya).all() and (d["yb"] == yb).all() for d in data), "models were evaluated on different validation sets"
    Pa_l, Pb_l, Pt_l = [d["Pa"] for d in data], [d["Pb"] for d in data], [d["Pt"] for d in data]

    w = fit_weights(Pa_l, ya)
    print("ensemble weights (fitted on validation A):", {f: round(float(x), 3) for f, x in zip(files, w)})
    rows = []
    for f, a, b, t in zip(files, Pa_l, Pb_l, Pt_l):
        rows.append(evaluate(f, a, b, t, ya, yb)[0])
    r, cal, src, q_test = evaluate("ENSEMBLE", combine(Pa_l, w), combine(Pb_l, w), combine(Pt_l, w), ya, yb)
    rows.append(r)
    table = pd.DataFrame(rows).set_index("model").round(4)
    print(table.T.to_string())
    print("estimated test class mix:", np.round(q_test, 3))

    Pt = cal(combine(Pt_l, w))
    pred = decide(Pt, q_test, src)
    pd.DataFrame({"predicted": pred}).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(pred):,} rows | predicted mix", np.round(np.bincount(pred, minlength=6)[1:] / len(pred), 3))

if __name__ == "__main__":
    main()
