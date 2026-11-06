"""Fine-tune a transformer to predict `overall` (1-5) and save probabilities for the ensemble.

Made for Google Colab (free T4 GPU), also runs on a Mac (slow). When run from this
repository, the default paths locate the data files and saved split automatically.
In Colab, pass paths to train_data.csv, test_data.csv, and split_indices.npz.

Colab:
    !pip install -q transformers
    !python scripts/transformer_colab.py --model roberta-base --n_train 120000 --epochs 1 --out_dir /content/drive/MyDrive/run1
Mac (small, slow; these are also the automatic defaults on Apple Silicon):
    python scripts/transformer_colab.py

If the run is interrupted, start the same command again with --resume.
Output: <out_dir>/transformer_probs.npz   (probabilities for validation A, validation B and test)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED, NUM_LABELS = 42, 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_input_path(filename, project_candidates=()):
    candidates = [
        *(PROJECT_ROOT / candidate for candidate in project_candidates),
        PROJECT_ROOT / "data" / filename,
        *(parent / filename for parent in PROJECT_ROOT.parents),
        Path(filename),
    ]
    return str(next((path for path in candidates if path.is_file()), Path(filename)))

def parse_args():
    p = argparse.ArgumentParser()
    on_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
    p.add_argument("--train_csv", default=default_input_path("train_data.csv"))
    p.add_argument("--test_csv", default=default_input_path("test_data.csv"))
    p.add_argument(
        "--split",
        default=default_input_path(
            "split_indices.npz",
            ("results/model_probabilities/split_indices.npz", "outputs/split_indices.npz"),
        ),
    )
    p.add_argument("--model", default="distilroberta-base" if on_mps else "roberta-base")
    default_out_dir = PROJECT_ROOT / "outputs" / "transformer"
    p.add_argument(
        "--out_dir",
        default=str(default_out_dir if PROJECT_ROOT.is_dir() else Path("runs/transformer")),
    )
    p.add_argument("--n_train", type=int, default=30_000 if on_mps else 120_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4 if on_mps else 32)
    p.add_argument("--max_len", type=int, default=128 if on_mps else 256)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def make_text(df):
    return (df["summary"].fillna("") + ". " + df["reviewText"].fillna("")).tolist()

def make_batches(lengths, batch_size, seed):
    """Shuffle, then sort by length inside chunks so that batches need little padding."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(lengths)); chunk = batch_size * 50; batches = []
    for i in range(0, len(idx), chunk):
        c = idx[i:i + chunk]; c = c[np.argsort([lengths[j] for j in c], kind="stable")]
        batches += [c[k:k + batch_size] for k in range(0, len(c), batch_size)]
    return [batches[o] for o in rng.permutation(len(batches))]

def collate(ids_list, pad_id, device):
    longest = max(len(x) for x in ids_list)
    x = torch.full((len(ids_list), longest), pad_id, dtype=torch.long); a = torch.zeros((len(ids_list), longest), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long); a[i, :len(ids)] = 1
    return x.to(device), a.to(device)

def predict_probs(model, ids, pad_id, device, use_amp, batch_size=128):
    model.eval(); order = np.argsort([len(x) for x in ids], kind="stable"); out = np.zeros((len(ids), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]; x, a = collate([ids[j] for j in sel], pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=x, attention_mask=a).logits
            out[sel] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    model.train(); return out

def main():
    args = parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    missing = [path for path in (args.train_csv, args.test_csv, args.split) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found: "
            + ", ".join(missing)
            + ". Pass the correct paths with --train_csv, --test_csv, and --split."
        )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device == "cuda"; print("device:", device, "| fp16:", use_amp); torch.manual_seed(SEED)

    data = pd.read_csv(args.train_csv, usecols=["overall", "reviewText", "summary"])      # default RangeIndex = row position
    sp = np.load(args.split)
    rng = np.random.default_rng(SEED)
    train = data.loc[rng.choice(sp["pool"], size=min(args.n_train, len(sp["pool"])), replace=False)]
    val_a, val_b = data.loc[sp["val_a"]], data.loc[sp["val_b"]]
    test = pd.read_csv(args.test_csv, usecols=["reviewText", "summary"])
    print(f"train {len(train):,} | validation A {len(val_a):,} | validation B {len(val_b):,} | test {len(test):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda df: tok(make_text(df), truncation=True, max_length=args.max_len)["input_ids"]
    tr_ids, a_ids, b_ids, te_ids = enc(train), enc(val_a), enc(val_b), enc(test)
    y_train = train["overall"].to_numpy() - 1; tr_len = [len(x) for x in tr_ids]
    print("mean tokens per training text:", round(float(np.mean(tr_len)), 1))

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_LABELS).to(device); model.train()
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = len(make_batches(tr_len, args.batch_size, SEED)); total = steps_per_epoch * args.epochs
    if args.max_steps: total = min(total, args.max_steps)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpt = os.path.join(args.out_dir, "checkpoint.pt"); step_done = 0
    if args.resume and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); step_done = ck["step"]; print("resumed from step", step_done)
    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step_done}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    start, seen, finished = time.time(), 0, False
    for epoch in range(args.epochs):
        for b_i, b in enumerate(make_batches(tr_len, args.batch_size, SEED + epoch)):
            step = epoch * steps_per_epoch + b_i
            if step < step_done: continue
            if step >= total: finished = True; break
            x, a = collate([tr_ids[j] for j in b], tok.pad_token_id, device); y = torch.tensor(y_train[b], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=x, attention_mask=a).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step_done = step + 1; seen += len(b)
            if step_done % 50 == 0 or step_done == total:
                rate = seen / (time.time() - start); eta = (total - step_done) * args.batch_size / max(rate, 1e-9) / 60
                print(f"step {step_done}/{total} loss {loss.item():.3f} {rate:.1f} samples/s ETA {eta:.0f} min", flush=True)
            if step_done % args.save_every == 0: save_ckpt()
        if finished: break
    save_ckpt()

    inference_batch_size = 4 if device == "mps" else 128
    pa = predict_probs(model, a_ids, tok.pad_token_id, device, use_amp, inference_batch_size)
    pb = predict_probs(model, b_ids, tok.pad_token_id, device, use_amp, inference_batch_size)
    ya, yb = val_a["overall"].to_numpy(), val_b["overall"].to_numpy()
    pred = np.r_[pa.argmax(1), pb.argmax(1)] + 1; truth = np.r_[ya, yb]
    print(f"VALIDATION (training class mix): micro F1 {f1_score(truth, pred, average='micro'):.4f} | macro F1 {f1_score(truth, pred, average='macro'):.4f}")
    pt = predict_probs(model, te_ids, tok.pad_token_id, device, use_amp, inference_batch_size)
    np.savez(os.path.join(args.out_dir, "transformer_probs.npz"), Pa=pa, Pb=pb, Pt=pt, ya=ya, yb=yb)
    print("saved", os.path.join(args.out_dir, "transformer_probs.npz"))

if __name__ == "__main__":
    main()
