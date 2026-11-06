"""Print architecture clues for local trusted PyTorch weight files, not their tensor data."""
from pathlib import Path

try:
    import torch
except ImportError:
    raise SystemExit("Please install torch first (see README_FA.md).")

root = Path(__file__).resolve().parent / "models"
for name in ["detection.pth", "anticipation.pth"]:
    path = root / name
    print("\n---", name, "---")
    if not path.exists():
        print("NOT FOUND:", path)
        continue
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        weights = data.get("model_state_dict", data.get("state_dict", data))
        keys = list(weights.keys())
        print("File MB:", round(path.stat().st_size/1024**2, 2))
        print("Tensor keys:", len(keys))
        print("Temporal transformer:", any("temporal_pool.temporal_transformer" in k for k in keys))
        print("Multi-scale CNN:", all(any(f"temporal_pool.conv{i}" in k for k in keys) for i in (3,5,7)))
        print("Plain backbone head:", any(k.startswith("backbone.head") for k in keys))
        print("Sample key names:", keys[:8])
        print("Checkpoint metadata:", {k: data.get(k) for k in ["epoch", "best_val_auc", "val_f1", "class_names"] if k in data})
    except Exception as exc:
        print("Cannot safely inspect:", type(exc).__name__, str(exc)[:400])
