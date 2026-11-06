# CrashLens v3 — Experiment graph and two-model video dashboard

Streamlit dashboard for Nexar collision detection and anticipation, with a traceable research archive.

## What changed since v2?

- Removed the Scored Video Explorer page.
- Replaced Project Journey with **Experiment Graph & Logs**. Four independent tracks, 15 documented steps; each stage shows recorded parameters, metric values with evaluation context, limitations, actual excerpt from source Markdown, and available epoch charts.
- Preserved all original reports and the final detection notebook.
- Moved optional prediction CSV import into Error Analysis (not global sidebar).
- Preserved **Run Two Models**, which only performs inference after compatible trained `.pth` checkpoints are present.

The graph represents relationships *within* each research track; it is not a claim that experiments from different reports form one chronological chain. Original event-centered clips can include frames after the collision; those results are not valid pre-event anticipation scores. Later alert-aware A/B results are validation-only, and experiment B is documented through 8/15 planned epochs.

## Windows startup

Run the commands from the extracted `CrashLens_v3` folder:

```bat
C:\Users\User\.venv\Scripts\python.exe -m pip install -r requirements.txt --default-timeout=120 --retries=10
C:\Users\User\.venv\Scripts\python.exe -m streamlit run app.py
```

See `README_FA.md` for Persian step-by-step instructions. The research dashboard works without PyTorch; inference requires correct checkpoint files plus torch, torchvision, OpenCV and Pillow. Do not treat this as a production-grade accident warning system without independent prospective evaluation.
