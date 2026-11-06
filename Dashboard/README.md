# CrashLens Final v6

A single Streamlit presentation dashboard combining the Nexar Computer Vision workstream and the team's Amazon-review NLP dashboard.

The main navigation has **five pages**: four Computer Vision pages and one NLP workspace. The NLP workspace embeds the original self-contained team dashboard with all eight interactive tabs, and adds a local classical-model training/inference console.

## Run on Windows

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --default-timeout=120 --retries=10
.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

If the virtual environment already exists, skip the first command. `run_windows.bat` can use the project-local `.venv` or `%USERPROFILE%\\.venv`.

## Main pages

1. **CV Overview & Architecture** — Nexar project summary, the Detection and Anticipation architecture trees, and EDA.
2. **Video Demo · Two Models** — upload a dashcam video and run either or both models when the matching checkpoints are present.
3. **Experiments & Data Audit** — experiment graph, source logs, training history, ablations and leakage checks.
4. **CV Evaluation & Errors** — reported evaluation results, confusion matrix and optional sample-level error analysis.
5. **NLP Dashboard & Run Model** — original team dashboard with eight tabs, plus a local training and inference workflow.

## NLP dashboard

The embedded team dashboard preserves: Overview, Ratings, Words, Reviewers & Votes, Text Length, Products & Brands, Warranty, and Model. Its chart data is embedded in the original HTML, so viewing it does not require the raw dataset or an internet connection.

The archive contains saved probability arrays and submissions, but does not contain the raw `train_data.csv`, `test_data.csv` or `title_brand.csv`, a fitted classical vectorizer/classifier, or Transformer weights. Saved probabilities support evaluation of the original validation/test samples; they cannot score arbitrary new text.

## NLP model runner

To create a deployable classical checkpoint, place the team's labelled training data at:

```text
sources/nlp_team/amazon-sentiment/data/train_data.csv
```

Then use the **Run NLP model** view in the NLP workspace, or run:

```bat
.venv\Scripts\python.exe scripts\train_nlp_inference.py --train-csv sources\nlp_team\amazon-sentiment\data\train_data.csv --n-train 200000
```

The checkpoint is saved to `models/nlp_classical.joblib`. It runs the classical TF-IDF + Logistic Regression branch with reviewer/product history features. This is a deployable retraining workflow, not the missing original ensemble checkpoint.

The original Transformer and final Ensemble cannot run on new text until the matching Transformer weights, tokenizer and base-model identity are supplied. The dashboard does not fabricate these predictions.

Optional EM label-shift correction is exposed only for uploaded batches of 1,000+ reviews and is explicitly exploratory.

## Computer Vision checkpoints

Place the two distinct, architecture-matched files at:

```text
models/detection.pth
models/anticipation.pth
```

They are not included in this ZIP. Do not copy one checkpoint into both locations. See `models/README.txt`.

## Sources

- `assets/nlp_team_dashboard.html` — exact self-contained NLP dashboard embedded in Streamlit.
- `sources/nlp_team/amazon-sentiment/` — original team code, notebooks, report, scripts and saved results.
- `sources/` and `data/` — Computer Vision source notebooks, experiment logs and dashboard data.
- `nlp_runtime.py` and `scripts/train_nlp_inference.py` — deployable classical NLP training and inference workflow.
- `docs/CrashLens_Final_v6_Team_Guide_FA.docx` — Persian team handoff guide.
