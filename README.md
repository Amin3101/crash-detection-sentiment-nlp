# Amazon electronics reviews — sentiment analysis

Quera AI bootcamp · project 2 · **problem 2**. Predict the rating `overall` (1–5) of an Amazon electronics review from its text, study the reviews, and measure satisfaction with the warranty.

| Part | What | Where |
|------|------|-------|
| **1** | Exploratory analysis: rating distribution, word clouds, top reviewers, text length, top products and brands, extra insights | `notebooks/01_eda.ipynb`, `notebooks/01b_extra_insights.ipynb` |
| **2** | Satisfaction about the warranty / guarantee per product (spelling variants + word embeddings) | `notebooks/02_aspect_warranty.ipynb`, `src/aspect.py` |
| **3** | Rating model: classical model + transformer → ensemble, corrected for the test set's class mix | `notebooks/03_model_ensemble.ipynb`, `scripts/` |
| Bonus | Interactive dashboard (one self-contained HTML file) | `dashboard/dashboard.html` |
| Report | Written report for the reviewers (Persian) | `docs/REPORT_fa.md`, `docs/report_fa.html` |
| **Submission** | `results/submission_2q.csv` — one column `predicted`, 20,000 rows | `results/` |

## Results in one table

| | Classical | Transformer | **Ensemble** |
|---|---|---|---|
| micro F1 at the training class mix (validation A+B) | 0.7187 | 0.7212 | **0.7398** |
| micro F1, validation B resampled to the **estimated test mix** | 0.5876 | 0.5916 | **0.6124** |
| macro F1, same | 0.5817 | 0.5863 | **0.6109** |

* The metric is micro F1 (= accuracy for single-label multi-class).
* The **test set is not like the training set**: 55 % of the training reviews are 5-star, while the model outputs indicate roughly 20 % per rating in the test set (estimated with BBSE and with EM + bias-corrected calibration; both agree). Test labels are hidden, so the second and third rows are *simulated* scores, and they assume that only the class mix changed (label shift). The first row is **not** the expected test score.
* Different mistakes of the two models make the ensemble about 2–3 points better than either (simulated test mix).

## Repository layout

```
data/            train_data.csv, test_data.csv, title_brand.csv         (not committed)
src/             config.py · data.py (loading, cleaning, brand normalisation) · text.py (pre-processing) · aspect.py (part 2)
notebooks/       01_eda · 01b_extra_insights · 02_aspect_warranty · 03_model_ensemble   (all with saved outputs)
scripts/         final_model.py · transformer_colab.py · ensemble.py
results/         submission_2q.csv · warranty_satisfaction_by_product.csv · model_probabilities/*.npz · run reports
dashboard/       build_dashboard.py + template.html → dashboard.html
docs/            REPORT_fa.md / report_fa.html
outputs/         scratch output of the scripts (not committed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# unzip train_data_csv.zip so that data/train_data.csv exists; put test_data.csv and title_brand.csv in data/
```
Run notebooks from the `notebooks/` folder, top to bottom. Notebook 02 writes `outputs/warranty_satisfaction_by_product.csv`; the committed copy is `results/warranty_satisfaction_by_product.csv`. `01_eda`, `01b` and `02` need the data files; `03` only needs `results/`.

## Reproducing the model

```bash
# 1) classical model (≈ 5–10 min, 8 GB RAM is enough; writes outputs/classical_probs.npz and outputs/split_indices.npz)
python scripts/final_model.py 200000

# 2) transformer (GPU, e.g. Google Colab T4). Locally, defaults find the CSVs and saved split.
python scripts/transformer_colab.py --model roberta-base --n_train 120000 --epochs 1 --out_dir run1
#    smoke test first:  python scripts/transformer_colab.py --n_train 2000 --max_steps 20 --out_dir test_run
#    Mac (slower):      python scripts/transformer_colab.py
#    On Apple Silicon, the script automatically uses distilroberta-base, 30k rows, batch size 4, and max_len 128.

# 3) ensemble + calibration + test-mix correction -> final submission
python scripts/ensemble.py results/model_probabilities/classical_probs.npz results/model_probabilities/transformer_probs.npz --out results/submission_2q.csv
```
Training settings used for the stored `transformer_probs.npz`: *(team: fill in model, number of training rows, epochs, hardware)*.
The data split is fixed (`results/model_probabilities/split_indices.npz`), so all models are evaluated on identical validation sets.

## Method in short

* **Classical model.** TF-IDF of `reviewText` and `summary` (words and word pairs, separate vectorisers), two numeric features (smoothed mean rating of the reviewer and of the product, computed out-of-fold to avoid label leakage), logistic regression.
* **Transformer.** `summary + review` fine-tuned with a standard classification head, natural class distribution.
* **Ensemble.** Log-linear combination, weights fitted on validation A (classical 0.56, transformer 0.27).
* **Label-shift correction.** Probabilities are calibrated (bias-corrected temperature scaling), the test class mix is estimated with EM, and the probabilities are multiplied by *(test mix / training mix)* before taking the maximum. Reference: Alexandari, Kundaje, Shrikumar (2020), *Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation*.

## Limitations (stated openly)

* The expected test score (≈ 0.6 micro F1) comes from a simulation under the label-shift assumption; the real score is unknown.
* Part 2 uses count-based embeddings (PPMI + SVD) instead of `word2vec`; negation is not handled, and the warranty ratings measure satisfaction *at the moment of a problem*.
* Ordinal-regression variants and larger transformers were not explored (an expected-value/threshold variant gave no gain in micro F1).
* The two decisions of part 2 (final word list, minimum 5 warranty reviews per product) were proposed by one team member and should be confirmed by the team.
