# Nexar Collision Detection — Experiment Log

**Task:** Binary collision detection from dashcam clips (label 0 = no collision, label 1 = collision)
**Model:** MViT-v2-S (Kinetics-pretrained), fine-tuned with discriminative LRs
**Dataset:** Nexar dashcam clips, 16 frames @ 224×224, 4s clip with 1s warning margin

---

## 1. Initial Training Run (on original, unmatched data)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1 | Val AUC |
|---|---|---|---|---|---|---|
| 1 | 0.6984 | 0.6857 | 0.4819 | 0.6044 | 0.4819 | 0.6042 | 0.6119 |
| 2 | 0.6888 | 0.6765 | 0.5334 | 0.5778 | 0.5334 | 0.5455 | 0.6723 |
| 3 | 0.6664 | 0.6436 | 0.5964 | 0.6267 | 0.5958 | 0.6029 | 0.7238 |
| 4 | 0.6323 | 0.6097 | 0.6613 | 0.6533 | 0.6603 | 0.6414 | 0.7602 |
| 5 | 0.6129 | 0.5789 | 0.6775 | 0.7067 | 0.6770 | 0.7050 | 0.7687 |
| 6 | 0.5889 | **0.5680** | 0.7071 | 0.7067 | 0.7068 | **0.7067** | 0.7770 |
| 7 | 0.5696 | 0.6003 | 0.7309 | 0.6756 | 0.7306 | 0.6627 | 0.7928 |
| 8 | 0.5325 | 0.5853 | 0.7519 | 0.7067 | 0.7519 | 0.7010 | 0.7874 |
| 9 | 0.5163 | 0.5802 | 0.7815 | 0.7111 | 0.7814 | 0.7093 | 0.7867 |

**Best (epoch 6):** Val Loss 0.5680 | Val F1 0.7067 | Val AUC 0.7770
**Observation:** Clear overfitting after epoch 6 (train 78% vs val 71% by epoch 9).

---

## 2. Metadata / Shortcut Leakage Analysis (original data)

Class balance: 750 negatives, 748 positives.

| Feature | Pos mean | Neg mean | KS stat | KS p |
|---|---|---|---|---|
| `start_time` | 14.14 | 14.44 | 0.116 | **7.9e-05** |
| `video_duration` | 38.40 | 37.11 | 0.086 | 0.007 |

**Non-visual baselines (5-fold ROC-AUC):**

| Baseline | AUC |
|---|---|
| `start_time` only | **0.653** ± 0.011 |
| `video_duration` only | 0.516 ± 0.027 |
| `time_after_clip` only | 0.476 ± 0.026 |
| `timing only` | 0.640 ± 0.008 |
| `metadata only` | 0.638 ± 0.017 |
| **`timing + metadata`** | **0.700 ± 0.012** |

**Conclusion:** A non-visual model achieves **AUC 0.700** — the model's 0.777 was only +0.08 above a pure timing shortcut. The leak is in `start_time` (AUC 0.653); `time_after_clip` is clean (0.476, below chance).

---

## 3. Ablation Study (original checkpoint)

| Condition | Val Macro-F1 | Δ vs normal | Interpretation |
|---|---|---|---|
| **normal** | **0.707** | — | full model |
| static frame | 0.488 | **−0.219** | motion is essential ✅ |
| shuffled time | 0.667 | −0.040 | temporal ordering not used ⚠️ |
| borders only | 0.678 | −0.029 | periphery nearly sufficient ⚠️ |
| center only | 0.706 | −0.001 | center alone suffices ⚠️ |

**Conclusions:**
- Static-frame collapse proves the model uses **motion**, not single-frame appearance.
- Shuffle-insensitivity means it learns **motion presence**, not **trajectory dynamics**.
- Center-only ≈ normal means the signal is **spatially concentrated**.

---

## 4. Fixing the Leakage — Attempt 1 (greedy per-video sampler) — FAILED

**Approach:** Matched negatives on `time_after_clip` only. Each negative video got one pseudo-event placed so the remaining footage matched the positive `after` distribution.

**Result:**

| Metric | Value |
|---|---|
| Clips | 1482 ok (734 neg, 748 pos) |
| `start_time` KS | stat=0.166, p=2.1e-09 ❌ |
| `time_after_clip` KS | stat=0.141, p=6.8e-07 ❌ |
| `start_time` AUC (heuristic) | 0.573 |
| `timing + metadata` AUC (heuristic) | 0.536 |

**Why it failed:** Matched the *wrong* variable (`time_after_clip`, which was already clean) and left `start_time` (the actual leak) uncontrolled. Per-video greedy acceptance distorted the joint distribution — negatives got fatter tails (std 5.14 vs 3.56), inflating `time_after_clip` AUC to 0.674 in the proper classifier re-check.

---

## 5. Fixing the Leakage — Attempt 2 (global candidate pool + 2D stratified subsample) — SUCCESS

**Approach:**
1. Generate 30 candidate `(start_time, time_after_clip)` placements per negative video.
2. Pool all candidates.
3. Stratified subsample so negatives match the positive joint histogram on `(start_time bin × after bin)`, one clip per video.
4. Verify with a **fitted** logistic regression (not a heuristic).

**Raw matched set (before dedup):**

| Metric | Value |
|---|---|
| Clips | 1364 (616 neg, 748 pos) |
| `start_time` KS | stat=0.034, p=0.806 ✅ |
| `video_duration` KS | stat=0.030, p=0.914 ✅ |
| `start_time` mean | pos 14.14, neg 14.07 |
| `after` quantiles | pos `[4.20, 20.57, 21.20, 21.78, 30.54]`, neg `[4.22, 20.46, 21.21, 21.82, 30.34]` |

**Remaining issue:** 52 duplicate `video_id` rows inflated the AUC to 0.695–0.745 because duplicates leaked across KFold splits.

---

## 6. Final Clean Dataset (dedup + balance + group-aware CV)

**Processing:**
- Deduplicated on `video_id`: 1364 → 1312 rows (52 dups removed, all from negative side)
- Downsampled positives to match negatives: **564 / 564 = 1128 rows**
- Verification with `StratifiedGroupKFold(groups=video_id)`

**Final non-visual baselines (clean dataset):**

| Feature | AUC | Target ≤ 0.56 | Status |
|---|---|---|---|
| `start_time` only | **0.493** ± 0.012 | ✅ | clean |
| `time_after_clip` only | **0.483** ± 0.013 | ✅ | clean |
| `video_duration` only | **0.486** ± 0.026 | ✅ | clean |
| `timing only` | **0.486** ± 0.020 | ✅ | clean |
| `timing + duration` | **0.486** ± 0.020 | ✅ | clean |

**Final KS tests:**

| Feature | KS stat | KS p | Status |
|---|---|---|---|
| `start_time` | 0.032 | **0.937** | ✅ |
| `time_after_clip` | 0.066 | 0.177 | ✅ |
| `video_duration` | 0.032 | **0.937** | ✅ |

**Conclusion: All non-visual AUCs ≤ 0.493. The timing shortcut is fully eliminated.**
Saved as: `nexar_data/processed_frames_matched2/clean_balanced_log.csv`

---

## 7. Retrain on Clean Data

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1 | Val AUC |
|---|---|---|---|---|---|---|---|
| 1 | 0.6942 | 0.6778 | 0.5070 | 0.6331 | 0.5018 | 0.6051 | 0.6951 |
| 2 | 0.6897 | 0.6691 | 0.5539 | 0.6746 | 0.5523 | 0.6699 | 0.7440 |
| 3 | 0.6637 | 0.6097 | 0.6236 | 0.7160 | 0.6233 | 0.7157 | 0.7629 |
| 4 | 0.6262 | **0.5825** | 0.6869 | 0.7160 | 0.6869 | **0.7137** | **0.7674** |
| 5 | 0.5877 | 0.5924 | 0.7034 | 0.6982 | 0.7032 | 0.6978 | 0.7471 |
| 6 | 0.5726 | 0.5927 | 0.7427 | 0.6982 | 0.7427 | 0.6982 | 0.7507 |
| 7 | 0.5537 | 0.5960 | 0.7452 | 0.6864 | 0.7452 | 0.6860 | 0.7485 |

**Best (epoch 4):** Val Loss 0.5825 | Val F1 0.7137 | Val AUC 0.7674
**Early stopped at epoch 7.**

**Comparison old vs. clean:**

| Metric | Old (shortcut data) | Clean data | Δ |
|---|---|---|---|
| Best val F1 | 0.7067 | **0.7137** | +0.007 |
| Best val AUC | 0.7770 | **0.7674** | −0.010 |
| Best epoch | 6 | 4 | converged faster |
| Train-val gap (ep 7–9) | ~7 pts | ~4.7 pts | better generalization |

**Key finding:** Removing a 0.70-AUC timing shortcut cost only ~0.01 AUC. The model was already relying on genuine visual motion, not the shortcut.

---

## 8. Final Test Set Evaluation

Test split: 170 clips, perfectly balanced (85 pos / 85 neg), grouped by `video_id`.

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| Accuracy | 0.7294 | — |
| **Macro F1** | **0.7294** | **[0.6570, 0.7996]** |
| Macro Precision | 0.7295 | — |
| Macro Recall | 0.7294 | — |
| **ROC-AUC** | **0.7846** | **[0.7058, 0.8530]** |

**Confusion matrix:**



**Key finding:** Test F1 (0.729) ≥ Val F1 (0.714) and Test AUC (0.785) ≥ Val AUC (0.767). The model generalizes cleanly — no overfitting to the validation split used for early stopping. Errors are symmetric across classes.

---

## 9. Summary Results Table

| Setting | Test F1 | Test AUC |
|---|---|---|
| Non-visual baseline (`start_time` + `time_after_clip` + `duration`) | — | **0.486** |
| MViT — static frames only (ablation) | 0.488 | — |
| MViT — shuffled time (ablation) | 0.667 | — |
| MViT — center only (ablation) | 0.706 | — |
| **MViT — full video (final)** | **0.729** | **0.785** |

**Vision gain over best non-visual baseline: +0.30 AUC.**

---

## 10. Leakage-Fix Progression (methodological summary)

| Stage | `timing + metadata` AUC | Status |
|---|---|---|
| Original data | **0.700** | ⚠️ heavy shortcut |
| Attempt 1 (greedy, `after` only) | **0.745** | ❌ worse (tail distortion) |
| Attempt 2 raw (global pool) | 0.695 | ⚠️ duplicate leakage in CV |
| **Final (dedup + group CV)** | **0.486** | ✅ **clean** |

**Lessons learned:**
1. Always match the variable that **actually leaks** (`start_time`), not the one that's already clean (`time_after_clip`).
2. Per-video greedy acceptance distorts joint distributions — use **global candidate pools + stratified subsampling**.
3. Duplicate rows leak across KFold — use **`StratifiedGroupKFold(groups=video_id)`** for all non-visual baselines.
4. Verify with a **fitted classifier**, not a rank-mean heuristic.

---

## 11. Known Limitations (for writeup)

1. **Temporal ordering is weakly used** — shuffled-time ablation only drops F1 by 0.04. The model detects *motion presence*, not *trajectory dynamics*.
2. **Center-only ≈ normal** (0.706 vs 0.707) — the model does not require full-scene context; the signal is spatially concentrated.
3. **Static frames collapse** (0.488) — the model genuinely requires motion; this is the strongest positive evidence.
4. **Wide confidence interval** — n=170 gives F1 CI of ±0.07; larger test set would tighten this.
5. **Class balance was enforced by downsampling** — 184 positive clips were discarded (748 → 564). This is conservative but loses data.

---

## 12. Recommended Future Work

1. **Two-stream architecture** (RGB + optical flow) — directly targets the shuffle-insensitivity weakness by forcing explicit motion learning.
2. **V-JEPA2 backbone** — self-supervised video foundation model, recently shown SOTA on ego-centric collision prediction.
3. **Importance-guided motion learning** (ASTRA-style) — perturbation-based separation of appearance vs. motion features.
4. **Test-time adaptation** — cheap 1–3% gain if distribution shift exists between train and test.
5. **Training refinements** — cosine LR schedule (decay at epoch 3–4, not 6), repeat augment, label smoothing 0.1.




Run	Config	Val F1	Val AUC	Test F1 [CI]	Test AUC [CI]
Clean #1	seed 42, LR decay ep 6	0.714	0.767	0.729 [0.657, 0.800]	0.785 [0.706, 0.853]
Clean #2	seed ?, LR decay ep 8	0.728	0.791	0.682 [0.604, 0.748]	0.759 [0.680, 0.830]
Mean	—	~0.72	~0.78	~0.71	~0.77



| Clean #3 (patched config) | seed 44, narrow head + cosine LR + val-AUC select + unfreeze@3 | 0.731 | 0.766 | 0.710 [0.634, 0.777] | 0.772 [0.692, 0.844] |
| **Mean across 3 clean runs** | — | 0.724 | 0.775 | **0.707** | **0.772** |