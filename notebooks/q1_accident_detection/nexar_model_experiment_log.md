# Nexar Collision Anticipation --- Model Experiment Log

**Status:** Experiment B is documented through epoch 8 of 15.\
**Task:** Anticipate an impending collision from video frames that occur
before the collision.

## 1. Task and temporal labeling

Dataset: `nexar-ai/nexar_collision_prediction`.

This is a temporal collision-anticipation task, not merely video-level
collision recognition.

  Video interval                                   Target
  -------------------------------------------- ----------
  Before `time_of_alert`                                0
  From `time_of_alert` until `time_of_event`            1
  After `time_of_event`                          Excluded
  Negative video                                        0

The original video-level label is not used as a substitute for the
temporal target.

### Clip sampling

-   4-second clips, 16 frames, 224 × 224 resolution.
-   Positive clips sampled approximately from `time_of_event - 5s` to
    `time_of_event - 1s`.
-   No post-collision frames are included.
-   Negative clips are matched to positive clip start-time and
    time-remaining distributions.
-   Processed data: `nexar_data/processed_frames_matched3`.

## 2. Dataset split and DataLoader

The temporal manifest contains 1,364 clips:

  Temporal group                Count
  ----------------------- -----------
  Negative                        616
  Positive before alert           188
  Alert to event                  560
  **Total**                 **1,364**

Target counts: class 0 = 804; class 1 = 560.

A `StratifiedGroupKFold` split grouped by `video_id` was used,
preventing the same video from appearing in both splits.

  Split          Clips   Videos   Target 0   Target 1
  ------------ ------- -------- ---------- ----------
  Train          1,091    1,050        643        448
  Validation       273      262        161        112

Files: `train_manifest.csv`, `val_manifest.csv`, stored in
`nexar_data/processed_frames_matched3`.

### Loading and augmentation

-   Dataset returns `(frames, target_label)`.
-   Tensor shape: `[B, 3, 16, 224, 224]`.
-   Training batch size: 4.
-   Weighted sampler with replacement.
-   Temporal-group weights: negative = 1, positive-before-alert = 2,
    alert-to-event = 3.
-   Sampled training distribution: approximately 36.8% negative / 63.2%
    positive.
-   Validation loader is unweighted and uses `shuffle=False`.
-   Augmentation: random horizontal flip when enabled, random
    crop/resize, brightness, contrast, saturation.
-   Normalization mean: `[0.45, 0.45, 0.45]`.
-   Normalization std: `[0.225, 0.225, 0.225]`.

Training metrics reflect weighted sampling and should not be directly
compared to unweighted validation metrics.

## 3. Shared backbone and forward path

Both experiments use pretrained Torchvision **MViT-v2-S**.

Input: `[B, 3, 16, 224, 224]`.

The custom forward path: 1. Applies `backbone.conv_proj`. 2. Flattens
and transposes patch tokens. 3. Applies MViT positional encoding. 4.
Runs the MViT transformer blocks. 5. Applies backbone normalization. 6.
Separates the CLS token from patch tokens. 7. Reshapes patch tokens to
`[B, T, H, W, C]`.

The temporal head uses feature dimension `C = 768`. The CLS token is
concatenated with temporal features before classification.

## 4. Experiment A --- Original MViT with attention-based temporal pooling

### Architecture

**Backbone:** pretrained MViT-v2-S.

**Spatial features:** - Average-pool patch tokens over H and W for each
frame. - Max-pool patch tokens over H and W for each frame. -
Concatenate the two 768-dimensional vectors (1,536 dimensions). -
Project 1,536 → 768 using Linear, LayerNorm, GELU, and Dropout.

**Temporal attention pooling:** - Linear 768 → 128. - Tanh. - Linear 128
→ 1. - Softmax over time and weighted sum.

**Temporal summaries:** attention-pooled vector, temporal mean, and
temporal maximum, each 768-dimensional.

The CLS token (768) and these three temporal vectors are concatenated to
3,072 dimensions.

**Classifier:**

``` python
LayerNorm(3072)
Linear(3072, 512)
GELU()
Dropout(0.4)
Linear(512, 256)
GELU()
Dropout(0.3)
Linear(256, 2)
```

A separate BiGRU experiment was previously tried and performed worse; it
is not part of these two current architectures.

### Optimizer configuration

  Parameter group                Learning rate
  ---------------------------- ---------------
  Lower MViT backbone                     1e-6
  Top four MViT blocks                    5e-6
  Temporal head + classifier              2e-5
  Weight decay                            0.01

The backbone was initially frozen and unfrozen at epoch 5.

### Training log

  ------------------------------------------------------------------------
        Epoch  Train loss    Val loss Train Macro   Val Macro      Val AUC
                                               F1          F1 
  ----------- ----------- ----------- ----------- ----------- ------------
            1      0.6724      0.7702      0.4305      0.2909       0.6049

            2      0.6547      0.6943      0.4302      0.4453       0.6790

            3      0.6573      0.7361      0.4825      0.3358       0.6920

            4      0.6602      0.6666      0.4821      0.6025       0.7231

            5      0.6409      0.7483      0.5032      0.3549       0.7180

            6      0.6494      0.6703      0.4814      0.5080       0.7372

            7      0.6365      0.6743      0.5569      0.5539       0.7434

            8      0.6090      0.7345      0.6228      0.5051       0.7498

            9      0.6201      0.5883      0.6412      0.6879       0.7483

           10      0.6163      0.6151      0.6483      0.6731       0.7527

           11      0.5970      0.6157      0.6513      0.6481       0.7550

           12      0.5909      0.6015      0.6640      0.6649       0.7569

           13      0.5639      0.6308      0.6681      0.6483       0.7568

           14      0.5722      0.6106      0.6742      0.6476       0.7569

           15      0.5655      0.6216      0.6873      0.6553   **0.7571**
  ------------------------------------------------------------------------

The retained log did not include train/validation accuracy for this run,
so those values are not reproduced here.

### Best validation results

Best AUC: **0.7571 at epoch 15**.

At threshold 0.50: - Macro F1: 0.6553. - Confusion matrix:

``` text
[[94, 67],
 [27, 85]]
```

At the validation-selected threshold of approximately 0.64: - Macro F1:
0.7131. - Macro precision: 0.7126. - Macro recall: 0.7137. - Confusion
matrix:

``` text
[[122, 39],
 [ 37, 75]]
```

Classification report:

  Class             Precision   Recall       F1   Support
  --------------- ----------- -------- -------- ---------
  0                    0.7673   0.7578   0.7625       161
  1                    0.6579   0.6696   0.6637       112
  Macro average        0.7126   0.7137   0.7131       273

Reported confidence intervals: - AUC: 0.7571, 95% CI \[0.6953,
0.8165\]. - Macro F1: 0.7131, 95% CI \[0.6542, 0.7682\].

Threshold sweep:

    Threshold   Macro F1   Macro Precision   Macro Recall
  ----------- ---------- ----------------- --------------
         0.50     0.6553            0.6680         0.6714
         0.55     0.6645            0.6684         0.6739
         0.60     0.6940            0.6930         0.6978
         0.65     0.7073            0.7083         0.7065
         0.70     0.7026            0.7152         0.6988
         0.75     0.6901            0.7284         0.6865
         0.79     0.6605            0.7284         0.6619
         0.85     0.5850            0.7632         0.6126

Checkpoint reported: `nexar_data/best_mvit_collision.pth`.

## 5. Experiment B --- MViT + Temporal Transformer + Multi-Scale Temporal CNN

### Motivation

This experiment changes the temporal head while keeping the pretrained
MViT-v2-S backbone and manual CLS/patch-token extraction.

It uses two parallel temporal branches: 1. Temporal Transformer for
relationships across the 16 frame-level feature vectors. 2. Multi-scale
Conv1D branch for temporal patterns at different receptive-field sizes.

### Architecture details

**Input to temporal head:** `[B, T, H, W, C]`, with `T = 16`, `C = 768`.

#### Spatial projection

-   Average-pool over H and W.
-   Max-pool over H and W.
-   Concatenate 768 + 768 = 1,536 features per frame.
-   Linear 1,536 → 768.
-   LayerNorm, GELU, Dropout 0.2.

Output: `[B, 16, 768]`.

#### Transformer branch

  Parameter                                Value
  ------------------------ ---------------------
  Model dimension                            768
  Attention heads                              8
  Encoder layers                               2
  Feed-forward dimension                   1,536
  Activation                                GELU
  Dropout                                    0.2
  Batch layout                `batch_first=True`
  Positional embeddings      Learnable, 16 × 768

The Transformer output is normalized and attention-pooled over time.

#### Multi-scale temporal CNN branch

The input is transposed to `[B, 768, 16]` and sent through three
parallel Conv1D branches:

  Branch     Kernel   Padding   Output channels
  -------- -------- --------- -----------------
  Short           3         1               256
  Medium          5         2               256
  Wider           7         3               256

Each branch uses Conv1D, BatchNorm1D, and GELU.

The outputs are concatenated (768 channels), then passed through: -
Conv1D with kernel 1, 768 → 768. - BatchNorm1D. - GELU. - Dropout 0.2.

The result is transposed back to `[B, 16, 768]` and attention-pooled.

#### Branch fusion and temporal summaries

The Transformer pooled vector and CNN pooled vector are concatenated
(1,536 dimensions), then fused using: - Linear 1,536 → 768. -
LayerNorm. - GELU. - Dropout 0.2.

The temporal head returns: - Fused attention feature: 768. - Temporal
mean: 768. - Temporal maximum: 768.

The main model concatenates these with the CLS token:

``` text
CLS token                 768
Fused attention feature   768
Temporal mean             768
Temporal maximum          768
Total                    3072
```

#### Classifier

Unchanged from Experiment A:

``` python
LayerNorm(3072)
Linear(3072, 512)
GELU()
Dropout(0.4)
Linear(512, 256)
GELU()
Dropout(0.3)
Linear(256, 2)
```

### Optimizer and schedule

The intended three parameter groups were:

  Parameter group                Configured learning rate
  ---------------------------- --------------------------
  Lower MViT backbone                                1e-6
  Top four MViT blocks                               5e-6
  Temporal head + classifier                         2e-5
  Weight decay                                       0.01
  Top backbone blocks                                   4

**Important discrepancy:** The actual epoch log prints an initial head
LR of `5e-6`, not `2e-5`. The printed learning rates are recorded below;
inspect the notebook's optimizer construction to determine the exact
configured initial head LR. The temporal head must be included in the
head parameter group.

The backbone was unfrozen at epoch 5.

### Learning rates printed in the log

    Epoch   Backbone LR   Head LR
  ------- ------------- ---------
        1       1.00e-6   5.00e-6
        2       9.89e-7   4.94e-6
        3       9.55e-7   4.76e-6
        4       9.02e-7   4.47e-6
        5       8.31e-7   4.08e-6
        6       7.45e-7   3.61e-6
        7       6.50e-7   3.10e-6
        8       5.50e-7   2.55e-6

### Training results reported so far

This run was planned for 15 epochs, but only epochs 1--8 have been
provided.

  ---------------------------------------------------------------------------
     Epoch    Train Val loss    Train  Val acc    Train      Val      Val AUC
               loss               acc          Macro F1 Macro F1 
  -------- -------- -------- -------- -------- -------- -------- ------------
         1   0.6711   0.7509   0.6169   0.4103   0.4130   0.2909       0.6542

         2   0.6670   0.7224   0.6141   0.4103   0.4747   0.2909       0.6978

         3   0.6598   0.6968   0.6398   0.4982   0.5032   0.4629       0.7019

         4   0.6525   0.7734   0.6315   0.4872   0.5556   0.4377       0.7257

         5   0.6340   0.6791   0.6719   0.5934   0.5833   0.5897       0.7183

         6   0.6350   0.6095   0.6664   0.6740   0.6106   0.6417       0.7264

         7   0.6380   0.7025   0.6645   0.5495   0.5954   0.5325       0.7466

         8   0.6151   0.6251   0.6984   0.6630   0.6353   0.6626   **0.7493**
  ---------------------------------------------------------------------------

### Epoch 8 validation results

At threshold 0.50: - Macro F1: 0.6626. - AUC: 0.7493. - Confusion
matrix:

``` text
[[95, 66],
 [26, 86]]
```

At validation-selected threshold 0.65: - Macro F1: 0.7209. - Macro
precision: 0.7238. - Macro recall: 0.7189. - Accuracy: 0.7326. -
Confusion matrix:

``` text
[[128, 33],
 [ 40, 72]]
```

Classification report:

  Class                Precision   Recall       F1   Support
  ------------------ ----------- -------- -------- ---------
  0                       0.7619   0.7950   0.7781       161
  1                       0.6857   0.6429   0.6636       112
  Macro average           0.7238   0.7189   0.7209       273
  Weighted average        0.7306   0.7326   0.7311       273

Reported intervals: - AUC: 0.7493, 95% CI \[0.6878, 0.8102\]. - Macro
F1: 0.7209, 95% CI \[0.6628, 0.7776\].

Threshold sweep:

    Threshold     Macro F1   Macro Precision   Macro Recall
  ----------- ------------ ----------------- --------------
         0.50       0.6626            0.6755         0.6790
         0.55       0.6761            0.6816         0.6873
         0.60       0.7021            0.7012         0.7067
         0.65   **0.7209**            0.7238         0.7189
         0.70       0.6546            0.6784         0.6528
         0.75       0.6252            0.7098         0.6337
         0.79       0.5581            0.7490         0.5947
         0.85       0.4089            0.7993         0.5179

Checkpoint message: `nexar_data/best_mvit_collision.pth`.

**Checkpoint warning:** This is the same path reported for Experiment A.
Verify whether the new run overwrote the old checkpoint. Future runs
should use distinct filenames, such as: -
`nexar_data/best_mvit_original_temporal.pth` -
`nexar_data/best_mvit_transformer_multiscale.pth`

## 6. Direct comparison

  -------------------------------------------------------------------------------
  Metric                      Experiment A:      Experiment B:   Difference B − A
                          Original head (15  Transformer + CNN 
                                    epochs)          (epoch 8) 
  ---------------------- ------------------ ------------------ ------------------
  Best reported AUC                  0.7571             0.7493            -0.0078

  Macro F1 at 0.50                   0.6553             0.6626            +0.0073

  Validation-optimized               0.7131             0.7209            +0.0078
  Macro F1                                                     

  Optimized threshold                \~0.64               0.65                ---
  -------------------------------------------------------------------------------

Interpretation: - Experiment B has a more expressive temporal head. - At
epoch 8, its AUC is slightly lower than Experiment A's completed AUC. -
Its threshold-optimized Macro F1 is slightly higher. - Experiment B is
incomplete (8/15 epochs), so this is a provisional comparison, not a
final ranking.

## 7. Evaluation cautions

### Threshold selection

The optimized thresholds were selected on the validation set. Scores
calculated at those same thresholds are useful for development but are
optimistic estimates of performance on unseen data.

### Confidence intervals

The reported confidence intervals are retained as supplied. The exact
bootstrap method is not documented. Because clips can be grouped within
videos, video-level bootstrap resampling by `video_id` is preferable to
clip-level bootstrap.

### Test-set status

No independent test set is documented for these experiments. The
reported metrics are validation results and should not be described as
independent test performance.

## 8. Hardware and software context

Reported Windows GPU environment:

  Component        Configuration
  ---------------- ------------------------------------
  GPU              NVIDIA GeForce RTX 5050 Laptop GPU
  VRAM             8 GB
  CPU              Intel Core i5, 13th generation
  RAM              16 GB
  PyTorch          2.14.1 + CUDA 13.0
  CUDA available   Yes

The current model experiments use PyTorch. A separate CPU-only
Linux/Omarchy environment was also discussed, but it is not the GPU
environment above.

## 9. Current status and next steps

-   [x] Define temporal anticipation labels and pre-event sampling.
-   [x] Create grouped train/validation manifests.
-   [x] Complete Experiment A for 15 epochs.
-   [x] Implement Experiment B with Transformer and multi-scale temporal
    convolutions.
-   [ ] Complete Experiment B epochs 9--15.
-   [ ] Verify separate checkpoint files.
-   [ ] Compare completed experiments on the same validation split.
-   [ ] Evaluate an independent video-grouped test set.

Recommended immediate action: finish Experiment B without changing its
architecture or data pipeline mid-run. Then compare AUC, Macro F1 at
threshold 0.50, threshold-selected Macro F1, and positive-class
precision/recall.

A possible later architecture experiment is a two-branch temporal model
with cross-attention between short-term and long-term representations.
This is a proposal only; it has not yet been trained or evaluated.

## 10. Reproducibility details still to verify

The available logs do not establish all of the following: - Exact random
seeds for splitting, sampling, and model initialization. - Exact
scheduler class and configuration. - Whether Experiment B's initial head
LR was `2e-5` or `5e-6` (the log prints `5e-6`). - Exact mixed-precision
settings. - Exact checkpoint-selection criterion. - Bootstrap method and
resampling unit for confidence intervals. - Whether horizontal flip was
enabled in each run. - Whether Experiment B overwrote Experiment A's
checkpoint.

These should be checked in the notebook/configuration rather than
inferred.

## 11. Executive summary

**Experiment A --- Original MViT attention temporal head** - 15 epochs
completed. - AUC: 0.7571. - Macro F1 at 0.50: 0.6553. -
Validation-optimized Macro F1: 0.7131 at approximately 0.64.

**Experiment B --- MViT + 2-layer, 8-head temporal Transformer +
parallel Conv1D branches (kernels 3, 5, 7)** - 8 of 15 epochs
reported. - AUC so far: 0.7493 at epoch 8. - Macro F1 at 0.50: 0.6626. -
Validation-optimized Macro F1: 0.7209 at 0.65.

**Conclusion:** Experiment B is competitive and currently has slightly
higher validation-optimized Macro F1, but slightly lower AUC than the
completed baseline. Its run is incomplete. Do not declare a definitive
winner until all 15 epochs are complete and checkpoint integrity is
verified.
