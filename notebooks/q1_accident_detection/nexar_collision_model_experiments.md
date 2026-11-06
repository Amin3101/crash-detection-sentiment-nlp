# Nexar Collision Prediction --- Model Experiments and Results

## 1. Project and task

Binary video classification using the Hugging Face dataset
`nexar-ai/nexar_collision_prediction`.

-   **Class 0:** Negative / no collision
-   **Class 1:** Positive / collision
-   Metadata discussed: `time_of_event`, `time_of_alert`,
    `light_conditions`, `weather`, `scene`, `time_to_accident`.
-   Video preprocessing experiment: a 4-second clip centered around
    `time_of_event` (approximately 2 seconds before and 2 seconds
    after), sampled into 16 frames at 224 × 224.

**Task-definition caveat:** Including frames after `time_of_event`
supports classifying whether a collision occurred, but does not
establish that the model predicts a collision before it happens. An
early-warning task must use only frames available before the event.

## 2. Current model architecture

The current model uses a pretrained **MViT V2 Small** video backbone
from `torchvision.models.video`, with its original classifier replaced
by a custom MLP head.

MViT means Multiscale Vision Transformer. It extracts spatiotemporal
video features using transformer attention and multiscale
representations. The pretrained backbone is initialized with
`MViT_V2_S_Weights.DEFAULT`.

### Architecture

``` text
Input video [B, C, T, H, W]
          |
          v
   MViT V2 Small
   (pretrained backbone)
          |
          v
      LayerNorm
          |
          v
 Linear(in_features, 512)
          |
          v
        GELU
          |
          v
      Dropout(0.3)
          |
          v
    Linear(512, 256)
          |
          v
        GELU
          |
          v
      Dropout(0.2)
          |
          v
      Linear(256, 2)
          |
          v
 Negative / Positive logits
```

`in_features` is obtained from the backbone at runtime, rather than
hard-coded.

### Current model code

``` python
class NexarCollisionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = mvit_v2_s(weights=weights)

        in_features = self.backbone.head[1].in_features
        self.backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


model = NexarCollisionModel(num_classes=2, pretrained=True).to(device)
```

The dataset returns each clip as `[C, T, H, W]`; a batch is therefore
`[B, C, T, H, W]`, the layout expected by the torchvision video model.

## 3. Dataset preprocessing and normalization

The stored `.npy` clip is assumed to have `THWC` layout and is converted
to `CTHW`:

``` python
frames = torch.from_numpy(np.load(row["clip_path"]).copy())
frames = frames.permute(3, 0, 1, 2).float() / 255.0

mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

frames = (frames - mean) / std
```

The same normalization is used for train, validation, and test.
Augmentation is enabled only for the training dataset.

### Earlier augmentation implementation

The original `NexarDatasetAug` included:

-   Temporal frame dropping/jitter, followed by repeating the last
    retained frame to restore the original frame count.
-   Random resized crop with sampled area between 60% and 100%.
-   Brightness multiplier between 0.8 and 1.2.
-   Contrast multiplier between 0.8 and 1.2.
-   Optional horizontal flip with probability 0.5.

The original temporal selection was:

``` python
drop = random.randint(1, 2)
keep = torch.randperm(T - drop)[:T - drop].sort().values
```

This selects indices only from the first `T - drop` frames and always
excludes the final frames. It is not unbiased random temporal sampling.

### Planned revised augmentation experiment

The next experiment is planned, not yet completed:

-   Random resized crop covering 85%--100% of image area.
-   Brightness: 0.85--1.15.
-   Contrast: 0.85--1.15.
-   Saturation: 0.8--1.2.
-   Gaussian noise with standard deviation between 0.005 and 0.02.
-   No temporal frame dropping initially.
-   Horizontal flip disabled initially.
-   Apply the same sampled color/noise parameters across every frame in
    a clip to avoid artificial flicker.
-   Validation and test clips remain unaugmented.

## 4. Training configuration

  Setting                                   Value
  ------------------------------- ---------------
  Backbone                          MViT V2 Small
  Classes                                       2
  Frames per clip                              16
  Resolution                            224 × 224
  Batch size                                    4
  Maximum epochs                               15
  Initial backbone LR                        1e-5
  Initial head LR                            1e-4
  Weight decay                               0.05
  Optimizer                                 AdamW
  Seed in supplied dataset code                44
  DataLoader workers                            0
  Train shuffle                              True
  Validation/test shuffle                   False

The reported training procedure initially freezes the backbone and later
unfreezes it. In the MLP experiment, the log reports `Backbone unfrozen`
after epoch 3, so epoch 4 is the first epoch after unfreezing. Learning
rates decrease during training; the exact scheduler configuration was
not provided.

## 5. Experiment 1 --- MViT V2 Small + MLP head

### Epoch results

  -------------------------------------------------------------------------------------------
    Epoch    Train Val Loss    Train  Val Acc    Train      Val  Val AUC   Val Macro      Val
              Loss               Acc          Macro F1 Macro F1            Precision    Macro
                                                                                       Recall
  ------- -------- -------- -------- -------- -------- -------- -------- ----------- --------
        1   0.6795   0.6466   0.5665   0.6272   0.5659   0.5945   0.7377      0.6836   0.6256

        2   0.6474   0.7111   0.6350   0.5621   0.6350   0.4683   0.7513      0.6963   0.5597

        3   0.6395   0.6153   0.6515   0.6509   0.6511   0.6379   0.7556      0.6784   0.6520

        4   0.6292   0.5932   0.6819   0.7337   0.6818   0.7307   0.7660      0.7436   0.7331

        5   0.6013   0.5853   0.6882   0.7041   0.6882   0.7040   0.7606      0.7042   0.7041

        6   0.5745   0.5922   0.7326   0.6686   0.7325   0.6681   0.7420      0.6694   0.6684

        7   0.5403   0.6293   0.7529   0.6923   0.7529   0.6785   0.7573      0.7293   0.6911
  -------------------------------------------------------------------------------------------

Early stopping occurred after epoch 7.

### Best reported validation checkpoint

-   Best validation AUC: **0.7660** (epoch 4).
-   Validation loss at epoch 4: **0.5932**.
-   Validation Macro F1 at epoch 4: **0.7307**.
-   Checkpoint: `nexar_data\best_mvit_collision.pth`.

The log reports checkpoint saves through epoch 4, but not epochs 5--7.
The exact checkpoint-selection condition should be confirmed in the
training code. Note that validation loss was numerically lowest at epoch
5 (0.5853), while the best reported AUC and Macro F1 were at epoch 4.

### Interpretation

Validation Macro F1 peaked at 0.7307 in epoch 4. Training Macro F1
continued increasing through epoch 7, while validation metrics
fluctuated. This can indicate overfitting or diminishing generalization
after epoch 4.

## 6. Earlier experiment --- MViT V2 Small + linear head

The earlier model replaced only the final classifier layer with a single
linear layer:

``` python
class NexarCollisionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = mvit_v2_s(weights=weights)

        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
```

### Epoch results

  -----------------------------------------------------------------------
     Epoch    Train Val Loss    Train  Val Acc    Train      Val  Val AUC
               Loss               Acc          Macro F1 Macro F1 
  -------- -------- -------- -------- -------- -------- -------- --------
         1   0.7632   0.6524   0.5146   0.6391   0.5146   0.6153   0.6982

         2   0.7048   0.6314   0.5691   0.6568   0.5686   0.6491   0.7265

         3   0.6713   0.5920   0.6198   0.6982   0.6196   0.6982   0.7584

         4   0.6328   0.5835   0.6679   0.6864   0.6678   0.6828   0.7625

         5   0.5914   0.5981   0.6958   0.6568   0.6958   0.6567   0.7368

         6   0.5819   0.6031   0.7123   0.6627   0.7123   0.6579   0.7457

         7   0.5550   0.5886   0.7414   0.6982   0.7414   0.6955   0.7605
  -----------------------------------------------------------------------

Reported best values:

-   Best validation Macro F1: **0.6828** (epoch 4).
-   Best validation AUC: **0.7625** (epoch 4).
-   Best validation loss: **0.5835** (epoch 4).

The earlier run froze the backbone for epochs 1--2 and unfroze it at
epoch 3.

## 7. Validation comparison: linear head vs MLP head

  Metric                            Linear head     MLP head
  ------------------------------- ------------- ------------
  Best validation Macro F1               0.6828   **0.7307**
  Best validation ROC-AUC                0.7625   **0.7660**
  Best validation loss               **0.5835**       0.5932
  Best epoch by reported F1/AUC               4            4

The MLP experiment improved validation Macro F1 by 0.0479 (4.79
percentage points) and AUC by 0.0035. Its best validation loss was
0.0097 higher. These are separate training runs; unless split, seed,
preprocessing, augmentation, and all training settings were identical,
treat this as suggestive rather than a perfectly controlled architecture
ablation.

## 8. Test-set results previously reported

The reported test set contained **170 videos: 85 positive and 85
negative**.

### Earlier evaluation (before explicit checkpoint loading was shown)

  Metric              Result
  ----------------- --------
  Accuracy            0.7353
  Macro F1            0.7319
  Macro precision     0.7477
  Macro recall        0.7353
  ROC-AUC             0.7942

Confusion matrix (rows = actual, columns = predicted):

``` text
                 Predicted
                 0    1
Actual 0        72   13
Actual 1        32   53
```

Collision class metrics: precision **0.8030**, recall **0.6235**, F1
**0.7020**.

Reported confidence intervals: ROC-AUC **\[0.7205, 0.8612\]**; F1
**\[0.6565, 0.7966\]**.

### Evaluation after explicitly loading `best_mvit_collision.pth`

The user showed this checkpoint-loading procedure:

``` python
checkpoint = torch.load(
    "nexar_data/best_mvit_collision.pth",
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Class-1 probability was calculated with softmax, and predictions used
threshold 0.50.

  Metric                   Result
  --------------------- ---------
  Test samples                170
  Positive / negative     85 / 85
  Threshold                  0.50
  Accuracy                 0.7118
  Macro F1                 0.7064
  Macro precision          0.7285
  Macro recall             0.7118
  ROC-AUC                  0.7855

Confusion matrix:

``` text
                 Predicted
                 0    1
Actual 0        72   13
Actual 1        36   49
```

  Class               Precision   Recall       F1
  ----------------- ----------- -------- --------
  0 --- Negative         0.6667   0.8471   0.7461
  1 --- Collision        0.7903   0.5765   0.6667

These two evaluations differ. The second is explicitly associated with
loading the saved checkpoint. Do not merge or average the two result
sets; the cause of the difference was not established in the supplied
information.

## 9. Exploratory test threshold sweep

The following threshold comparison was reported on the test set:

    Threshold     Macro F1   Collision precision   Collision recall
  ----------- ------------ --------------------- ------------------
         0.25       0.5888                0.5769             0.8824
         0.30       0.6585                0.6239             0.8588
         0.35       0.7207                0.6863             0.8235
         0.40   **0.7351**                0.7222             0.7647
         0.45       0.7049                0.7333             0.6471
         0.50       0.7064                0.7903             0.5765
         0.55       0.6990                0.8600             0.5059
         0.60       0.6544                0.8571             0.4235
         0.65       0.5895                0.8438             0.3176
         0.70       0.5787                0.9231             0.2824
         0.75       0.4799                0.9286             0.1529

The best observed test Macro F1 in this sweep was **0.7351 at threshold
0.40**. This is exploratory only. Selecting a threshold based on test
labels leaks test information into model selection. Choose the final
threshold on validation data, then freeze it before the final test
evaluation.

## 10. Evaluation and experiment-integrity notes

1.  Load the intended checkpoint before evaluation; do not evaluate
    whichever weights remain in memory after the final epoch.
2.  Keep `test_loader` at `shuffle=False`.
3.  Ensure saved `video_id` values are aligned with the actual
    evaluation order when exporting predictions to CSV.
4.  Select the classification threshold on validation data, not the test
    set.
5.  If multiple clips come from one source video, use a group-based
    split by source video to avoid source-video leakage across train,
    validation, and test.
6.  The reported test set is balanced. Metrics such as precision and
    accuracy can change under a different real-world class prevalence.
7.  Training metrics may be computed with augmentation/dropout active
    while validation is computed in evaluation mode; these metrics are
    not always directly comparable.
8.  State clearly whether the model classifies a collision using
    post-event evidence or predicts an impending collision using
    pre-event evidence.

## 11. Next planned experiment --- revised video augmentation

**Goal:** Keep the MViT + MLP architecture fixed and test whether
revised, temporally consistent augmentation improves generalization.

Keep fixed: - MViT V2 Small pretrained backbone. - MLP head:
`in_features → 512 → 256 → 2`. - Batch size 4. - 16 frames at 224 ×
224. - Optimizer, learning rates, and weight decay. - Same data splits
and evaluation procedure.

Change only training augmentation: - Crop area: 85%--100%. - Brightness:
0.85--1.15. - Contrast: 0.85--1.15. - Saturation: 0.8--1.2. - Gaussian
noise standard deviation: 0.005--0.02. - Disable temporal frame dropping
initially. - Disable horizontal flip initially. - Apply consistent
random parameters across frames in each clip. - No augmentation on
validation or test data.

Baseline for this next experiment:

  Metric                       Current MLP baseline
  -------------------------- ----------------------
  Best validation Macro F1                   0.7307
  Best validation ROC-AUC                    0.7660
  Best validation loss                       0.5932
  Best epoch                                      4

**Status:** Planned; no results have been reported for this augmentation
experiment yet.

## 12. Current conclusion

The strongest validation Macro F1 reported so far is **0.7307**, from
MViT V2 Small with the custom two-hidden-layer MLP head. Its best
reported validation AUC is **0.7660**. The MLP head outperformed the
earlier linear head on validation Macro F1, although its best validation
loss was slightly higher.

The next controlled step is to keep the model architecture unchanged and
test the revised video augmentation pipeline. The checkpoint-loaded test
evaluation reported Macro F1 **0.7064** and ROC-AUC **0.7855** at
threshold 0.50; keep it distinct from the earlier test evaluation and
the exploratory threshold sweep.
