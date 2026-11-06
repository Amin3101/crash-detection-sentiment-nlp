Two distinct real MViT checkpoints are required:

  detection.pth: from final.ipynb, best_mvit_collision_from_jpg_frames.pth
  anticipation.pth: from main.ipynb, transformer + multi-scale CNN checkpoint

The original project ZIP contains neither checkpoint. Never copy an unrelated
state dictionary or rename a different architecture to make the status green.

The notebook main.ipynb reuses a filename also used for other experiments.
Verify the checkpoint before use (see inspect_checkpoints.py and README_FA.md).
