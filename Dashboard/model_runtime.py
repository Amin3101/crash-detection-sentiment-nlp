"""Actual MViT notebook architecture + deterministic inference preprocessing.

Architecture is transcribed from `sources/final.ipynb` cell 7 and matches
`main.ipynb` cell 8 for the multi-scale temporal head. No fake scores.

Weights MUST correspond to this architecture. Separate checkpoints per task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Lazy loaded by the Streamlit page; torch is an OPTIONAL dependency.
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

class MultiScaleTemporalPool(nn.Module):

    def __init__(self, feature_dim=768, num_heads=8, num_layers=2, dropout=0.2, num_frames=16):
        super().__init__()

        # spatial feature projection
        # average pooling + max pooling
        self.spatial_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # learning temporal 
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # temporal transform
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_norm = nn.LayerNorm(feature_dim)


        # multy temporal cnn
        branch_dim = feature_dim // 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(feature_dim, branch_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_dim),
            nn.GELU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(feature_dim, branch_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_dim),
            nn.GELU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(feature_dim, branch_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_dim),
            nn.GELU()
        )

        self.multiscale_fusion = nn.Sequential(
            nn.Conv1d(branch_dim * 3, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # attention
        self.transformer_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.multiscale_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # fuse transform + cnn representations
        self.attention_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def attention_pool(self, x, attention):
        weights = torch.softmax(attention(x), dim=1)
        return torch.sum(x * weights, dim=1)

    def forward(self, patch_tokens):

        # Input: [B, T, H, W, C]
        B, T, H, W, C = patch_tokens.shape

        avg_features = patch_tokens.mean(dim=(2, 3))
        max_features = patch_tokens.amax(dim=(2, 3))

        x = torch.cat([avg_features, max_features], dim=-1)

        # [B, T, 2C] -> [B, T, C]
        x = self.spatial_projection(x)


        # transformer branch
        transformer_input = x + self.pos_embedding[:, :T, :]
        

        transformer_features = self.temporal_transformer(transformer_input)
        transformer_features = self.transformer_norm(transformer_features)
        transformer_pooled = self.attention_pool(transformer_features, self.transformer_attention)

        # cnn branch
        conv_input = x.transpose(1, 2)

        conv3 = self.conv3(conv_input)
        conv5 = self.conv5(conv_input)
        conv7 = self.conv7(conv_input)

        multiscale_features = torch.cat([conv3, conv5, conv7], dim=1)
        multiscale_features = self.multiscale_fusion(multiscale_features)
        multiscale_features = multiscale_features.transpose(1, 2)
        multiscale_pooled = self.attention_pool(multiscale_features, self.multiscale_attention)

    

        attention_features = self.attention_fusion(torch.cat([transformer_pooled,multiscale_pooled], dim=1))

        # temporla statistics
        temporal_mean = x.mean(dim=1)
        temporal_max = x.amax(dim=1)

        # three 768-dimensional vectors
        return attention_features, temporal_mean, temporal_max

class NexarCollisionModel(nn.Module):

    def __init__(self, num_classes=2, pretrained=True, dropout=0.4):
        super().__init__()

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        
        self.backbone = mvit_v2_s(weights=weights)

        in_features = self.backbone.head[1].in_features

        # removeing original classification head
        self.backbone.head = nn.Identity()

        # multi temporal pooling
        self.temporal_pool = MultiScaleTemporalPool(
            feature_dim=768,
            num_heads=8,
            num_layers=2,
            dropout=0.2,
            num_frames=16
        )

    
        fusion_dim = in_features * 4

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),

            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # Input: [B, C, T, H, W]

        x = self.backbone.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.backbone.pos_encoding(x)

        thw = (
            self.backbone.pos_encoding.temporal_size,
            *self.backbone.pos_encoding.spatial_size
        )

        # MViT transformer blocks
        for block in self.backbone.blocks:
            x, thw = block(x, thw)

        x = self.backbone.norm(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        B = x.shape[0]
        T, H, W = thw
        C = x.shape[-1]

        # [B, T*H*W, C] -> [B, T, H, W, C]
        patch_tokens = patch_tokens.reshape(B, T, H, W, C)

        
        attention_features, temporal_mean, temporal_max = self.temporal_pool(patch_tokens)

        # Feature fusion
        features = torch.cat(
            [
                cls_token,
                attention_features,
                temporal_mean,
                temporal_max
            ],
            dim=1
        )

        return self.classifier(features)

# No ImageNet/Kinetics weights fetched when instantiating the model.
def load_checkpoint(checkpoint: str | Path, device: str = "cpu"):
    """Load ONLY trusted checkpoints, strictly validating architecture keys."""
    path = Path(checkpoint).resolve()
    if not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=True avoids unpickling arbitrary Python objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected state_dict or checkpoint dictionary. Verify training export.")
    weights = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint does not contain a state dictionary.")
    if all(str(k).startswith("module.") for k in weights):
        weights = {k[7:]: v for k, v in weights.items()}
    if any(not isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("State dictionary includes non-tensor values; check the checkpoint format.")
    model = NexarCollisionModel(num_classes=2, pretrained=False)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as ex:
        raise RuntimeError("The checkpoint architecture differs from the notebook-derived MViT temporal model. "
                           "Do not use the wrong task checkpoint. Original error: " + str(ex)[:1100]) from ex
    model = model.to(torch.device(device)).eval()
    metadata = {k: payload.get(k) for k in ("epoch", "best_val_auc", "val_f1", "class_names") if k in payload}
    return model, metadata


def get_video_info(video_path: str | Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Unable to decode video. Convert to MP4/H.264 and retry.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames <= 0 or not np.isfinite(fps):
            raise ValueError("Video has invalid FPS or frame count.")
        return {"fps":fps,"frames":frames,"duration":frames/fps}
    finally:
        cap.release()


def _decode_frame_indices(cap, indices: np.ndarray, image_size: int = 224, resize_mode: str = "opencv") -> np.ndarray:
    """Output uint8 [T,H,W,C] in RGB; no random augmentation at inference."""
    import cv2
    from PIL import Image
    results=[]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok,frame=cap.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot read frame {int(idx)}. Try MP4/H.264.")
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if resize_mode=="pil":
            rgb=np.asarray(Image.fromarray(rgb).resize((image_size,image_size),Image.Resampling.BILINEAR))
        else:
            rgb=cv2.resize(rgb,(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        results.append(rgb)
    return np.stack(results).astype(np.uint8)


def sample_video(video_path: str | Path, mode: str, start: float | None = None,
                 duration: float = 4.0, num_frames: int = 16) -> np.ndarray:
    """Detection: 16 frames across whole video (final.ipynb).
    Anticipation: 16 pre-event frames in [start,start+4) (clipsampling.ipynb).
    """
    import cv2
    info=get_video_info(video_path)
    cap=cv2.VideoCapture(str(video_path))
    try:
        if mode=="detection":
            # Mirrors np.linspace(0, len(paths)-1, 16).astype(int) in final.ipynb.
            idx=np.linspace(0,info["frames"]-1,num_frames).astype(int)
            return _decode_frame_indices(cap,idx,resize_mode="pil")
        if mode!="anticipation" or start is None:
            raise ValueError("Anticipation requires an explicit window start time.")
        if start<0 or start+duration>info["duration"]+1e-6:
            raise ValueError("4-second window outside available video; padding disabled.")
        # Same endpoint=False and int(t*fps) as the project's clipsampling notebook.
        times=np.linspace(start,start+duration,num_frames,endpoint=False)
        idx=np.minimum((times*info["fps"]).astype(int),info["frames"]-1)
        return _decode_frame_indices(cap,idx,resize_mode="opencv")
    finally:
        cap.release()


def preprocess(clip: np.ndarray):
    """[T,H,W,C] uint8 -> [1,C,T,H,W] normalized like the training notebooks."""
    if clip.shape != (16,224,224,3) or clip.dtype != np.uint8:
        raise ValueError(f"Unexpected clip shape or dtype: {clip.shape}, {clip.dtype}")
    tensor=torch.from_numpy(clip.copy()).permute(3,0,1,2).float().div(255.0)
    mean=torch.tensor([0.45]*3).view(3,1,1,1)
    std=torch.tensor([0.225]*3).view(3,1,1,1)
    return ((tensor-mean)/std).unsqueeze(0)


@torch.inference_mode()
def predict_probability(model, clip: np.ndarray, device: str) -> float:
    batch=preprocess(clip).to(device)
    logits=model(batch)
    if logits.shape!=(1,2):
        raise ValueError(f"Expected two binary output logits, got {tuple(logits.shape)}")
    return float(torch.softmax(logits.float(),dim=1)[0,1].cpu().item())


def scan_windows(video_path: str | Path, window: float = 4.0, stride: float = 1.0):
    info=get_video_info(video_path)
    if info["duration"]<window:
        return []
    last=info["duration"]-window
    starts=list(np.arange(0,last+1e-9,stride,dtype=float))
    # Include the last complete window without duplicates.
    if not starts or last-starts[-1] > 1e-4:
        starts.append(float(last))
    return [(round(s,4),round(s+window,4)) for s in starts]
