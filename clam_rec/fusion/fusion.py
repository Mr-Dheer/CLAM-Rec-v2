"""
Fusion strategies for combining CLIP text and image embeddings (RQ2).

All operate on aligned (n_items+1, D) text and image arrays and produce the
per-item multimodal embedding fed to the alignment / injection network.

  - concat  : [text || image]           -> dim 2D   (paper's default)
  - mean    : (text + image) / 2, L2     -> dim D    (image-missing => text only)
  - gating  : learned per-item gate g in [0,1]; g*text + (1-g)*image, L2 -> dim D
              (a small MLP on [text||image] predicts the gate)

concat/mean are parameter-free and can be precomputed. gating is a learnable
nn.Module trained inside Stage 1.
"""

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# static (parameter-free) fusions -> return a numpy (n, out_dim) array
# --------------------------------------------------------------------------- #
def fuse_concat(text: np.ndarray, image: np.ndarray) -> np.ndarray:
    return np.concatenate([text, image], axis=1).astype(np.float32)


def fuse_mean(text: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Element-wise mean, L2-normalized. If image half is zero (missing image),
    falls back to the text embedding so image-less items are not down-weighted."""
    out = np.zeros_like(text, dtype=np.float32)
    img_norm = np.linalg.norm(image, axis=1)
    for i in range(text.shape[0]):
        t, im = text[i], image[i]
        v = (t + im) / 2.0 if img_norm[i] > 1e-6 else t
        n = np.linalg.norm(v)
        out[i] = v / n if n > 1e-6 else v
    return out


def static_fuse(method: str, text: np.ndarray, image: np.ndarray) -> np.ndarray:
    if method == "concat":
        return fuse_concat(text, image)
    if method == "mean":
        return fuse_mean(text, image)
    if method == "text_only":
        # ablation: CLIP-TEXT only (drop the image half) — isolates whether the
        # benefit comes from the image or just from CLIP-text being > SBERT.
        return text.astype(np.float32)
    raise ValueError(f"static_fuse: '{method}' is not a static method (use gating module)")


def fused_dim(method: str, per_modality_dim: int) -> int:
    return 2 * per_modality_dim if method == "concat" else per_modality_dim


# --------------------------------------------------------------------------- #
# learned gating fusion (a trainable module)
# --------------------------------------------------------------------------- #
class GatingFusion(nn.Module):
    """Per-item learned gate over text vs image.

    Input: text (B, D), image (B, D). A small MLP on [text||image] predicts a
    scalar gate g; output = L2(g*text + (1-g)*image). For image-missing items
    (image half all-zero) the gate is forced to 1 (text only).
    Output dim = D.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        has_image = (image.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        g = self.gate(torch.cat([text, image], dim=-1))
        g = g * has_image + (1.0 - has_image)  # force g=1 when no image
        fused = g * text + (1.0 - g) * image
        return torch.nn.functional.normalize(fused, p=2, dim=-1)
