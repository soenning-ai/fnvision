"""Gaussian dual-fovea weight-field utilities.

MF1 Step 2 (Codex scope):
- pure, vectorized NumPy algorithm
- deterministic
- no side effects
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _as_point_xy_norm(value: Sequence[float], name: str) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}")
    x = float(value[0])
    y = float(value[1])
    return x, y


def compute_weight_map(
    height: int,
    width: int,
    p1_xy_norm: Sequence[float],
    p2_xy_norm: Sequence[float],
    sigma_norm: float,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the normalized dual-Gaussian weight map w_out in [0, 1].

    Formal definition:
    - w1(x) = exp(-||x - p1||^2 / (2*sigma^2))
    - w2(x) = exp(-||x - p2||^2 / (2*sigma^2))
    - w_raw = w1 + w2
    - w_norm = w_raw / max(w_raw)
    - w_out = w_norm ** gamma

    Args:
        height: Output map height (H), must be > 0.
        width: Output map width (W), must be > 0.
        p1_xy_norm: F1 center as (x, y) in normalized coordinates.
        p2_xy_norm: F2 center as (x, y) in normalized coordinates.
        sigma_norm: Gaussian sigma in normalized coordinates.
        gamma: Optional shaping exponent, > 0, default 1.0.
        eps: Numerical floor used for sigma and denominator guards.

    Returns:
        np.ndarray[H, W], float32, values in [0, 1].
    """
    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError(f"height/width must be > 0, got ({height}, {width})")

    p1x, p1y = _as_point_xy_norm(p1_xy_norm, "p1_xy_norm")
    p2x, p2y = _as_point_xy_norm(p2_xy_norm, "p2_xy_norm")
    p1x = float(np.clip(p1x, 0.0, 1.0))
    p1y = float(np.clip(p1y, 0.0, 1.0))
    p2x = float(np.clip(p2x, 0.0, 1.0))
    p2y = float(np.clip(p2y, 0.0, 1.0))

    sigma = max(float(sigma_norm), float(eps))
    g = float(gamma)
    if g <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    x = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    d2_1 = (xx - p1x) * (xx - p1x) + (yy - p1y) * (yy - p1y)
    d2_2 = (xx - p2x) * (xx - p2x) + (yy - p2y) * (yy - p2y)

    w1 = np.exp(-d2_1 * inv_two_sigma2).astype(np.float32, copy=False)
    w2 = np.exp(-d2_2 * inv_two_sigma2).astype(np.float32, copy=False)
    w_raw = w1 + w2

    denom = max(float(np.max(w_raw)), float(eps))
    w_norm = w_raw / denom
    if g != 1.0:
        w_norm = np.power(w_norm, g).astype(np.float32, copy=False)

    return np.clip(w_norm, 0.0, 1.0).astype(np.float32, copy=False)
