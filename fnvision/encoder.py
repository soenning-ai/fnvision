# fnvision â€“ Fovea Native Vision
# Apache License 2.0
#
# encoder.py â€“ FoveaEncoder: stateless binocular fovea encoder (MF1)
#
# Spec reference:   SPEC_fnvision_v1.md Sections 3, 6, 7, 9
# Formal math:      Opus 4.6 review (DEV_NOTES.md, 2026-02-21)
# Zone sampling:    _sample_zones() â€“ refined by Codex in Step 5

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import FoveaConfig, FoveaOutput
from .weight_field import compute_weight_map


class FoveaEncoder:
    """Binocular fovea encoder â€“ stateless MF1 implementation.

    Encodes a single RGB frame into three resolution zones (fovea, parafovea,
    periphery) driven by a dual-Gaussian weight field over two coupled foveal
    centers F1 and F2.

    MF1 scope â€“ stateless encode() only:
        The caller fully controls gaze position and F-center separation.
        Stateful gaze dynamics (spring-damper pull, saccades, jitter) are
        added in MF2 via gaze.py.

    API decision (Codex Addendum, Section B1):
        Option 1 applies for MF1 â€“ encode() is fully stateless.
        Each call is independent; no internal state is mutated.

    Usage::

        from fnvision import FoveaConfig, FoveaEncoder

        cfg = FoveaConfig(focal_radius_norm=0.12)
        encoder = FoveaEncoder(cfg)

        result = encoder.encode(
            frame_rgb=frame,
            gaze_xy=(0.5, 0.5),
            f_separation=1.0,
            attention_level=1.0,
        )
        print(result.fovea.shape)     # (96, 96, 3)  float32
        print(result.weight_map.shape) # (H, W)       float32
    """

    def __init__(self, config: FoveaConfig | None = None) -> None:
        """
        Args:
            config: FoveaConfig instance.  Defaults to FoveaConfig() (all
                    parameters at their spec-defined defaults).
        """
        self.config: FoveaConfig = config if config is not None else FoveaConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        frame_rgb: np.ndarray,
        gaze_xy: Tuple[float, float] = (0.5, 0.5),
        f_separation: float = 1.0,
        attention_level: float = 1.0,
    ) -> FoveaOutput:
        """Encode one frame through the binocular fovea model.

        Args:
            frame_rgb: Input frame, uint8 RGB, shape (H, W, 3).
                       H and W must each be >= 1.
            gaze_xy: Gaze center (x, y) in normalised image coordinates
                     [0, 1]^2.  (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right.
                     Clamped to [0, 1] if out of range.
            f_separation: F1/F2 separation as a fraction of the maximum allowed
                          separation (``FoveaConfig.f_separation_max_norm``).
                          1.0 = resting wide-angle, 0.0 = fully co-located (~140% zoom).
                          Clamped to [0, 1] if out of range.
            attention_level: External attention signal in [0, 1].
                             1.0 = full attention (no tunnel effect).
                             Clamped to [0, 1] if out of range.

        Returns:
            FoveaOutput with all spatial tensors as float32 in [0, 1].

        Raises:
            ValueError: If frame_rgb does not have shape (H, W, 3).
            TypeError:  If frame_rgb is not uint8.
        """
        cfg = self.config

        # --- Input validation ------------------------------------------------
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(
                f"frame_rgb must have shape (H, W, 3), got {frame_rgb.shape}"
            )
        if frame_rgb.dtype != np.uint8:
            raise TypeError(
                f"frame_rgb must be dtype uint8, got {frame_rgb.dtype}"
            )

        H, W = frame_rgb.shape[:2]
        if H < 1 or W < 1:
            raise ValueError(
                f"frame_rgb spatial dimensions must be >= 1, got ({H}, {W})"
            )

        # Clamp caller inputs to valid ranges
        gx = float(np.clip(gaze_xy[0], 0.0, 1.0))
        gy = float(np.clip(gaze_xy[1], 0.0, 1.0))
        f_sep = float(np.clip(f_separation, 0.0, 1.0))
        att = float(np.clip(attention_level, 0.0, 1.0))

        # --- uint8 â†’ float32 -------------------------------------------------
        frame_f32: np.ndarray = frame_rgb.astype(np.float32) * (1.0 / 255.0)

        # --- F1/F2 positions -------------------------------------------------
        f1_pos, f2_pos, sep_norm = self._compute_f_positions(gx, gy, f_sep)

        # --- Gaussian weight map ---------------------------------------------
        weight_map: np.ndarray = compute_weight_map(
            height=H,
            width=W,
            p1_xy_norm=f1_pos,
            p2_xy_norm=f2_pos,
            sigma_norm=cfg.sigma_norm,
            gamma=cfg.weight_gamma,
        )

        # --- Attention scale factors -----------------------------------------
        peri_factor, inner_factor = self._attention_factors(att)

        # --- Zoom factor (Opus 4.6 formal definition) ------------------------
        # zoom = 1.0 + zoom_max_bonus * (1 - sep)
        # sep=1.0 â†’ zoom=1.0 (wide-angle), sep=0.0 â†’ zoom=1+bonus (~1.4)
        zoom: float = 1.0 + cfg.zoom_max_bonus * (1.0 - f_sep)

        # --- Zone sampling (refined by Codex in Step 5) ----------------------
        fovea, parafovea, periphery = self._sample_zones(
            frame_f32=frame_f32,
            weight_map=weight_map,
            gx=gx,
            gy=gy,
            zoom=zoom,
            peri_factor=peri_factor,
            inner_factor=inner_factor,
            H=H,
            W=W,
        )

        return FoveaOutput(
            fovea=fovea,
            parafovea=parafovea,
            periphery=periphery,
            weight_map=weight_map,
            f1_pos_norm=f1_pos,
            f2_pos_norm=f2_pos,
            f_separation_norm=sep_norm,
            attention_level=att,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_f_positions(
        self,
        gx: float,
        gy: float,
        f_separation: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Compute F1/F2 positions and actual separation from gaze + f_separation.

        F1 and F2 are placed symmetrically about gaze_xy along the x-axis
        (horizontal binocular axis, MF1 default).  Asymmetric placement
        (including vertical offset) is deferred to MF2 gaze dynamics.

        The weighted centroid of F1/F2 equals gaze_xy when both centers share
        the same sigma (symmetric MF1 configuration).  Per Opus 4.6 correction:
        the crop center is the weighted centroid, not the geometric midpoint â€“
        these are identical in the symmetric case, but the formulation is
        already correct for the asymmetric MF2 extension.

        Args:
            gx, gy:       Gaze center in normalised coords, already clamped.
            f_separation: Separation fraction in [0, 1].

        Returns:
            (f1_pos_norm, f2_pos_norm, sep_norm)
            sep_norm is the actual separation distance in normalised coords.
        """
        cfg = self.config

        # Actual separation in normalised coords
        sep_actual = f_separation * cfg.f_separation_max_norm
        sep_actual = float(
            np.clip(sep_actual, cfg.f_separation_min_norm, cfg.f_separation_max_norm)
        )

        half = sep_actual / 2.0
        f1 = (float(np.clip(gx - half, 0.0, 1.0)), gy)
        f2 = (float(np.clip(gx + half, 0.0, 1.0)), gy)

        return f1, f2, sep_actual

    def _attention_factors(
        self, attention_level: float
    ) -> Tuple[float, float]:
        """Compute zone resolution scale factors from attention_level.

        Implements the formal attention model from Opus 4.6 review:

            peri_factor = floor + (1 - floor) * attention
            inner_factor = 1.0                          if attention >= threshold
            inner_factor = 0.5 + 0.5*(attention/thresh) if attention < threshold

        Args:
            attention_level: Clamped attention in [0, 1].

        Returns:
            peri_factor:  Periphery scale in [floor, 1.0].
            inner_factor: Inner-zone (fovea/parafovea) scale in [0.5, 1.0].
        """
        cfg = self.config
        floor = cfg.periphery_attention_floor
        thresh = cfg.attention_inner_threshold

        peri_factor = floor + (1.0 - floor) * attention_level

        if attention_level >= thresh:
            inner_factor = 1.0
        else:
            inner_factor = 0.5 + 0.5 * (attention_level / max(thresh, 1e-9))

        return float(peri_factor), float(inner_factor)

    def _sample_zones(
        self,
        frame_f32: np.ndarray,
        weight_map: np.ndarray,
        gx: float,
        gy: float,
        zoom: float,
        peri_factor: float,
        inner_factor: float,
        H: int,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract fovea, parafovea, and periphery tensors from the frame.

        Step 5 implementation:
        - weight-driven zone masks with smoothstep transitions
        - compositing invariant m_f + m_p + m_r = 1 at each pixel
        - full-frame blend before zone-specific crops/resizes
        """
        cfg = self.config

        # Crop centre in pixel coordinates (weighted centroid = gaze for MF1).
        cx = int(np.clip(round(gx * (W - 1)), 0, W - 1))
        cy = int(np.clip(round(gy * (H - 1)), 0, H - 1))

        # Build multi-resolution full-frame representations first.
        effective_para = float(np.clip(cfg.parafovea_res_factor * inner_factor, 0.01, 1.0))
        effective_peri = float(np.clip(cfg.periphery_res_factor * peri_factor, 0.01, 1.0))

        img_full = frame_f32.astype(np.float32, copy=False)
        img_para = _resize_f32(_apply_res_factor(img_full, effective_para), (H, W))
        img_peri = _resize_f32(_apply_res_factor(img_full, effective_peri), (H, W))

        # Soft masks from weight thresholds (Opus formalization).
        w = weight_map.astype(np.float32, copy=False)
        bw_f = max(0.01, 0.08 * cfg.fovea_threshold)
        bw_p = max(0.01, 0.08 * cfg.para_threshold)

        m_f = _smoothstep(cfg.fovea_threshold - bw_f, cfg.fovea_threshold + bw_f, w)
        para_lo = _smoothstep(cfg.para_threshold - bw_p, cfg.para_threshold + bw_p, w)
        para_hi = 1.0 - m_f
        m_p = np.clip(para_lo * para_hi, 0.0, 1.0)
        m_r = np.clip(1.0 - m_f - m_p, 0.0, 1.0)

        # Enforce invariant m_f + m_p + m_r = 1.
        m_sum = m_f + m_p + m_r
        m_sum = np.where(m_sum > 1e-9, m_sum, 1.0).astype(np.float32, copy=False)
        m_f = (m_f / m_sum).astype(np.float32, copy=False)
        m_p = (m_p / m_sum).astype(np.float32, copy=False)
        m_r = (m_r / m_sum).astype(np.float32, copy=False)

        foveated_full = (
            m_f[..., None] * img_full
            + m_p[..., None] * img_para
            + m_r[..., None] * img_peri
        )
        foveated_full = np.clip(foveated_full, 0.0, 1.0).astype(np.float32, copy=False)

        # Fovea crop: 2*sigma around gaze, shrinks with zoom.
        z = max(float(zoom), 1e-6)
        fov_half_w = max(1, int(round(cfg.focal_radius_norm * 2.0 / z * W)))
        fov_half_h = max(1, int(round(cfg.focal_radius_norm * 2.0 / z * H)))
        fov_crop = _safe_crop(foveated_full, cy, cx, fov_half_h, fov_half_w)
        fovea = _resize_f32(fov_crop, cfg.fovea_res)

        # Parafovea crop: approximately up to 3*sigma.
        para_half_w = max(1, int(round(cfg.focal_radius_norm * 6.0 * W)))
        para_half_h = max(1, int(round(cfg.focal_radius_norm * 6.0 * H)))
        para_crop = _safe_crop(foveated_full, cy, cx, para_half_h, para_half_w)
        parafovea = _resize_f32(para_crop, cfg.parafovea_res)

        # Periphery output: global context from composited frame.
        periphery = _resize_f32(foveated_full, cfg.periphery_res)

        return fovea, parafovea, periphery

# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no encoder state)
# ---------------------------------------------------------------------------

def _safe_crop(
    frame: np.ndarray,
    cy: int,
    cx: int,
    half_h: int,
    half_w: int,
) -> np.ndarray:
    """Crop a (2*half_h) Ã— (2*half_w) region centred on (cy, cx).

    Regions outside the frame boundary are zero-padded (constant 0.0).
    This ensures the fovea crop never fails at image edges.
    """
    H, W = frame.shape[:2]

    # Source region (clamped)
    src_y0 = max(0, cy - half_h)
    src_y1 = min(H, cy + half_h)
    src_x0 = max(0, cx - half_w)
    src_x1 = min(W, cx + half_w)

    crop = frame[src_y0:src_y1, src_x0:src_x1]

    # Padding needed?
    pad_top = max(0, half_h - cy)
    pad_bot = max(0, (cy + half_h) - H)
    pad_lft = max(0, half_w - cx)
    pad_rgt = max(0, (cx + half_w) - W)

    if pad_top > 0 or pad_bot > 0 or pad_lft > 0 or pad_rgt > 0:
        # Edge-replicate rather than zero-pad: avoids artificial black halos
        # when gaze is near an image boundary or when the parafovea crop
        # window (6 * sigma) exceeds the frame dimensions.  Zero-padding would
        # violate the compositing invariant on constant-colour frames and
        # introduce non-physical dark regions in the encoder output.
        crop = np.pad(
            crop,
            ((pad_top, pad_bot), (pad_lft, pad_rgt), (0, 0)),
            mode="edge",
        )

    return crop.astype(np.float32, copy=False)


def _apply_res_factor(frame: np.ndarray, factor: float) -> np.ndarray:
    """Downsample frame by ``factor`` using area averaging (INTER_AREA).

    Args:
        frame:  float32 array, shape (H, W, 3) or (H, W).
        factor: Scale in (0, 1].  Values >= 1.0 are returned unchanged.

    Returns:
        float32 array at the reduced size.
    """
    factor = float(np.clip(factor, 0.01, 1.0))
    if factor >= 1.0 - 1e-6:
        return frame.astype(np.float32, copy=False)
    H, W = frame.shape[:2]
    new_h = max(1, int(round(H * factor)))
    new_w = max(1, int(round(W * factor)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32, copy=False)


def _resize_f32(frame: np.ndarray, target_res: Tuple[int, int]) -> np.ndarray:
    """Resize frame to (height, width) using bilinear interpolation.

    Note: target_res is (height, width); cv2.resize expects (width, height).

    Args:
        frame:      float32 array.
        target_res: (height, width) target pixel dimensions.

    Returns:
        float32 array clipped to [0, 1], shape (height, width, C).
    """
    th, tw = target_res
    if frame.shape[:2] == (th, tw):
        return np.clip(frame, 0.0, 1.0).astype(np.float32, copy=False)
    resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Vectorized smoothstep output in [0, 1]."""
    e0 = float(edge0)
    e1 = float(edge1)
    if e1 <= e0:
        return (x >= e0).astype(np.float32, copy=False)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0).astype(np.float32, copy=False)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)
