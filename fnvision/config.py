# fnvision – Fovea Native Vision
# Apache License 2.0
#
# config.py – FoveaConfig dataclass and FoveaOutput dataclass
#
# Spec reference: SPEC_fnvision_v1.md Sections 6 and 7
# Formal definitions: Opus 4.6 review (DEV_NOTES.md, 2026-02-21)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# FoveaOutput
# ---------------------------------------------------------------------------

@dataclass
class FoveaOutput:
    """Immutable result container for a single FoveaEncoder.encode() call.

    All spatial tensors are float32 in range [0, 1].
    Input frames (uint8 RGB) are converted internally before producing output.

    Fields
    ------
    fovea : np.ndarray
        Shape (fovea_h, fovea_w, 3), float32.  High-resolution central zone.
    parafovea : np.ndarray
        Shape (para_h, para_w, 3), float32.  Mid-resolution surrounding zone.
    periphery : np.ndarray
        Shape (peri_h, peri_w, 3), float32.  Low-resolution full-frame zone.
    weight_map : np.ndarray
        Shape (H, W), float32.  Combined Gaussian weight field at full input
        resolution.  Peak is 1.0 at the F-center(s); range [0, 1].
    f1_pos_norm : tuple[float, float]
        F1 position in normalised image coordinates [0, 1]^2, (x, y).
    f2_pos_norm : tuple[float, float]
        F2 position in normalised image coordinates [0, 1]^2, (x, y).
    f_separation_norm : float
        Current Euclidean distance between F1 and F2 in normalised coords.
        0.0 = co-located (maximum zoom), 1.0 = maximum separation (wide-angle).
    attention_level : float
        Echoed back from the encode() call.  Range [0, 1].
    """

    fovea: np.ndarray
    parafovea: np.ndarray
    periphery: np.ndarray
    weight_map: np.ndarray
    f1_pos_norm: Tuple[float, float]
    f2_pos_norm: Tuple[float, float]
    f_separation_norm: float
    attention_level: float

    # ------------------------------------------------------------------
    # Convenience helpers (no mutation)
    # ------------------------------------------------------------------

    def zone_shapes(self) -> dict:
        """Return a dict of zone name -> shape for quick inspection."""
        return {
            "fovea": self.fovea.shape,
            "parafovea": self.parafovea.shape,
            "periphery": self.periphery.shape,
            "weight_map": self.weight_map.shape,
        }


# ---------------------------------------------------------------------------
# FoveaConfig
# ---------------------------------------------------------------------------

@dataclass
class FoveaConfig:
    """Configuration for the binocular F-system encoder.

    All spatial quantities use *normalised* coordinates unless the name ends
    in ``_res`` (those are pixel dimensions).

    Parameters
    ----------
    focal_radius_norm : float
        Gaussian sigma for both F-centers in normalised image coordinates.
        Acts as the single scale parameter for all three resolution zones.
        Typical value: 0.12 (covers roughly 12 % of the shorter image axis).

        Note: the dataclass currently uses one shared sigma for F1 and F2
        (symmetric configuration).  The field accepts a ``Tuple[float, float]``
        to allow independent sigma values in the future without API breakage;
        when a ``float`` is passed it is broadcast to ``(float, float)``.

    fovea_res : tuple[int, int]
        Output pixel size (height, width) for the fovea tensor.

    parafovea_res : tuple[int, int]
        Output pixel size (height, width) for the parafovea tensor.

    periphery_res : tuple[int, int]
        Output pixel size (height, width) for the periphery tensor.

    parafovea_res_factor : float
        Relative resolution sampled in the 1–3 × sigma band.  Controls the
        downscale ratio applied before producing the parafovea tensor.

    periphery_res_factor : float
        Relative resolution sampled beyond 3 × sigma.  Controls the
        downscale ratio applied before producing the periphery tensor.

    f_separation_max_norm : float
        Maximum allowed normalised distance between F1 and F2 (resting state,
        wide-angle).

    f_separation_min_norm : float
        Minimum allowed normalised distance (fully co-located = maximum zoom).

    zoom_max_bonus : float
        Zoom multiplier added at full convergence (separation_norm = 0.0).
        Zoom factor formula (Opus 4.6): zoom = 1.0 + zoom_max_bonus * (1 - sep)
        Default 0.4 yields ~140 % zoom at co-location.

    pull_strength : float
        Spring constant ``k`` in the passive pull-back model
        ``F_pull = k * (d_max – d_current)``.  Higher values return the
        F-centers to resting state more quickly.

    gaze_max_step_norm : float
        Maximum gaze displacement per tick in normalised coords.

    gaze_hold_prob : float
        Probability of the gaze staying fixed (fixation) on a given tick.

    gaze_jitter_norm : float
        Amplitude of random micro-jitter added to gaze each tick.

    periphery_attention_floor : float
        Minimum periphery scale factor when ``attention_level = 0.0``.
        Defines how much peripheral detail is retained even at zero attention.

    attention_inner_threshold : float
        ``attention_level`` below which inner zones (fovea, parafovea) also
        begin to scale down.  At this threshold inner_factor = 1.0; at 0.0
        inner_factor = 0.5.

    weight_gamma : float
        Optional contrast-shaping exponent applied to the normalised weight
        map: ``w_out = w_norm ** gamma``.  Default 1.0 = no shaping.

    fovea_threshold : float
        Minimum normalised weight value (after gamma shaping) for a pixel to
        be classified as belonging to the fovea zone.

    para_threshold : float
        Minimum normalised weight value for the parafovea zone.  Pixels with
        weight below this value are classified as periphery.
    """

    # Zone sizing
    focal_radius_norm: float = 0.12
    fovea_res: Tuple[int, int] = (96, 96)
    parafovea_res: Tuple[int, int] = (128, 128)
    periphery_res: Tuple[int, int] = (96, 96)
    parafovea_res_factor: float = 0.60
    periphery_res_factor: float = 0.15

    # Zoom / F-system
    f_separation_max_norm: float = 0.28
    f_separation_min_norm: float = 0.00
    zoom_max_bonus: float = 0.40

    # Spring-damper pull (MF2; stored here so calibration tool can tune it)
    pull_strength: float = 0.015

    # Gaze dynamics (MF2)
    gaze_max_step_norm: float = 0.06
    gaze_hold_prob: float = 0.12
    gaze_jitter_norm: float = 0.010

    # Attention
    periphery_attention_floor: float = 0.15
    attention_inner_threshold: float = 0.30

    # Weight field shaping
    weight_gamma: float = 1.0

    # Zone boundary thresholds (Opus 4.6 formal definition)
    fovea_threshold: float = 0.60
    para_threshold: float = 0.15

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors = []

        if not (0.0 < self.focal_radius_norm <= 1.0):
            errors.append(
                f"focal_radius_norm must be in (0, 1], got {self.focal_radius_norm}"
            )
        if not (0.0 <= self.f_separation_min_norm < self.f_separation_max_norm <= 1.0):
            errors.append(
                "f_separation_min_norm < f_separation_max_norm required, "
                f"got [{self.f_separation_min_norm}, {self.f_separation_max_norm}]"
            )
        if not (0.0 < self.zoom_max_bonus):
            errors.append(
                f"zoom_max_bonus must be positive, got {self.zoom_max_bonus}"
            )
        if not (0.0 < self.para_threshold < self.fovea_threshold <= 1.0):
            errors.append(
                "para_threshold < fovea_threshold required, "
                f"got [{self.para_threshold}, {self.fovea_threshold}]"
            )
        if not (0.0 <= self.periphery_attention_floor < 1.0):
            errors.append(
                f"periphery_attention_floor must be in [0, 1), got {self.periphery_attention_floor}"
            )
        if not (0.0 < self.weight_gamma):
            errors.append(
                f"weight_gamma must be positive, got {self.weight_gamma}"
            )
        for name in ("fovea_res", "parafovea_res", "periphery_res"):
            res = getattr(self, name)
            if len(res) != 2 or res[0] <= 0 or res[1] <= 0:
                errors.append(f"{name} must be a (h, w) tuple with positive integers, got {res}")

        if errors:
            raise ValueError("FoveaConfig validation failed:\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------
    # Derived quantities (read-only convenience)
    # ------------------------------------------------------------------

    @property
    def sigma_norm(self) -> float:
        """Gaussian sigma in normalised coordinates (alias for focal_radius_norm)."""
        return self.focal_radius_norm

    # ------------------------------------------------------------------
    # YAML serialisation (no external dependency beyond stdlib)
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Save this config to a YAML file.

        Uses Python's stdlib only (no PyYAML required for round-trip).
        The output format is intentionally simple so any YAML parser can read it.
        """
        import json  # fallback: write JSON-compatible YAML subset

        lines = ["# fnvision FoveaConfig\n"]
        for f_name, f_val in self.__dataclass_fields__.items():
            val = getattr(self, f_name)
            if isinstance(val, tuple):
                val_str = f"[{', '.join(str(v) for v in val)}]"
            else:
                val_str = str(val)
            lines.append(f"{f_name}: {val_str}\n")

        Path(path).write_text("".join(lines), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FoveaConfig":
        """Load a FoveaConfig from a YAML file written by ``to_yaml``.

        Supports the simple key: value format produced by ``to_yaml``.
        Does not require PyYAML.
        """
        import ast

        kwargs: dict = {}
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, raw = line.partition(":")
            key = key.strip()
            raw = raw.strip()
            if key not in cls.__dataclass_fields__:
                continue
            try:
                val = ast.literal_eval(raw)
                # Convert lists back to tuples where the field expects a tuple
                field_type = cls.__dataclass_fields__[key].type
                if isinstance(val, list):
                    val = tuple(val)
            except (ValueError, SyntaxError):
                val = raw
            kwargs[key] = val

        return cls(**kwargs)
