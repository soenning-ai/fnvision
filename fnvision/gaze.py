"""Stateful gaze controller for MF2 (Phase A + Phase B)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import FoveaConfig


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


@dataclass
class GazeState:
    """Mutable state for gaze dynamics."""

    gaze_xy: Tuple[float, float]
    f_separation_norm: float
    tick: int


class GazeController:
    """Stateful controller for gaze dynamics.

    Phase A (deterministic):
    - Saccade toward target with overshoot cap.
    - First-order pull of separation toward max separation.
    - Hard bounds so derived F1/F2 remain inside [0,1].

    Phase B (stochastic):
    - Hold-probability can skip the saccade update.
    - Additive jitter on gaze.
    - Both stochastic terms are disabled when dt == 0.
    """

    def __init__(
        self,
        config: FoveaConfig | None = None,
        rng: np.random.Generator | None = None,
        initial_gaze: Tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self._config = config or FoveaConfig()
        self._rng = rng if rng is not None else np.random.default_rng()
        gx = _clamp(float(initial_gaze[0]), 0.0, 1.0)
        gy = _clamp(float(initial_gaze[1]), 0.0, 1.0)
        self._state = GazeState(
            gaze_xy=(gx, gy),
            f_separation_norm=float(self._config.f_separation_max_norm),
            tick=0,
        )
        self._apply_bounds()

    @property
    def state(self) -> GazeState:
        return self._state

    def copy_state(self) -> GazeState:
        """Return a detached copy of current controller state."""
        return GazeState(
            gaze_xy=(float(self._state.gaze_xy[0]), float(self._state.gaze_xy[1])),
            f_separation_norm=float(self._state.f_separation_norm),
            tick=int(self._state.tick),
        )

    def snapshot(self) -> tuple[GazeState, dict]:
        """Return (state_copy, rng_state) for reproducible replay."""
        return self.copy_state(), copy.deepcopy(self._rng.bit_generator.state)

    def reset(
        self,
        state: GazeState,
        rng: np.random.Generator | None = None,
        rng_state: dict | None = None,
    ) -> None:
        """Replace controller state and optionally RNG/RNG-state."""
        if rng is not None:
            self._rng = rng
        if rng_state is not None:
            self._rng.bit_generator.state = copy.deepcopy(rng_state)
        self._state = GazeState(
            gaze_xy=(float(state.gaze_xy[0]), float(state.gaze_xy[1])),
            f_separation_norm=float(state.f_separation_norm),
            tick=int(state.tick),
        )
        self._apply_bounds()

    @property
    def f1_pos_norm(self) -> Tuple[float, float]:
        half = 0.5 * self._state.f_separation_norm
        return (
            float(self._state.gaze_xy[0] - half),
            float(self._state.gaze_xy[1]),
        )

    @property
    def f2_pos_norm(self) -> Tuple[float, float]:
        half = 0.5 * self._state.f_separation_norm
        return (
            float(self._state.gaze_xy[0] + half),
            float(self._state.gaze_xy[1]),
        )

    def step(
        self,
        target_xy: Tuple[float, float],
        attention_level: float = 1.0,
        dt: float = 1.0,
    ) -> GazeState:
        """Advance one tick.

        Parameters are clamped per A1/A3 decisions:
        - target_xy -> [0,1]^2
        - attention_level -> [0,1] (reserved for future coupling)
        - dt >= 0 (negative dt raises ValueError)
        """

        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")
        if dt < 0.0:
            raise ValueError("dt must be >= 0")

        tx = _clamp(float(target_xy[0]), 0.0, 1.0)
        ty = _clamp(float(target_xy[1]), 0.0, 1.0)
        _ = _clamp(float(attention_level), 0.0, 1.0)

        gaze = np.array(self._state.gaze_xy, dtype=np.float64)
        target = np.array((tx, ty), dtype=np.float64)

        if dt > 0.0:
            do_hold = self._rng.random() < float(self._config.gaze_hold_prob)
            if not do_hold:
                delta = target - gaze
                dist = float(np.linalg.norm(delta))
                if dist > 0.0:
                    max_step = float(self._config.gaze_max_step_norm) * dt
                    if dist <= max_step:
                        gaze = target
                    else:
                        gaze = gaze + (delta / dist) * max_step

        sep = float(self._state.f_separation_norm)
        sep = sep + float(self._config.pull_strength) * (
            float(self._config.f_separation_max_norm) - sep
        ) * dt
        sep = _clamp(
            sep,
            float(self._config.f_separation_min_norm),
            float(self._config.f_separation_max_norm),
        )

        if dt > 0.0 and float(self._config.gaze_jitter_norm) > 0.0:
            jitter = self._rng.normal(
                0.0, float(self._config.gaze_jitter_norm), size=2
            )
            gaze = gaze + jitter

        self._state.gaze_xy = (float(gaze[0]), float(gaze[1]))
        self._state.f_separation_norm = sep
        self._apply_bounds()
        self._state.tick += 1
        return self._state

    def _apply_bounds(self) -> None:
        """Clamp state so derived F1/F2 remain in [0,1]."""
        sep = _clamp(
            float(self._state.f_separation_norm),
            float(self._config.f_separation_min_norm),
            float(self._config.f_separation_max_norm),
        )
        half = 0.5 * sep
        gx = _clamp(float(self._state.gaze_xy[0]), half, 1.0 - half)
        gy = _clamp(float(self._state.gaze_xy[1]), 0.0, 1.0)
        self._state.gaze_xy = (gx, gy)
        self._state.f_separation_norm = sep
