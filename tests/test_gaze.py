"""Unit tests for MF2 gaze controller (Phase A + Phase B)."""

from __future__ import annotations

import numpy as np
import pytest

from fnvision import FoveaConfig, GazeController


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


class TestGazeControllerInit:
    def test_defaults(self):
        cfg = FoveaConfig()
        ctrl = GazeController(config=cfg)
        assert ctrl.state.tick == 0
        assert ctrl.state.f_separation_norm == pytest.approx(cfg.f_separation_max_norm)
        assert 0.0 <= ctrl.state.gaze_xy[0] <= 1.0
        assert 0.0 <= ctrl.state.gaze_xy[1] <= 1.0

    def test_initial_gaze_is_clamped(self):
        ctrl = GazeController(initial_gaze=(-1.0, 2.0))
        assert 0.0 <= ctrl.state.gaze_xy[0] <= 1.0
        assert 0.0 <= ctrl.state.gaze_xy[1] <= 1.0


class TestStepAndBounds:
    def test_tick_increments(self):
        ctrl = GazeController()
        ctrl.step((0.7, 0.4))
        assert ctrl.state.tick == 1
        ctrl.step((0.2, 0.6))
        assert ctrl.state.tick == 2

    def test_negative_dt_raises(self):
        ctrl = GazeController()
        with pytest.raises(ValueError, match="dt"):
            ctrl.step((0.6, 0.6), dt=-0.1)

    @pytest.mark.parametrize("bad_dt", [np.nan, np.inf, -np.inf])
    def test_nonfinite_dt_raises(self, bad_dt):
        ctrl = GazeController()
        with pytest.raises(ValueError, match="finite"):
            ctrl.step((0.6, 0.6), dt=bad_dt)

    def test_dt_zero_is_noop_except_tick(self):
        ctrl = GazeController(initial_gaze=(0.2, 0.8))
        before = (ctrl.state.gaze_xy, ctrl.state.f_separation_norm, ctrl.state.tick)
        ctrl.step((0.9, 0.1), dt=0.0)
        assert ctrl.state.gaze_xy == pytest.approx(before[0], abs=1e-9)
        assert ctrl.state.f_separation_norm == pytest.approx(before[1], abs=1e-9)
        assert ctrl.state.tick == before[2] + 1

    def test_target_clamped_to_unit_square(self):
        cfg = FoveaConfig(gaze_max_step_norm=10.0, gaze_hold_prob=0.0, gaze_jitter_norm=0.0)
        ctrl = GazeController(config=cfg, initial_gaze=(0.5, 0.5))
        ctrl.step((2.0, -3.0), dt=1.0)
        # With huge step cap, state should move to clamped target (1.0, 0.0),
        # then apply x-bound for current separation.
        assert ctrl.state.gaze_xy[1] == pytest.approx(0.0, abs=1e-9)
        assert ctrl.state.gaze_xy[0] <= 1.0

    def test_f_positions_are_always_in_bounds(self):
        cfg = FoveaConfig(f_separation_max_norm=0.8)
        ctrl = GazeController(config=cfg, initial_gaze=(0.0, 0.5))
        for target in [(0.0, 0.0), (1.0, 1.0), (0.2, 0.8), (0.8, 0.2)]:
            ctrl.step(target, dt=1.0)
            for p in (ctrl.f1_pos_norm, ctrl.f2_pos_norm):
                assert 0.0 <= p[0] <= 1.0
                assert 0.0 <= p[1] <= 1.0


class TestSaccadeAndSpring:
    def test_saccade_has_overshoot_cap(self):
        cfg = FoveaConfig(gaze_max_step_norm=0.06, gaze_hold_prob=0.0, gaze_jitter_norm=0.0)
        ctrl = GazeController(config=cfg, initial_gaze=(0.5, 0.5))
        before = ctrl.state.gaze_xy
        ctrl.step((1.0, 1.0), dt=1.0)
        moved = _dist(before, ctrl.state.gaze_xy)
        assert moved == pytest.approx(0.06, abs=1e-6)

    def test_saccade_does_not_overshoot_target(self):
        cfg = FoveaConfig(gaze_max_step_norm=0.50, gaze_hold_prob=0.0, gaze_jitter_norm=0.0)
        ctrl = GazeController(config=cfg, initial_gaze=(0.5, 0.5))
        target = (0.55, 0.50)
        ctrl.step(target, dt=1.0)
        assert ctrl.state.gaze_xy == pytest.approx(target, abs=1e-6)

    def test_pull_strength_zero_keeps_separation_constant(self):
        cfg = FoveaConfig(
            pull_strength=0.0,
            f_separation_min_norm=0.0,
            f_separation_max_norm=0.5,
        )
        ctrl = GazeController(config=cfg)
        ctrl.state.f_separation_norm = 0.2
        ctrl.step((0.6, 0.6), dt=1.0)
        assert ctrl.state.f_separation_norm == pytest.approx(0.2, abs=1e-9)

    def test_pull_moves_separation_toward_max(self):
        cfg = FoveaConfig(
            pull_strength=0.2,
            f_separation_min_norm=0.0,
            f_separation_max_norm=0.6,
        )
        ctrl = GazeController(config=cfg)
        ctrl.state.f_separation_norm = 0.1
        ctrl.step((0.5, 0.5), dt=1.0)
        assert ctrl.state.f_separation_norm > 0.1
        assert ctrl.state.f_separation_norm <= 0.6

    def test_separation_is_clamped_for_large_dt(self):
        cfg = FoveaConfig(
            pull_strength=0.5,
            f_separation_min_norm=0.1,
            f_separation_max_norm=0.4,
        )
        ctrl = GazeController(config=cfg)
        ctrl.state.f_separation_norm = 0.2
        ctrl.step((0.5, 0.5), dt=100.0)
        assert 0.1 <= ctrl.state.f_separation_norm <= 0.4
        assert ctrl.state.f_separation_norm == pytest.approx(0.4, abs=1e-9)


class TestDeterminismPhaseA:
    def test_different_rngs_same_result_in_phase_a(self):
        cfg = FoveaConfig(gaze_jitter_norm=0.0, gaze_hold_prob=0.0)
        a = GazeController(config=cfg, rng=np.random.default_rng(1), initial_gaze=(0.2, 0.8))
        b = GazeController(config=cfg, rng=np.random.default_rng(999), initial_gaze=(0.2, 0.8))
        for target in [(0.8, 0.2), (0.6, 0.3), (0.4, 0.7)]:
            sa = a.step(target, attention_level=0.1, dt=1.0)
            sb = b.step(target, attention_level=0.1, dt=1.0)
            assert sa.gaze_xy == pytest.approx(sb.gaze_xy, abs=1e-9)
            assert sa.f_separation_norm == pytest.approx(sb.f_separation_norm, abs=1e-9)
            assert sa.tick == sb.tick


class TestStochasticPhaseB:
    def _target_at(self, i: int) -> tuple[float, float]:
        return (((i * 37) % 101) / 100.0, ((i * 13) % 97) / 96.0)

    def _state_tuple(self, ctrl: GazeController) -> tuple[float, float, float, int]:
        return (
            float(ctrl.state.gaze_xy[0]),
            float(ctrl.state.gaze_xy[1]),
            float(ctrl.state.f_separation_norm),
            int(ctrl.state.tick),
        )

    def _pose_tuple(self, ctrl: GazeController) -> tuple[float, float, float]:
        return (
            float(ctrl.state.gaze_xy[0]),
            float(ctrl.state.gaze_xy[1]),
            float(ctrl.state.f_separation_norm),
        )

    def test_seed_reproducibility_over_100_ticks(self):
        cfg = FoveaConfig(gaze_hold_prob=0.30, gaze_jitter_norm=0.01)
        a = GazeController(config=cfg, rng=np.random.default_rng(42), initial_gaze=(0.4, 0.6))
        b = GazeController(config=cfg, rng=np.random.default_rng(42), initial_gaze=(0.4, 0.6))
        for i in range(100):
            t = self._target_at(i)
            a.step(t, dt=1.0)
            b.step(t, dt=1.0)
            assert self._state_tuple(a) == pytest.approx(self._state_tuple(b), abs=1e-12)

    def test_hold_probability_effect(self):
        cfg_hold = FoveaConfig(
            gaze_hold_prob=1.0,
            gaze_jitter_norm=0.0,
            pull_strength=0.2,
            f_separation_min_norm=0.0,
            f_separation_max_norm=0.6,
        )
        c_hold = GazeController(config=cfg_hold, rng=np.random.default_rng(5), initial_gaze=(0.5, 0.5))
        c_hold.state.f_separation_norm = 0.1
        before_gaze = tuple(c_hold.state.gaze_xy)
        c_hold.step((0.95, 0.95), dt=1.0)
        assert c_hold.state.gaze_xy == pytest.approx(before_gaze, abs=1e-9)
        assert c_hold.state.f_separation_norm > 0.1

        cfg_move = FoveaConfig(gaze_hold_prob=0.0, gaze_jitter_norm=0.0, gaze_max_step_norm=0.2)
        c_move = GazeController(config=cfg_move, rng=np.random.default_rng(5), initial_gaze=(0.5, 0.5))
        c_move.step((0.95, 0.95), dt=1.0)
        assert _dist((0.5, 0.5), c_move.state.gaze_xy) > 0.0

    def test_jitter_bounds_fuzz_1000_ticks(self):
        cfg = FoveaConfig(gaze_hold_prob=0.25, gaze_jitter_norm=0.03, f_separation_max_norm=0.4)
        ctrl = GazeController(config=cfg, rng=np.random.default_rng(123), initial_gaze=(0.5, 0.5))
        for i in range(1000):
            t = self._target_at(i)
            ctrl.step(t, dt=1.0)
            for p in (ctrl.f1_pos_norm, ctrl.f2_pos_norm):
                assert 0.0 <= p[0] <= 1.0
                assert 0.0 <= p[1] <= 1.0

    def test_jitter_off_matches_phase_a_behavior(self):
        cfg = FoveaConfig(gaze_hold_prob=0.0, gaze_jitter_norm=0.0)
        a = GazeController(config=cfg, rng=np.random.default_rng(1), initial_gaze=(0.3, 0.3))
        b = GazeController(config=cfg, rng=np.random.default_rng(999), initial_gaze=(0.3, 0.3))
        for i in range(120):
            t = self._target_at(i)
            a.step(t, dt=1.0)
            b.step(t, dt=1.0)
            assert self._state_tuple(a) == pytest.approx(self._state_tuple(b), abs=1e-12)

    def test_dt_zero_consumes_no_rng_and_no_jitter(self):
        cfg = FoveaConfig(gaze_hold_prob=1.0, gaze_jitter_norm=0.05)
        a = GazeController(config=cfg, rng=np.random.default_rng(77), initial_gaze=(0.4, 0.4))
        b = GazeController(config=cfg, rng=np.random.default_rng(77), initial_gaze=(0.4, 0.4))
        # a: dt=0 step then dt=1
        a.step((0.9, 0.9), dt=0.0)
        a.step((0.9, 0.9), dt=1.0)
        # b: direct dt=1 (should match if dt=0 consumed no RNG)
        b.step((0.9, 0.9), dt=1.0)
        assert self._pose_tuple(a) == pytest.approx(self._pose_tuple(b), abs=1e-12)

    def test_replay_reproducibility_with_snapshot_reset(self):
        cfg = FoveaConfig(gaze_hold_prob=0.35, gaze_jitter_norm=0.02)
        source = GazeController(config=cfg, rng=np.random.default_rng(42), initial_gaze=(0.2, 0.8))
        for i in range(80):
            source.step(self._target_at(i), dt=1.0)
        state_at_80, rng_state_at_80 = source.snapshot()

        expected = []
        for i in range(80, 120):
            source.step(self._target_at(i), dt=1.0)
            expected.append(self._state_tuple(source))

        replay = GazeController(config=cfg, rng=np.random.default_rng(999), initial_gaze=(0.5, 0.5))
        replay.reset(state_at_80, rng_state=rng_state_at_80)
        observed = []
        for i in range(80, 120):
            replay.step(self._target_at(i), dt=1.0)
            observed.append(self._state_tuple(replay))

        for o, e in zip(observed, expected):
            assert o == pytest.approx(e, abs=1e-12)
