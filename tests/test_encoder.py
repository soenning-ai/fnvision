# fnvision – Fovea Native Vision
# Apache License 2.0
#
# tests/test_encoder.py – Unit tests for MF1 Core Encoder
#
# Coverage targets (Opus 4.6 review, Schritt 7 mandate):
#   1. Compositing invariant: m_f + m_p + m_r = 1  (proxy: constant-frame test)
#   2. Output dtypes: all zones + weight_map must be float32
#   3. Output ranges: all values in [0, 1]
#   4. Output shapes: per FoveaConfig defaults
#   5. Determinism: same input → bitwise-identical output
#   6. Convergence (f_sep=0.0): no NaN / Inf
#   7. Edge gaze positions: (0,0) and (1,1) – _safe_crop must pad correctly
#   8. Attention extremes: attention=0.0 and attention=1.0
#   9. Zoom effect: fovea crops a smaller region at high zoom
#  10. Aspect ratio: non-square input works correctly
#
# Additional tests (beyond Opus mandate):
#  11. FoveaConfig validation: invalid parameters raise ValueError
#  12. FoveaConfig YAML round-trip: save + reload is lossless within float32
#  13. encode() input guards: wrong dtype and wrong ndim raise errors
#  14. FoveaOutput metadata: f_separation_norm and attention_level echoed correctly
#  15. weight_map: peak near F-center, range [0,1], shape (H,W)
#  16. compute_weight_map: symmetry and convergence case

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fnvision import FoveaConfig, FoveaEncoder, FoveaOutput
from fnvision.weight_field import compute_weight_map

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FRAME_H, FRAME_W = 480, 640  # standard test frame size (4:3)


def _make_frame(h: int = FRAME_H, w: int = FRAME_W, value: int | None = None) -> np.ndarray:
    """Return a uint8 RGB frame.

    value=None → random noise (repeatable via seed).
    value=int  → solid colour frame (all pixels = value).
    """
    if value is not None:
        return np.full((h, w, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.fixture
def default_encoder() -> FoveaEncoder:
    return FoveaEncoder(FoveaConfig())


@pytest.fixture
def default_frame() -> np.ndarray:
    return _make_frame()


@pytest.fixture
def default_result(default_encoder, default_frame) -> FoveaOutput:
    return default_encoder.encode(default_frame)


# ---------------------------------------------------------------------------
# 1. Compositing invariant (proxy test via constant-colour frame)
# ---------------------------------------------------------------------------

class TestCompositingInvariant:
    """Proxy for the m_f + m_p + m_r = 1 invariant.

    If the invariant holds, encoding a solid-colour frame must return
    all three zone tensors with values equal to that colour.
    Constant images are unaffected by any linear resize/downsample:
    img_full = img_para = img_peri = c,  so foveated = (m_f+m_p+m_r)*c = c.
    """

    @pytest.mark.parametrize("color_value", [0, 128, 255])
    def test_solid_frame_preserved_center_gaze(self, color_value):
        frame = _make_frame(value=color_value)
        encoder = FoveaEncoder()
        result = encoder.encode(frame, gaze_xy=(0.5, 0.5), f_separation=1.0)
        expected = color_value / 255.0
        for zone_name in ("fovea", "parafovea", "periphery"):
            zone = getattr(result, zone_name)
            assert np.allclose(zone, expected, atol=1e-3), (
                f"Compositing invariant failed for {zone_name}: "
                f"color={color_value}, max_err={np.max(np.abs(zone - expected)):.4f}"
            )

    @pytest.mark.parametrize("gaze", [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7)])
    def test_solid_frame_preserved_various_gaze(self, gaze):
        frame = _make_frame(value=200)
        encoder = FoveaEncoder()
        result = encoder.encode(frame, gaze_xy=gaze, attention_level=1.0)
        expected = 200 / 255.0
        for zone_name in ("fovea", "parafovea", "periphery"):
            zone = getattr(result, zone_name)
            assert np.allclose(zone, expected, atol=1e-3), (
                f"Compositing invariant failed at gaze={gaze}, zone={zone_name}"
            )


# ---------------------------------------------------------------------------
# 2. Output dtypes
# ---------------------------------------------------------------------------

class TestOutputDtypes:
    def test_fovea_dtype(self, default_result):
        assert default_result.fovea.dtype == np.float32

    def test_parafovea_dtype(self, default_result):
        assert default_result.parafovea.dtype == np.float32

    def test_periphery_dtype(self, default_result):
        assert default_result.periphery.dtype == np.float32

    def test_weight_map_dtype(self, default_result):
        assert default_result.weight_map.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Output ranges: all values in [0, 1]
# ---------------------------------------------------------------------------

class TestOutputRanges:
    @pytest.mark.parametrize("zone", ["fovea", "parafovea", "periphery", "weight_map"])
    def test_range(self, default_result, zone):
        arr = getattr(default_result, zone)
        assert float(arr.min()) >= 0.0, f"{zone} min < 0: {arr.min()}"
        assert float(arr.max()) <= 1.0, f"{zone} max > 1: {arr.max()}"

    def test_range_random_noise_frame(self):
        encoder = FoveaEncoder()
        frame = _make_frame()
        result = encoder.encode(frame)
        for zone_name in ("fovea", "parafovea", "periphery", "weight_map"):
            arr = getattr(result, zone_name)
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 4. Output shapes (default config: fovea 96×96, para 128×128, peri 96×96)
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_fovea_shape(self, default_result):
        assert default_result.fovea.shape == (96, 96, 3)

    def test_parafovea_shape(self, default_result):
        assert default_result.parafovea.shape == (128, 128, 3)

    def test_periphery_shape(self, default_result):
        assert default_result.periphery.shape == (96, 96, 3)

    def test_weight_map_shape_matches_frame(self, default_frame):
        encoder = FoveaEncoder()
        result = encoder.encode(default_frame)
        assert result.weight_map.shape == (FRAME_H, FRAME_W)

    def test_custom_resolution(self):
        cfg = FoveaConfig(fovea_res=(64, 64), parafovea_res=(80, 80), periphery_res=(48, 48))
        encoder = FoveaEncoder(cfg)
        result = encoder.encode(_make_frame())
        assert result.fovea.shape == (64, 64, 3)
        assert result.parafovea.shape == (80, 80, 3)
        assert result.periphery.shape == (48, 48, 3)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self, default_encoder, default_frame):
        r1 = default_encoder.encode(default_frame, gaze_xy=(0.4, 0.6), f_separation=0.7)
        r2 = default_encoder.encode(default_frame, gaze_xy=(0.4, 0.6), f_separation=0.7)
        np.testing.assert_array_equal(r1.fovea, r2.fovea)
        np.testing.assert_array_equal(r1.parafovea, r2.parafovea)
        np.testing.assert_array_equal(r1.periphery, r2.periphery)
        np.testing.assert_array_equal(r1.weight_map, r2.weight_map)

    def test_encoder_is_stateless(self, default_encoder, default_frame):
        """Encoding a different frame and then the original must return the same result."""
        other_frame = _make_frame(value=50)
        r_before = default_encoder.encode(default_frame)
        _unused = default_encoder.encode(other_frame)
        r_after = default_encoder.encode(default_frame)
        np.testing.assert_array_equal(r_before.fovea, r_after.fovea)
        np.testing.assert_array_equal(r_before.weight_map, r_after.weight_map)


# ---------------------------------------------------------------------------
# 6. Convergence (f_separation=0.0): no NaN / Inf
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_no_nan_inf_at_full_convergence(self, default_encoder, default_frame):
        result = default_encoder.encode(default_frame, f_separation=0.0)
        for zone_name in ("fovea", "parafovea", "periphery", "weight_map"):
            arr = getattr(result, zone_name)
            assert not np.any(np.isnan(arr)), f"NaN in {zone_name} at convergence"
            assert not np.any(np.isinf(arr)), f"Inf in {zone_name} at convergence"

    def test_f_separation_norm_zero_at_convergence(self, default_encoder, default_frame):
        result = default_encoder.encode(default_frame, f_separation=0.0)
        assert result.f_separation_norm == pytest.approx(0.0, abs=1e-9)

    def test_f1_f2_collocated_at_convergence(self, default_encoder, default_frame):
        result = default_encoder.encode(
            default_frame, gaze_xy=(0.5, 0.5), f_separation=0.0
        )
        assert result.f1_pos_norm == pytest.approx(result.f2_pos_norm, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Edge gaze positions: (0,0) and (1,1) – padding must not crash
# ---------------------------------------------------------------------------

class TestEdgeGaze:
    @pytest.mark.parametrize("gaze", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
    def test_no_crash_at_corners(self, default_encoder, default_frame, gaze):
        result = default_encoder.encode(default_frame, gaze_xy=gaze)
        assert result.fovea.shape == (96, 96, 3)
        assert result.parafovea.shape == (128, 128, 3)

    @pytest.mark.parametrize("gaze", [(0.0, 0.0), (1.0, 1.0)])
    def test_no_nan_at_corners(self, default_encoder, default_frame, gaze):
        result = default_encoder.encode(default_frame, gaze_xy=gaze)
        for zone_name in ("fovea", "parafovea", "periphery"):
            assert not np.any(np.isnan(getattr(result, zone_name)))

    def test_out_of_range_gaze_clamped(self, default_encoder, default_frame):
        """Gaze values outside [0,1] must be silently clamped."""
        result_clamped = default_encoder.encode(default_frame, gaze_xy=(-0.5, 1.5))
        result_corner = default_encoder.encode(default_frame, gaze_xy=(0.0, 1.0))
        np.testing.assert_array_equal(result_clamped.fovea, result_corner.fovea)


# ---------------------------------------------------------------------------
# 8. Attention extremes: 0.0 and 1.0
# ---------------------------------------------------------------------------

class TestAttentionExtremes:
    def test_no_nan_at_zero_attention(self, default_encoder, default_frame):
        result = default_encoder.encode(default_frame, attention_level=0.0)
        for zone_name in ("fovea", "parafovea", "periphery"):
            assert not np.any(np.isnan(getattr(result, zone_name)))

    def test_no_nan_at_full_attention(self, default_encoder, default_frame):
        result = default_encoder.encode(default_frame, attention_level=1.0)
        for zone_name in ("fovea", "parafovea", "periphery"):
            assert not np.any(np.isnan(getattr(result, zone_name)))

    def test_attention_echoed_in_output(self, default_encoder, default_frame):
        r0 = default_encoder.encode(default_frame, attention_level=0.0)
        r1 = default_encoder.encode(default_frame, attention_level=1.0)
        assert r0.attention_level == pytest.approx(0.0)
        assert r1.attention_level == pytest.approx(1.0)

    def test_out_of_range_attention_clamped(self, default_encoder, default_frame):
        """Values outside [0,1] must be silently clamped."""
        r_neg = default_encoder.encode(default_frame, attention_level=-1.0)
        r_zero = default_encoder.encode(default_frame, attention_level=0.0)
        np.testing.assert_array_equal(r_neg.fovea, r_zero.fovea)


# ---------------------------------------------------------------------------
# 9. Zoom effect: fovea should capture a smaller region at high zoom
# ---------------------------------------------------------------------------

class TestZoomEffect:
    def test_fovea_separation_norm_wide_vs_zoomed(self, default_frame):
        """f_separation_norm is larger at f_separation=1.0 than 0.0."""
        encoder = FoveaEncoder()
        r_wide = encoder.encode(default_frame, f_separation=1.0)
        r_zoom = encoder.encode(default_frame, f_separation=0.0)
        assert r_wide.f_separation_norm > r_zoom.f_separation_norm

    def test_fovea_outputs_differ_at_different_zoom(self, default_frame):
        """Fovea output must differ between wide-angle and full-zoom."""
        encoder = FoveaEncoder()
        r_wide = encoder.encode(default_frame, gaze_xy=(0.5, 0.5), f_separation=1.0)
        r_zoom = encoder.encode(default_frame, gaze_xy=(0.5, 0.5), f_separation=0.0)
        # At different zoom levels the cropped region differs → outputs must differ
        assert not np.array_equal(r_wide.fovea, r_zoom.fovea)

    def test_f_separation_norm_wide_equals_max_norm(self, default_frame):
        """At f_separation=1.0, sep_norm should equal f_separation_max_norm."""
        cfg = FoveaConfig()
        encoder = FoveaEncoder(cfg)
        result = encoder.encode(default_frame, f_separation=1.0)
        assert result.f_separation_norm == pytest.approx(cfg.f_separation_max_norm, rel=1e-5)


# ---------------------------------------------------------------------------
# 10. Aspect ratio: non-square input
# ---------------------------------------------------------------------------

class TestAspectRatio:
    @pytest.mark.parametrize("h, w", [(720, 1280), (1080, 1920), (240, 320), (100, 100)])
    def test_non_square_shapes_correct(self, h, w):
        frame = _make_frame(h=h, w=w)
        encoder = FoveaEncoder()
        result = encoder.encode(frame)
        assert result.fovea.shape == (96, 96, 3)
        assert result.parafovea.shape == (128, 128, 3)
        assert result.periphery.shape == (96, 96, 3)
        assert result.weight_map.shape == (h, w)

    @pytest.mark.parametrize("h, w", [(720, 1280), (480, 640)])
    def test_non_square_no_nan(self, h, w):
        frame = _make_frame(h=h, w=w)
        encoder = FoveaEncoder()
        result = encoder.encode(frame)
        for zone_name in ("fovea", "parafovea", "periphery", "weight_map"):
            assert not np.any(np.isnan(getattr(result, zone_name)))


# ---------------------------------------------------------------------------
# 11. FoveaConfig validation
# ---------------------------------------------------------------------------

class TestFoveaConfigValidation:
    def test_invalid_focal_radius_zero(self):
        with pytest.raises(ValueError, match="focal_radius_norm"):
            FoveaConfig(focal_radius_norm=0.0)

    def test_invalid_focal_radius_too_large(self):
        with pytest.raises(ValueError, match="focal_radius_norm"):
            FoveaConfig(focal_radius_norm=1.5)

    def test_invalid_separation_order(self):
        with pytest.raises(ValueError, match="f_separation"):
            FoveaConfig(f_separation_min_norm=0.5, f_separation_max_norm=0.1)

    def test_invalid_threshold_order(self):
        with pytest.raises(ValueError, match="para_threshold"):
            FoveaConfig(para_threshold=0.8, fovea_threshold=0.6)

    def test_invalid_weight_gamma_zero(self):
        with pytest.raises(ValueError, match="weight_gamma"):
            FoveaConfig(weight_gamma=0.0)

    def test_invalid_resolution_tuple(self):
        with pytest.raises(ValueError, match="fovea_res"):
            FoveaConfig(fovea_res=(0, 96))

    def test_valid_config_no_error(self):
        cfg = FoveaConfig(focal_radius_norm=0.10, fovea_threshold=0.7, para_threshold=0.2)
        assert cfg.focal_radius_norm == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 12. FoveaConfig YAML round-trip
# ---------------------------------------------------------------------------

class TestConfigYamlRoundTrip:
    def test_default_config_round_trips(self):
        cfg_orig = FoveaConfig()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cfg_orig.to_yaml(tmp_path)
            cfg_loaded = FoveaConfig.from_yaml(tmp_path)
            assert cfg_loaded.focal_radius_norm == pytest.approx(cfg_orig.focal_radius_norm)
            assert cfg_loaded.fovea_res == cfg_orig.fovea_res
            assert cfg_loaded.pull_strength == pytest.approx(cfg_orig.pull_strength)
            assert cfg_loaded.fovea_threshold == pytest.approx(cfg_orig.fovea_threshold)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_custom_config_round_trips(self):
        cfg_orig = FoveaConfig(focal_radius_norm=0.15, zoom_max_bonus=0.3, weight_gamma=1.5)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cfg_orig.to_yaml(tmp_path)
            cfg_loaded = FoveaConfig.from_yaml(tmp_path)
            assert cfg_loaded.focal_radius_norm == pytest.approx(0.15, rel=1e-4)
            assert cfg_loaded.zoom_max_bonus == pytest.approx(0.3, rel=1e-4)
            assert cfg_loaded.weight_gamma == pytest.approx(1.5, rel=1e-4)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 13. encode() input guards
# ---------------------------------------------------------------------------

class TestEncodeInputGuards:
    def test_wrong_dtype_raises_type_error(self, default_encoder):
        frame_f32 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="uint8"):
            default_encoder.encode(frame_f32)

    def test_wrong_ndim_raises_value_error(self, default_encoder):
        frame_2d = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            default_encoder.encode(frame_2d)

    def test_wrong_channels_raises_value_error(self, default_encoder):
        frame_1ch = np.zeros((FRAME_H, FRAME_W, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            default_encoder.encode(frame_1ch)


# ---------------------------------------------------------------------------
# 14. FoveaOutput metadata: echo-back fields
# ---------------------------------------------------------------------------

class TestFoveaOutputMetadata:
    def test_attention_level_echoed(self, default_encoder, default_frame):
        for att in (0.0, 0.5, 1.0):
            result = default_encoder.encode(default_frame, attention_level=att)
            assert result.attention_level == pytest.approx(att)

    def test_f_separation_norm_range(self, default_encoder, default_frame):
        cfg = default_encoder.config
        for f_sep in (0.0, 0.5, 1.0):
            result = default_encoder.encode(default_frame, f_separation=f_sep)
            assert 0.0 <= result.f_separation_norm <= cfg.f_separation_max_norm + 1e-9

    def test_f_positions_within_unit_square(self, default_encoder, default_frame):
        result = default_encoder.encode(default_frame)
        for pos in (result.f1_pos_norm, result.f2_pos_norm):
            assert 0.0 <= pos[0] <= 1.0
            assert 0.0 <= pos[1] <= 1.0

    def test_zone_shapes_helper(self, default_result):
        shapes = default_result.zone_shapes()
        assert shapes["fovea"] == (96, 96, 3)
        assert shapes["parafovea"] == (128, 128, 3)
        assert shapes["weight_map"] == (FRAME_H, FRAME_W)


# ---------------------------------------------------------------------------
# 15. weight_map properties
# ---------------------------------------------------------------------------

class TestWeightMap:
    def test_weight_map_peak_near_center_gaze(self):
        encoder = FoveaEncoder()
        frame = _make_frame()
        result = encoder.encode(frame, gaze_xy=(0.5, 0.5), f_separation=0.0)
        # At convergence, F1==F2 at gaze center → weight peak at (0.5, 0.5)
        peak_y, peak_x = np.unravel_index(np.argmax(result.weight_map), result.weight_map.shape)
        peak_y_norm = peak_y / (FRAME_H - 1)
        peak_x_norm = peak_x / (FRAME_W - 1)
        assert abs(peak_y_norm - 0.5) < 0.05, f"Peak y off center: {peak_y_norm}"
        assert abs(peak_x_norm - 0.5) < 0.05, f"Peak x off center: {peak_x_norm}"

    def test_weight_map_max_is_one(self):
        encoder = FoveaEncoder()
        result = encoder.encode(_make_frame())
        assert float(result.weight_map.max()) == pytest.approx(1.0, abs=1e-5)

    def test_weight_map_min_is_nonneg(self):
        encoder = FoveaEncoder()
        result = encoder.encode(_make_frame())
        assert float(result.weight_map.min()) >= 0.0


# ---------------------------------------------------------------------------
# 16. compute_weight_map: symmetry and convergence
# ---------------------------------------------------------------------------

class TestComputeWeightMap:
    def test_symmetric_points_symmetric_map(self):
        """Symmetric F1/F2 around center → weight map symmetric about vertical axis."""
        w = compute_weight_map(
            height=64, width=64,
            p1_xy_norm=(0.3, 0.5),
            p2_xy_norm=(0.7, 0.5),
            sigma_norm=0.15,
        )
        # Left half should mirror right half
        left = w[:, :32]
        right = np.fliplr(w[:, 32:])
        np.testing.assert_allclose(left, right, atol=1e-5)

    def test_convergence_single_peak(self):
        """When F1==F2, result must equal a single Gaussian (no bimodal artefact)."""
        w_conv = compute_weight_map(
            height=64, width=64,
            p1_xy_norm=(0.5, 0.5),
            p2_xy_norm=(0.5, 0.5),
            sigma_norm=0.15,
        )
        # Single peak at center, max = 1.0
        peak_y, peak_x = np.unravel_index(np.argmax(w_conv), w_conv.shape)
        assert peak_y == pytest.approx(31, abs=2)
        assert peak_x == pytest.approx(31, abs=2)
        assert w_conv.max() == pytest.approx(1.0, abs=1e-5)

    def test_output_dtype_float32(self):
        w = compute_weight_map(10, 10, (0.5, 0.5), (0.5, 0.5), 0.1)
        assert w.dtype == np.float32

    def test_output_range(self):
        w = compute_weight_map(100, 100, (0.2, 0.3), (0.7, 0.6), 0.12)
        assert float(w.min()) >= 0.0
        assert float(w.max()) <= 1.0 + 1e-6

    def test_aspect_ratio_non_square_shape(self):
        """Output shape must match (height, width) regardless of aspect ratio."""
        w = compute_weight_map(height=120, width=320, p1_xy_norm=(0.3, 0.5),
                               p2_xy_norm=(0.7, 0.5), sigma_norm=0.12)
        assert w.shape == (120, 320)

    def test_no_nan_small_sigma(self):
        """Extreme sigma near epsilon must not produce NaN."""
        w = compute_weight_map(32, 32, (0.5, 0.5), (0.5, 0.5), sigma_norm=1e-7)
        assert not np.any(np.isnan(w))
        assert not np.any(np.isinf(w))

    def test_gamma_shaping_monotone(self):
        """gamma > 1 must reduce values (compress toward 0) except at peak."""
        w1 = compute_weight_map(64, 64, (0.5, 0.5), (0.5, 0.5), 0.15, gamma=1.0)
        w2 = compute_weight_map(64, 64, (0.5, 0.5), (0.5, 0.5), 0.15, gamma=2.0)
        # Peak stays at 1.0 for both; non-peak values must be <= w1
        off_peak = (w1 < 0.99)
        assert np.all(w2[off_peak] <= w1[off_peak] + 1e-6)
