import sys
import types

import pytest

from fnvision import FoveaConfig
from fnvision.tools import calibration as calibmod
from fnvision.tools.calibration import FrameSource, _parse_region, _sep01_to_abs


def test_parse_region_valid() -> None:
    assert _parse_region("10,20,640,480") == (10, 20, 640, 480)
    assert _parse_region("-100,0,320,200") == (-100, 0, 320, 200)
    assert _parse_region(None) is None


@pytest.mark.parametrize("raw", ["10,20,640", "1,2,3,0", "1,2,-3,4"])
def test_parse_region_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        _parse_region(raw)


def test_screen_source_read_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pil = types.ModuleType("PIL")
    fake_imagegrab = types.ModuleType("PIL.ImageGrab")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("grab failed")

    fake_imagegrab.grab = _boom  # type: ignore[attr-defined]
    fake_pil.ImageGrab = fake_imagegrab  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", fake_imagegrab)

    source = FrameSource(source="screen", path=None, camera_index=0, region=None)
    assert source.read_rgb() is None


def test_trackbar_thresholds_clamp_para_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "attention x100": 100,
        "auto_spring 0/1": 0,
        "sep_max x100": 28,
        "f_sep x100": 100,
        "focal_radius x1000": 120,
        "para_factor x100": 60,
        "peri_factor x100": 15,
        "weight_gamma x100": 100,
        "fovea_thr x100": 60,
        "para_thr x100": 0,
    }

    def _fake_get_trackbar_pos(name: str, _window: str) -> int:
        return values[name]

    monkeypatch.setattr(calibmod.cv2, "getTrackbarPos", _fake_get_trackbar_pos)
    cfg, attention, f_sep, auto_mode = calibmod._read_cfg_from_trackbars(FoveaConfig())

    assert attention == 1.0
    assert f_sep == 1.0
    assert auto_mode is False
    assert cfg.f_separation_max_norm == pytest.approx(0.28)
    assert 0.0 < cfg.para_threshold < cfg.fovea_threshold <= 1.0


def test_trackbar_auto_mode_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "attention x100": 50,
        "auto_spring 0/1": 1,
        "sep_max x100": 35,
        "f_sep x100": 25,
        "focal_radius x1000": 120,
        "para_factor x100": 60,
        "peri_factor x100": 15,
        "weight_gamma x100": 100,
        "fovea_thr x100": 60,
        "para_thr x100": 10,
    }

    def _fake_get_trackbar_pos(name: str, _window: str) -> int:
        return values[name]

    monkeypatch.setattr(calibmod.cv2, "getTrackbarPos", _fake_get_trackbar_pos)
    cfg, attention, f_sep, auto_mode = calibmod._read_cfg_from_trackbars(FoveaConfig())

    assert attention == 0.5
    assert f_sep == 0.25
    assert auto_mode is True
    assert cfg.f_separation_max_norm == pytest.approx(0.35)


def test_sep01_to_abs_clamped() -> None:
    cfg = FoveaConfig(f_separation_min_norm=0.05, f_separation_max_norm=0.25)
    assert _sep01_to_abs(0.0, cfg) == pytest.approx(0.05)
    assert _sep01_to_abs(0.5, cfg) == pytest.approx(0.125)
    assert _sep01_to_abs(1.0, cfg) == pytest.approx(0.25)
