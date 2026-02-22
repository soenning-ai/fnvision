import sys
import types

import pytest

from fnvision import FoveaConfig
from fnvision.tools import calibration as calibmod
from fnvision.tools.calibration import FrameSource, _parse_region


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
    cfg, attention, f_sep = calibmod._read_cfg_from_trackbars(FoveaConfig())

    assert attention == 1.0
    assert f_sep == 1.0
    assert 0.0 < cfg.para_threshold < cfg.fovea_threshold <= 1.0
