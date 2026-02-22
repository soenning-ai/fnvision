"""Interactive calibration tool for fnvision.

M3 baseline:
- live preview of source + weight overlay + fovea/parafovea/periphery
- mouse-driven gaze target
- sliders for key FoveaConfig parameters
- YAML save/load support via FoveaConfig
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fnvision import FoveaConfig, FoveaEncoder


WINDOW_NAME = "fnvision M3 Calibration"
TRACKBAR_WINDOW = "fnvision M3 Controls"


def _noop(_value: int) -> None:
    return


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _to_bgr_u8(img_rgb_or_gray: np.ndarray) -> np.ndarray:
    if img_rgb_or_gray.ndim == 2:
        img = np.clip(img_rgb_or_gray * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.clip(img_rgb_or_gray * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _parse_region(raw: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be 'x,y,w,h'")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--region width/height must be > 0")
    return x, y, w, h


class FrameSource:
    def __init__(
        self,
        source: str,
        path: Optional[str],
        camera_index: int,
        region: Optional[tuple[int, int, int, int]],
    ) -> None:
        self.source = source
        self.path = path
        self.camera_index = camera_index
        self.region = region
        self.cap: Optional[cv2.VideoCapture] = None
        self.static_rgb: Optional[np.ndarray] = None
        self._init_source()

    def _init_source(self) -> None:
        if self.source == "camera":
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"camera index {self.camera_index} not available")
            return

        if self.source == "file":
            if not self.path:
                raise ValueError("--path is required for --source file")
            ext = Path(self.path).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                bgr = cv2.imread(self.path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"failed to read image file: {self.path}")
                self.static_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                self.cap = cv2.VideoCapture(self.path)
                if not self.cap.isOpened():
                    raise RuntimeError(f"failed to open video file: {self.path}")
            return

        if self.source == "screen":
            # Pillow is optional dependency for tool mode.
            try:
                from PIL import ImageGrab  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "screen source requires Pillow (pip install fnvision[tools])"
                ) from exc
            _ = ImageGrab
            return

        raise ValueError(f"unsupported source: {self.source}")

    def read_rgb(self) -> Optional[np.ndarray]:
        if self.static_rgb is not None:
            return self.static_rgb.copy()

        if self.source == "screen":
            try:
                from PIL import ImageGrab  # type: ignore

                if self.region is None:
                    img = ImageGrab.grab(all_screens=True)
                else:
                    x, y, w, h = self.region
                    img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
                return np.array(img.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                print(f"screen capture failed: {exc}")
                return None

        if self.cap is None:
            return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


@dataclass
class UIState:
    gaze_xy: tuple[float, float] = (0.5, 0.5)
    mouse_down: bool = False
    panel_w: int = 320
    panel_h: int = 240


def _mouse_cb(event: int, x: int, y: int, _flags: int, user_data: UIState) -> None:
    panel_w = user_data.panel_w
    panel_h = user_data.panel_h
    inside_source = 0 <= x < panel_w and 0 <= y < panel_h

    if event == cv2.EVENT_LBUTTONDOWN:
        user_data.mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        user_data.mouse_down = False

    if (event == cv2.EVENT_MOUSEMOVE and user_data.mouse_down) or event == cv2.EVENT_LBUTTONDOWN:
        if inside_source:
            gx = _clamp01(x / max(panel_w - 1, 1))
            gy = _clamp01(y / max(panel_h - 1, 1))
            user_data.gaze_xy = (gx, gy)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="fnvision interactive calibration tool")
    p.add_argument("--source", choices=["camera", "file", "screen"], default="camera")
    p.add_argument("--path", default=None, help="image/video path when --source file")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--region", default=None, help="screen region as x,y,w,h")
    p.add_argument("--config", default=None, help="optional path to load FoveaConfig YAML")
    p.add_argument("--save-path", default="fnvision_calibration.yaml")
    p.add_argument("--panel-width", type=int, default=320)
    p.add_argument("--panel-height", type=int, default=240)
    p.add_argument("--max-fps", type=float, default=30.0)
    return p


def _create_trackbars(cfg: FoveaConfig) -> None:
    cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("attention x100", TRACKBAR_WINDOW, 100, 100, _noop)
    cv2.createTrackbar("f_sep x100", TRACKBAR_WINDOW, 100, 100, _noop)
    cv2.createTrackbar(
        "focal_radius x1000", TRACKBAR_WINDOW, int(round(cfg.focal_radius_norm * 1000)), 500, _noop
    )
    cv2.createTrackbar(
        "para_factor x100", TRACKBAR_WINDOW, int(round(cfg.parafovea_res_factor * 100)), 100, _noop
    )
    cv2.createTrackbar(
        "peri_factor x100", TRACKBAR_WINDOW, int(round(cfg.periphery_res_factor * 100)), 100, _noop
    )
    cv2.createTrackbar(
        "weight_gamma x100", TRACKBAR_WINDOW, int(round(cfg.weight_gamma * 100)), 400, _noop
    )
    cv2.createTrackbar(
        "fovea_thr x100", TRACKBAR_WINDOW, int(round(cfg.fovea_threshold * 100)), 99, _noop
    )
    cv2.createTrackbar(
        "para_thr x100", TRACKBAR_WINDOW, int(round(cfg.para_threshold * 100)), 98, _noop
    )


def _read_cfg_from_trackbars(base_cfg: FoveaConfig) -> tuple[FoveaConfig, float, float]:
    att = cv2.getTrackbarPos("attention x100", TRACKBAR_WINDOW) / 100.0
    f_sep = cv2.getTrackbarPos("f_sep x100", TRACKBAR_WINDOW) / 100.0
    focal = max(0.005, cv2.getTrackbarPos("focal_radius x1000", TRACKBAR_WINDOW) / 1000.0)
    para_factor = max(0.01, cv2.getTrackbarPos("para_factor x100", TRACKBAR_WINDOW) / 100.0)
    peri_factor = max(0.01, cv2.getTrackbarPos("peri_factor x100", TRACKBAR_WINDOW) / 100.0)
    gamma = max(0.1, cv2.getTrackbarPos("weight_gamma x100", TRACKBAR_WINDOW) / 100.0)
    raw_f_thr = cv2.getTrackbarPos("fovea_thr x100", TRACKBAR_WINDOW) / 100.0
    raw_p_thr = cv2.getTrackbarPos("para_thr x100", TRACKBAR_WINDOW) / 100.0

    # Keep thresholds valid for config validation:
    # 0 < para_threshold < fovea_threshold <= 1
    f_thr = float(np.clip(raw_f_thr, 0.02, 0.99))
    p_thr = float(np.clip(raw_p_thr, 0.01, 0.98))
    if p_thr >= f_thr:
        p_thr = max(0.01, f_thr - 0.01)

    cfg = FoveaConfig(
        focal_radius_norm=focal,
        fovea_res=base_cfg.fovea_res,
        parafovea_res=base_cfg.parafovea_res,
        periphery_res=base_cfg.periphery_res,
        parafovea_res_factor=para_factor,
        periphery_res_factor=peri_factor,
        f_separation_max_norm=base_cfg.f_separation_max_norm,
        f_separation_min_norm=base_cfg.f_separation_min_norm,
        zoom_max_bonus=base_cfg.zoom_max_bonus,
        pull_strength=base_cfg.pull_strength,
        gaze_max_step_norm=base_cfg.gaze_max_step_norm,
        gaze_hold_prob=base_cfg.gaze_hold_prob,
        gaze_jitter_norm=base_cfg.gaze_jitter_norm,
        periphery_attention_floor=base_cfg.periphery_attention_floor,
        attention_inner_threshold=base_cfg.attention_inner_threshold,
        weight_gamma=gamma,
        fovea_threshold=f_thr,
        para_threshold=p_thr,
    )
    return cfg, att, f_sep


def _panel(img_bgr: np.ndarray, size: tuple[int, int], title: str) -> np.ndarray:
    w, h = size
    out = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        out,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def run_calibration(args: argparse.Namespace) -> int:
    cfg = FoveaConfig.from_yaml(args.config) if args.config else FoveaConfig()
    source = FrameSource(
        source=args.source,
        path=args.path,
        camera_index=args.camera_index,
        region=_parse_region(args.region),
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    ui = UIState(panel_w=max(200, int(args.panel_width)), panel_h=max(160, int(args.panel_height)))
    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb, ui)
    _create_trackbars(cfg)

    target_dt = 1.0 / max(args.max_fps, 1.0)

    try:
        while True:
            t0 = time.perf_counter()
            frame_rgb = source.read_rgb()
            if frame_rgb is None:
                print("source ended or unavailable, exiting calibration")
                break

            cfg_now, attention, f_sep = _read_cfg_from_trackbars(cfg)
            encoder = FoveaEncoder(cfg_now)
            out = encoder.encode(
                frame_rgb=frame_rgb,
                gaze_xy=ui.gaze_xy,
                f_separation=f_sep,
                attention_level=attention,
            )

            src_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            H, W = src_bgr.shape[:2]
            f1x = int(round(out.f1_pos_norm[0] * max(W - 1, 1)))
            f1y = int(round(out.f1_pos_norm[1] * max(H - 1, 1)))
            f2x = int(round(out.f2_pos_norm[0] * max(W - 1, 1)))
            f2y = int(round(out.f2_pos_norm[1] * max(H - 1, 1)))
            gx = int(round(ui.gaze_xy[0] * max(W - 1, 1)))
            gy = int(round(ui.gaze_xy[1] * max(H - 1, 1)))

            cv2.circle(src_bgr, (f1x, f1y), 6, (255, 80, 80), 2)
            cv2.circle(src_bgr, (f2x, f2y), 6, (80, 255, 80), 2)
            cv2.drawMarker(src_bgr, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

            wm_u8 = np.clip(out.weight_map * 255.0, 0, 255).astype(np.uint8)
            wm_color = cv2.applyColorMap(wm_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(src_bgr, 0.65, wm_color, 0.35, 0.0)

            fovea_bgr = _to_bgr_u8(out.fovea)
            para_bgr = _to_bgr_u8(out.parafovea)
            peri_bgr = _to_bgr_u8(out.periphery)

            panel_size = (ui.panel_w, ui.panel_h)
            row = np.concatenate(
                [
                    _panel(src_bgr, panel_size, "source"),
                    _panel(overlay, panel_size, "weight_overlay"),
                    _panel(fovea_bgr, panel_size, "fovea"),
                    _panel(para_bgr, panel_size, "parafovea"),
                    _panel(peri_bgr, panel_size, "periphery"),
                ],
                axis=1,
            )

            status = (
                f"gaze=({ui.gaze_xy[0]:.2f},{ui.gaze_xy[1]:.2f}) "
                f"att={attention:.2f} sep={f_sep:.2f} "
                f"sigma={cfg_now.focal_radius_norm:.3f} "
                f"gamma={cfg_now.weight_gamma:.2f} "
                "keys: [s]=save [r]=center [q/esc]=quit"
            )
            cv2.putText(
                row,
                status,
                (8, row.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, row)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                ui.gaze_xy = (0.5, 0.5)
            if key == ord("s"):
                save_path = Path(args.save_path)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if save_path.suffix:
                    target = save_path.with_name(f"{save_path.stem}_{stamp}{save_path.suffix}")
                else:
                    target = Path(str(save_path) + f"_{stamp}.yaml")
                cfg_now.to_yaml(target)
                print(f"saved config: {target}")

            elapsed = time.perf_counter() - t0
            sleep_s = target_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        source.close()
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_calibration(args)


if __name__ == "__main__":
    raise SystemExit(main())
