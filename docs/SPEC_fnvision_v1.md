# fnvision - Fovea Native Vision

## Technical Specification (v1.2)

License: Apache 2.0

---

## 1. What Is fnvision

A lightweight, biologically motivated foveated vision encoder for autonomous agents,
robotics, and AI research.

- No transformers required
- No log-polar geometric distortion
- Dual fovea (F1/F2) with emergent zoom by convergence
- Float32 output tensors from uint8 RGB input

---

## 2. Core Concept

### 2.1 Dual Fovea Weight Field

Two foveal centers `F1` and `F2` define the spatial acuity field.

```text
w_i(x)   = exp(-||x - p_i||^2 / (2 * sigma_i^2))
w_raw(x) = w_1(x) + w_2(x)
w_norm   = w_raw / max(w_raw)
w_out    = w_norm ** gamma
```

`w_out` is used for zone blending and is normalized to `[0,1]`.

### 2.2 Three Resolution Zones

| Zone | Threshold on `w_out` | Effective Resolution |
| --- | --- | --- |
| Fovea | `w_out >= 0.60` | Full |
| Parafovea | `0.15 <= w_out < 0.60` | Reduced (`parafovea_res_factor`) |
| Periphery | `w_out < 0.15` | Reduced (`periphery_res_factor`) |

Boundaries use smoothstep blending. The zone masks sum to 1.0 at each pixel.

### 2.3 Zoom by Convergence

Zoom is controlled by normalized F1/F2 separation:

- max separation -> wide-angle
- min separation (co-located) -> max zoom

Formal zoom function:

```text
zoom = 1.0 + zoom_max_bonus * (1.0 - f_separation_norm)
```

With `zoom_max_bonus = 0.40`, convergence yields ~1.4x zoom.

---

## 3. Gaze Dynamics

The stateful gaze layer is implemented in `GazeController.step(...)`.

### 3.1 State

```python
@dataclass
class GazeState:
    gaze_xy: tuple[float, float]
    f_separation_norm: float
    tick: int
```

F1/F2 are derived from `gaze_xy` and `f_separation_norm` (horizontal symmetric in v1.x).

### 3.2 Runtime Order per Tick

1. Clamp inputs (`target_xy`, `attention_level`, `dt`)
2. Hold check (stochastic): if triggered, skip only saccade movement
3. Saccade update with overshoot cap (`gaze_max_step_norm * dt`)
4. First-order spring pull on separation
5. Jitter update if `dt > 0` and `gaze_jitter_norm > 0`
6. Bounds clamp so derived F1/F2 stay inside `[0,1]`
7. `tick += 1`

### 3.3 Invariants

- `dt < 0` or non-finite `dt` -> `ValueError`
- `dt == 0` -> no motion update (tick still increments)
- hold/jitter consume no RNG when `dt == 0`
- separation is clamped to `[f_separation_min_norm, f_separation_max_norm]`
- derived F1/F2 remain in `[0,1]`

### 3.4 RNG Semantics (Option A)

- `rng=None` -> internal `default_rng()` (non-deterministic)
- deterministic reproducibility requires explicit seeded RNG or restored RNG state

---

## 4. Attention Level

`attention_level: float` in `[0,1]` is accepted by the encoder API.

MF1 behavior:

- periphery scales with `attention_level`
- inner zones remain full until `attention_level < attention_inner_threshold`

MF2 behavior:

- jitter is not attention-coupled in v1.2
- optional attention-coupled jitter is deferred to v2 ideas

---

## 5. Output Data Contract

```python
@dataclass
class FoveaOutput:
    fovea: np.ndarray
    parafovea: np.ndarray
    periphery: np.ndarray
    weight_map: np.ndarray
    f1_pos_norm: tuple[float, float]
    f2_pos_norm: tuple[float, float]
    f_separation_norm: float
    attention_level: float
```

All image-like outputs are `float32` in `[0,1]`.

---

## 6. Configuration

```python
@dataclass
class FoveaConfig:
    focal_radius_norm: float = 0.12
    fovea_res: tuple[int, int] = (96, 96)
    parafovea_res: tuple[int, int] = (128, 128)
    periphery_res: tuple[int, int] = (96, 96)
    parafovea_res_factor: float = 0.60
    periphery_res_factor: float = 0.15

    f_separation_max_norm: float = 0.28
    f_separation_min_norm: float = 0.00
    zoom_max_bonus: float = 0.40
    pull_strength: float = 0.015

    gaze_max_step_norm: float = 0.06
    gaze_hold_prob: float = 0.12
    gaze_jitter_norm: float = 0.010

    periphery_attention_floor: float = 0.15
    attention_inner_threshold: float = 0.30

    weight_gamma: float = 1.0
    fovea_threshold: float = 0.60
    para_threshold: float = 0.15
```

---

## 7. API

Stateless encoder usage:

```python
from fnvision import FoveaConfig, FoveaEncoder

cfg = FoveaConfig()
encoder = FoveaEncoder(cfg)
out = encoder.encode(frame_rgb=frame, gaze_xy=(0.5, 0.5), f_separation=1.0, attention_level=1.0)
```

Stateful MF2 usage:

```python
from fnvision import FoveaConfig, FoveaEncoder, GazeController

cfg = FoveaConfig()
encoder = FoveaEncoder(cfg)
gaze = GazeController(config=cfg)

for frame in stream:
    st = gaze.step(target_xy=(0.6, 0.4), attention_level=1.0, dt=1.0)
    sep_norm_01 = st.f_separation_norm / cfg.f_separation_max_norm if cfg.f_separation_max_norm > 0 else 0.0
    out = encoder.encode(frame_rgb=frame, gaze_xy=st.gaze_xy, f_separation=sep_norm_01, attention_level=1.0)
```

---

## 8. Dependencies

Runtime:

- `numpy`
- `opencv-python`

Optional tools:

- `Pillow`
- OpenCV HighGUI (`cv2.namedWindow` + trackbars) for calibration UI

---

## 9. Project Structure

```text
fnvision_dev/
|-- fnvision/
|   |-- __init__.py
|   |-- config.py
|   |-- encoder.py
|   |-- weight_field.py
|   |-- gaze.py
|   `-- tools/
|       |-- __init__.py
|       `-- calibration.py
|-- tests/
|   |-- test_encoder.py
|   `-- test_gaze.py
|-- tools/
|   `-- build_index.py
|-- docs/
|   `-- SPEC_fnvision_v1.md
|-- README.md
|-- CHANGELOG.md
|-- pyproject.toml
|-- LICENSE
`-- NOTICE
```

---
