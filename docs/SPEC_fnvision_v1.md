# fnvision – Fovea Native Vision

## Technical Specification v1

Status: Ratified v1.1
Date: 2026-02-21
Updated: 2026-02-21 (v1.1 – reflects MF1 implementation decisions)
License: Apache 2.0
Tag: #fnvision

---

## 1. What Is fnvision

A lightweight, biologically-motivated foveated vision encoder for autonomous agents, robotics,
and AI research.

No heavy transformers. No log-polar distortion. No patch tricks.

Native float32 sampling with a dual fovea (F1/F2) system and emergent zoom via convergence.
A live calibration tool lets you see through the encoder's eyes in real time – like an optician
appointment for your model.

---

## 2. The Gap in Existing Approaches

| Approach | Problem |
| --- | --- |
| Log-Polar (classical, PMC8645638) | Geometric distortion – CNNs cannot handle curved straight lines |
| Foveated Tokenization (Meta, CVPR 2025) | Patch-based, transformer-bound, no true fovea |
| Foveated Dynamic Transformer (ICLR 2025) | ViT + MHSA overhead, ~34% compute reduction but still heavyweight |

fnvision takes the first-principles approach: model vision the way biology does it, without
cramming it into existing heavy architectures.

---

## 3. Core Concept – Binocular F-System

### 3.1 Two Foveal Centers

Vision is built around two coupled foveal centers: **F1** and **F2**.

Each center contributes a Gaussian weight field centered on its position:

```text
w_i(x) = exp(-||x - p_i||^2 / (2 * sigma_i^2))
```

Where:

- `p_i` = normalized position of foveal center i in `[0, 1]^2`
- `sigma_i` = focal radius (configured per center)
- Combined weight: `w(x) = w_1(x) + w_2(x)` (additive superposition)

In the overlap zone both Gaussians add. Effective sharpness is higher there without any explicit
logic – it emerges from the geometry.

### 3.2 Three Resolution Zones

Resolution is sampled non-uniformly based on the normalised combined weight field `w_out(x)`:

| Zone | w_out threshold | Effective Resolution |
| --- | --- | --- |
| Fovea | w_out >= 0.60 | Full (1.0) |
| Parafovea | 0.15 <= w_out < 0.60 | ~60% (configurable) |
| Periphery | w_out < 0.15 | ~15% (configurable) |

Zone boundaries use Hermite smoothstep blending (t²(3-2t)) to avoid hard ring artifacts.
All three zone masks sum to exactly 1.0 at every pixel (compositing invariant).

The zone thresholds (0.60 and 0.15) are configurable via `FoveaConfig` and can be tuned
using the calibration tool. Using weight-field thresholds rather than distance from F-center
avoids a hard boundary artifact on the perpendicular bisector between F1 and F2.

### 3.3 Zoom via Convergence

**The single control parameter for zoom is the normalized distance between F1 and F2.**

| F1/F2 State | Effect |
| --- | --- |
| Maximum separation (resting state) | Wide-angle oval field |
| Minimum separation (co-located) | ~140% effective zoom on focal point |
| Partial convergence | Continuous zoom interpolation |

A passive pull force returns the centers toward maximum separation:

```text
F_pull = k * (d_max - d_current)
```

Zoom costs active energy. Relaxation returns to wide-angle. `k` is the single calibration
parameter.

This mirrors biological reality: accommodation and convergence are mechanically coupled in the
human eye. fnvision models this relationship directly instead of adding a separate zoom
abstraction.

---

## 4. Gaze Dynamics

The gaze point (center of the F-system) moves independently of any motor effector. The caller
drives gaze via salient target coordinates. fnvision handles the movement dynamics.

Movement follows a saccade model:

- Smooth pursuit toward salient target
- Step size capped per tick (`gaze_max_step_norm`)
- Optional hold probability (simulate fixation)
- Configurable jitter for biological realism

fnvision does not compute saliency internally. Saliency is the caller's responsibility.
fnvision is a pure encoder.

---

## 5. Attention Level Parameter

fnvision accepts one optional external signal: `attention_level: float [0.0, 1.0]`.

When `attention_level < 1.0`:

- Periphery resolution scales down proportionally (tunnel effect)
- Inner zones are unaffected until `attention_level < 0.3`

This is a generalized interface. Callers map their own state onto it:

- Autonomous agents: map arousal / energy level
- Robotics: map battery or task urgency
- Not needed: pass `1.0` (default = full attention)

---

## 6. Output Data Contract

```python
@dataclass
class FoveaOutput:
    fovea: np.ndarray           # float32[fovea_h, fovea_w, 3]
    parafovea: np.ndarray       # float32[para_h, para_w, 3]
    periphery: np.ndarray       # float32[peri_h, peri_w, 3]
    weight_map: np.ndarray      # float32[H, W] – combined Gaussian weights (full resolution)
    f1_pos_norm: tuple          # (float, float) – F1 position in [0,1]^2
    f2_pos_norm: tuple          # (float, float) – F2 position in [0,1]^2
    f_separation_norm: float    # current F1/F2 distance (0 = co-located, 1 = max)
    attention_level: float      # echoed back from input
```

All spatial tensors are `float32` in range `[0, 1]`.
Input frames are `uint8` RGB; conversion is handled internally.

---

## 7. Configuration

```python
@dataclass
class FoveaConfig:
    # Zone sizing
    focal_radius_norm: float = 0.12       # sigma for Gaussian in normalized coords
    fovea_res: tuple = (96, 96)           # output resolution for fovea tensor
    parafovea_res: tuple = (128, 128)     # output resolution for parafovea tensor
    periphery_res: tuple = (96, 96)       # output resolution for periphery tensor
    parafovea_res_factor: float = 0.60    # relative resolution at 1–3x sigma
    periphery_res_factor: float = 0.15    # relative resolution beyond 3x sigma

    # Zoom / F-system
    f_separation_max_norm: float = 0.28   # max F1/F2 distance (normalized)
    f_separation_min_norm: float = 0.00   # min distance (fully co-located = max zoom)
    pull_strength: float = 0.015          # spring constant k

    # Gaze dynamics
    gaze_max_step_norm: float = 0.06      # max gaze movement per tick
    gaze_hold_prob: float = 0.12          # probability of fixation hold per tick
    gaze_jitter_norm: float = 0.010       # random jitter amplitude

    # Attention
    periphery_attention_floor: float = 0.15   # min periphery factor at attention_level=0
    attention_inner_threshold: float = 0.30   # attention below this → inner zones also scale

    # Weight field and zone thresholds
    weight_gamma: float = 1.0                 # w_out shaping: w_out = w_norm ** gamma
    fovea_threshold: float = 0.60             # w_out >= threshold → fovea zone
    para_threshold: float = 0.15              # w_out >= threshold → parafovea zone
    zoom_max_bonus: float = 0.40              # zoom = 1.0 + zoom_max_bonus * (1 - sep)
```

Config can be loaded from and saved to YAML via the calibration tool (no PyYAML required).

### 7.1 Formal Definitions (MF1)

**Zoom function** (Opus 4.6, 2026-02-21):

```text
zoom = 1.0 + zoom_max_bonus * (1.0 - f_separation_norm)
  f_separation_norm = 0.0 → zoom = 1.40  (co-located, maximum zoom)
  f_separation_norm = 1.0 → zoom = 1.00  (resting state, wide-angle)
```

**Attention functions** (Opus 4.6, 2026-02-21):

```text
peri_factor  = periphery_attention_floor + (1 - periphery_attention_floor) * attention_level

inner_factor = 1.0                               if attention_level >= attention_inner_threshold
             = 0.5 + 0.5 * (attention_level
                             / attention_inner_threshold)  otherwise
```

Both functions are continuous and monotone in `attention_level`.

---

## 8. Calibration Tool (Optiker-Prinzip)

An interactive live tool to tune all config parameters while watching through the encoder's
eyes in real time.

"If you can see through it, you can tune it."

### Features

- Live preview: what the encoder sees at each zone (fovea / parafovea / periphery side by side)
- Weight map overlay: Gaussian fields as heatmap over the source frame
- Sliders for all `FoveaConfig` parameters with live feedback
- F1/F2 position visualization with draggable handles
- F-separation slider for testing zoom behavior
- Attention level slider for simulating low-attention states
- Input: screen region capture, camera feed, or static image file
- Config export: saves current settings as YAML

### Input Sources

- `--source screen` – capture a configurable screen region (default)
- `--source camera` – webcam feed
- `--source <file>` – static image or video file for offline tuning

---

## 9. API

```python
from fnvision import FoveaConfig, FoveaEncoder

cfg = FoveaConfig(focal_radius_norm=0.12, pull_strength=0.015)
encoder = FoveaEncoder(cfg)

# Encode a frame: uint8 RGB (H x W x 3), gaze at center, wide-angle
result = encoder.encode(
    frame_rgb=frame,
    gaze_xy=(0.5, 0.5),
    f_separation=1.0,      # 1.0 = max separation (wide-angle)
    attention_level=1.0,   # 1.0 = full attention, no tunnel effect
)

# Access outputs
print(result.fovea.shape)        # (96, 96, 3)  float32
print(result.parafovea.shape)    # (128, 128, 3) float32
print(result.periphery.shape)    # (96, 96, 3)  float32
print(result.weight_map.shape)   # (H, W)        float32
print(result.f_separation_norm)  # 1.0

# Zoom in (converge F-centers)
result_zoomed = encoder.encode(
    frame_rgb=frame,
    gaze_xy=(0.5, 0.5),
    f_separation=0.0,      # fully co-located = ~140% zoom
    attention_level=1.0,
)
```

---

## 10. Dependencies

Runtime:

- `numpy`
- `opencv-python` (cv2) – resize and color space operations

Calibration tool additionally:

- `tkinter` (stdlib) or `PyQt5` / `PySide6`
- `Pillow` (PIL)

No deep learning framework required. No CUDA. CPU-only baseline.

---

## 11. Comparison to Existing Approaches

| | fnvision | Log-Polar | Meta CVPR 2025 | ICLR 2025 ViT |
| --- | --- | --- | --- | --- |
| Architecture | Standalone encoder | Classical | Transformer | ViT |
| Distortion | None | High (geometric) | None (patches) | None (patches) |
| Binocular F-system | Yes (F1 + F2) | No | No | No |
| Zoom control | Single parameter | N/A | N/A | N/A |
| GPU required | No | No | Yes | Yes |
| External attention signal | Optional | No | No | No |
| Live calibration tool | Yes | No | No | No |
| pip installable | Yes | No | No | No |

---

## 12. Project Structure

```text
fnvision_dev/                   <- dev environment (not published directly)
├── fnvision/                   <- Python package
│   ├── __init__.py
│   ├── encoder.py              <- FoveaEncoder
│   ├── config.py               <- FoveaConfig
│   ├── weight_field.py         <- Gaussian dual-fovea weight field (pure function)
│   ├── gaze.py                 <- gaze dynamics + F-system spring model (MF2)
│   └── tools/
│       └── calibration.py      <- live calibration tool
├── examples/
│   ├── basic_usage.py
│   ├── calibration_demo.py
│   └── agent_integration.py
├── tests/
│   └── test_encoder.py
├── docs/
│   └── SPEC_fnvision_v1.md     <- this file
├── pyproject.toml
├── LICENSE                     <- Apache 2.0
├── NOTICE                      <- Apache 2.0 required attribution file
├── README.md                   <- public-facing, sells the project
├── CHANGELOG.md
└── CONTRIBUTING.md
```

The `fnvision/` package is the only thing that goes to GitHub (plus examples, tests, docs,
and the metadata files). The dev environment around it stays local.

---

## 13. Development Milestones

### MF1 – Core Encoder

- `FoveaConfig` dataclass + YAML load/save
- `FoveaEncoder.encode()` with Gaussian weight field for F1 + F2
- Three-zone sampling (fovea / parafovea / periphery) from weight field
- `float32` output, `uint8` input conversion
- Basic unit tests

### MF2 – Gaze Dynamics and Zoom

- F1/F2 separation control with spring-damper pull model
- Gaze saccade model (step cap, hold probability, jitter)
- `attention_level` integration (periphery scaling)
- Stateful encoder (gaze state persists between ticks)

### MF3 – Calibration Tool

- Live preview: three zones side by side
- Gaussian weight map overlay as heatmap
- Sliders for all `FoveaConfig` parameters
- F1/F2 draggable handles on preview
- F-separation and attention sliders
- Config export as YAML

### MF4 – Public Release

- `pyproject.toml` (pip installable as `fnvision`)
- Full `README.md` with comparison table and quick start
- All examples runnable standalone
- Apache 2.0 `LICENSE` and `NOTICE` files
- GitHub release with changelog

---

## 14. Open Points

- **Gaussian sampling strategy**: **Resolved (MF1): Dense weight map.** Ring-based
  approximation creates spatial discontinuities on the perpendicular bisector between F1/F2,
  especially visible during convergence zoom. Dense vectorized NumPy pipeline is fast enough
  on CPU. Performance profiling deferred to MF2.
- **Parafovea tensor origin**: **Resolved (MF1): Sampled from weight field.** A fixed annular
  crop would require its own radius parameterisation as a function of F-separation — a
  redundant model of geometry already encoded by the weight field.
- **Calibration tool UI framework**: `tkinter` (zero extra deps, ships with Python) vs.
  `PyQt5` / `PySide6` (richer UI, one extra dep). Current preference: `tkinter` for MF3,
  optional PyQt5 backend later.
- **Binocular asymmetry**: allow F1 and F2 to have independent `sigma` values (e.g. dominant
  eye simulation) or keep symmetric for simplicity. Evaluate after MF2.
