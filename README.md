# fnvision – Fovea Native Vision

<!-- badges: activate after first GitHub release -->
<!-- ![Build](https://github.com/soenning-ai/fnvision/actions/workflows/tests.yml/badge.svg) -->
<!-- ![PyPI](https://img.shields.io/pypi/v/fnvision) -->
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

A lightweight, biologically-motivated foveated vision encoder for autonomous agents,
robotics, and AI research.

**No heavy transformers. No log-polar distortion. No patch tricks.**

---

## The Idea

Human vision does not sample the world uniformly. The fovea sees in sharp detail; the
periphery sees motion. This resolution gradient is what makes biological vision so efficient.

fnvision models this directly, from first principles:

- Two coupled foveal centers (F1, F2) with Gaussian resolution fields
- Three zones: fovea, parafovea, periphery – driven by the combined weight field
- Zoom via convergence: the distance between F1 and F2 is the only zoom parameter
- Float32 output, no geometric distortion
- An interactive calibration tool to tune parameters while watching through the encoder's eyes

## Why Not Existing Approaches

| | fnvision | Log-Polar | Meta CVPR 2025 | ICLR 2025 ViT |
| --- | --- | --- | --- | --- |
| Architecture | Standalone encoder | Classical | Transformer | ViT |
| Distortion | None | High (geometric) | None | None |
| Binocular F-system | Yes | No | No | No |
| Zoom control | Single parameter | N/A | N/A | N/A |
| GPU required | No | No | Yes | Yes |
| Live calibration tool | Yes | No | No | No |

## Installation

```bash
pip install fnvision
```

For the calibration tool:

```bash
pip install fnvision[tools]
```

## Quick Start

### Stateless Encoding (MF1)

```python
import numpy as np
from fnvision import FoveaConfig, FoveaEncoder

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
encoder = FoveaEncoder(FoveaConfig())

result = encoder.encode(
    frame_rgb=frame,
    gaze_xy=(0.5, 0.5),         # center of frame
    f_separation=1.0,           # max separation = wide-angle
    attention_level=1.0,        # full attention
)

print(result.fovea.shape)       # (96, 96, 3)  float32
print(result.periphery.shape)   # (96, 96, 3)  float32
print(result.weight_map.shape)  # (480, 640)   float32
```

### Stateful Gaze Dynamics (MF2)

```python
from fnvision import FoveaConfig, FoveaEncoder, GazeController

cfg = FoveaConfig()
encoder = FoveaEncoder(cfg)
gaze = GazeController(cfg, initial_gaze=(0.5, 0.5))

for frame in video_frames:
    state = gaze.step(target_xy=(0.7, 0.3), dt=1.0)
    result = encoder.encode(
        frame_rgb=frame,
        gaze_xy=state.gaze_xy,
        f_separation=state.f_separation_norm,
    )
    # result.fovea tracks the target with saccades, hold, and jitter
```

### Deterministic Replay

```python
import numpy as np
from fnvision import FoveaConfig, GazeController

gaze = GazeController(
    FoveaConfig(),
    rng=np.random.default_rng(42),   # fixed seed = reproducible
)
# snapshot mid-sequence, reset later for exact replay
state_copy, rng_state = gaze.snapshot()
```

### Calibration Tool (M3 Baseline)

```bash
fnvision-calibrate --source camera
```

Optional sources:

```bash
fnvision-calibrate --source file --path path\\to\\image_or_video.mp4
fnvision-calibrate --source screen --region 100,100,1280,720
```

Keys:

- `q` / `Esc`: quit
- `r`: reset gaze to center
- `s`: save current config YAML snapshot

### M3 Calibration Examples

Sample collages captured from the calibration tool:

![M3 calibration sample 1](image/calibration_demo_1.png)
![M3 calibration sample 2](image/calibration_demo_2.png)

## Documentation

- [Technical Specification](docs/SPEC_fnvision_v1.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [GitHub](https://github.com/soenning-ai/fnvision)

## Development Documentation (Internal)

- [Project Index](INDEX.md)
- [DEV_NOTES](DEV_NOTES.md)
- [DEV_NOTES_M2](DEV_NOTES_M2.md)
- [DEV_NOTES_M3](DEV_NOTES_M3.md)
- [Nachbesprechung M1](nachbesprechung_m1.md)
- [Nachbesprechung M2](nachbesprechung_m2.md)

## License

Apache 2.0 – see [LICENSE](LICENSE).
