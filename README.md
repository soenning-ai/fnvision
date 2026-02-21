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

```python
# placeholder – fill in after MF1 implementation
```

## Documentation

- [Technical Specification](docs/SPEC_fnvision_v1.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [GitHub](https://github.com/soenning-ai/fnvision)

## License

Apache 2.0 – see [LICENSE](LICENSE).
