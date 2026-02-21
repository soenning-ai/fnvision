# Changelog

All notable changes to fnvision are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

MF2 – Gaze Dynamics: F1/F2 spring-damper pull, saccade model, stateful encoder.

---

## [0.1.0] – 2026-02-21 (MF1 – Core Encoder)

### Added

- `fnvision/config.py` – `FoveaConfig` dataclass with full validation and YAML round-trip
  (stdlib only, no PyYAML dependency); `FoveaOutput` dataclass (8 fields: fovea, parafovea,
  periphery, weight_map, f1_pos_norm, f2_pos_norm, f_separation_norm, attention_level)
- `fnvision/weight_field.py` – `compute_weight_map()`: pure, vectorized NumPy dual-Gaussian
  weight field; float32, deterministic, no side effects; optional gamma shaping
- `fnvision/encoder.py` – `FoveaEncoder` with stateless `encode()` method; three-zone
  compositing (fovea / parafovea / periphery) with Hermite smoothstep blending and
  enforced compositing invariant (m_f + m_p + m_r = 1 at every pixel)
- `fnvision/__init__.py` – public API: `FoveaConfig`, `FoveaOutput`, `FoveaEncoder`
- `tests/test_encoder.py` – 16 test classes covering all Opus-mandated cases: dtypes,
  value ranges, output shapes, compositing invariant, determinism, convergence (f_sep=0),
  edge gaze, attention extremes, zoom effect, aspect ratio, YAML round-trip, input guards,
  metadata echo-back, weight map properties, weight field symmetry and gamma
- `SPEC_fnvision_v1.md` – full technical specification (binocular F-system, zoom via
  convergence, three-zone sampling, calibration tool, API design)
- Project metadata: `pyproject.toml`, `LICENSE` (Apache 2.0), `NOTICE`, `CONTRIBUTING.md`,
  `README.md` placeholder

### Architecture decisions

- Zone assignment via weight-field thresholds (fovea: w_out >= 0.60, parafovea: w_out >= 0.15)
  rather than min-distance — avoids hard boundary artifacts on the F1/F2 bisector
- Dense weight map over ring approximation — single vectorized NumPy pipeline, no spatial
  discontinuities during convergence zoom
- Stateless `encode()` for MF1; stateful gaze dynamics deferred to MF2
- Edge-replication padding (`mode="edge"`) in crop helper — prevents zero-border artifacts
  when crop extends beyond frame boundaries

### Fixed

- `pyproject.toml`: build-backend corrected from `setuptools.backends.legacy:build` to
  `setuptools.build_meta` (PEP 517 compatible, works with all supported setuptools versions)

### Spec

- `SPEC_fnvision_v1.md` updated to v1.1: zone assignment via weight-field thresholds,
  formal definitions for zoom and attention functions, additional config parameters,
  resolved open design questions, `weight_field.py` added to project structure
