# Changelog

All notable changes to fnvision are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `fnvision/gaze.py`: stateful `GazeController` + `GazeState`
- `tests/test_gaze.py`: MF2 deterministic/stochastic/replay test coverage
- `nachbesprechung_m2.md`: MF2 completion documentation

### Changed

- `SPEC_fnvision_v1.md` updated to v1.2 (MF2 completion, gates, validation snapshot)
- `README.md` technical specification link now points to `SPEC_fnvision_v1.md`
- `pyproject.toml` documentation URL now points to `.../SPEC_fnvision_v1.md`

---

## [0.1.0] - 2026-02-21 (MF1 - Core Encoder)

### Added

- `fnvision/config.py`: `FoveaConfig` validation + YAML round-trip, `FoveaOutput`
- `fnvision/weight_field.py`: vectorized dual-Gaussian weight map (`compute_weight_map`)
- `fnvision/encoder.py`: stateless `encode()` with three-zone compositing
- `fnvision/__init__.py`: public MF1 API exports
- `tests/test_encoder.py`: MF1 test coverage
- `SPEC_fnvision_v1.md` (v1.1 at MF1 close)
- Project metadata: `pyproject.toml`, `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `README.md`

### Architecture Decisions

- Weight-field threshold zoning over min-distance zoning
- Dense weight-map pipeline over ring approximation
- Stateless `encode()` for MF1; stateful gaze deferred to MF2
- Edge padding in crop helper for border robustness

### Fixed

- `pyproject.toml` build backend corrected to `setuptools.build_meta`

