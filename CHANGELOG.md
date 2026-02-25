# Changelog

All notable changes to fnvision are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `fnvision/tools/calibration.py`: interactive calibration tool (camera/file/screen, live preview, sliders, YAML save).
- Calibration tool auto mode: `auto_spring` toggle using `GazeController`.
- Calibration tool slider `sep_max x100` for live tuning of spring rest separation.
- `tests/test_calibration.py`: parser/threshold/auto-mode/screen-failure coverage.
- M3 visual examples in `README.md`:
  - `image/calibration_demo_1.png`
  - `image/calibration_demo_2.png`

### Changed

- README installation flow is now GitHub-first; PyPI path is optional.
- README MF2 example now normalizes `f_separation` correctly to `[0,1]`.
- Public spec links consistently point to `docs/SPEC_fnvision_v1.md`.
- `docs/SPEC_fnvision_v1.md` cleaned to a timeless technical state (history moved to changelog).
- `tools/build_index.py` no longer emits absolute root paths into `INDEX.md`.

### Fixed

- Calibration crash when `para_thr` slider was set to `0`.
- Screen source capture now fails gracefully (no hard crash on grab errors).
- In auto mode, `f_sep` slider changes now apply as live set-impulses.

---

## [0.1.0] - 2026-02-21

### Added

- `fnvision/config.py`: `FoveaConfig` validation + YAML round-trip, `FoveaOutput`.
- `fnvision/weight_field.py`: vectorized dual-Gaussian weight map (`compute_weight_map`).
- `fnvision/encoder.py`: stateless `encode()` with three-zone compositing.
- `fnvision/__init__.py`: public MF1 API exports.
- `tests/test_encoder.py`: MF1 test coverage.
- Project metadata: `pyproject.toml`, `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `README.md`.

### Architecture Decisions

- Weight-field threshold zoning over min-distance zoning.
- Dense weight-map pipeline over ring approximation.
- Stateless `encode()` for MF1; stateful gaze deferred to MF2.
- Edge padding in crop helper for border robustness.

### Fixed

- `pyproject.toml` build backend corrected to `setuptools.build_meta`.
