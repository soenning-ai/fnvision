# Contributing to fnvision

Thank you for your interest in contributing.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Before You Start

- Read `docs/SPEC_fnvision_v1.md` to understand the design philosophy.
- Read [ACCEPTABLE_USE_POLICY.md](ACCEPTABLE_USE_POLICY.md) to understand the project's ethical usage intent.
- The core principle: first-principles biological accuracy, no architectural bloat.
  If a contribution requires adding a heavy dependency (e.g. PyTorch, TensorFlow),
  it belongs in a separate integration package, not in the core encoder.

## Reporting Issues

- Use GitHub Issues.
- Use the provided issue templates (Bug report / Feature request) where applicable.
- For bugs: include Python version, OS, a minimal reproducible example, and the full
  traceback.
- For feature requests: describe the use case first, not the implementation.

## Submitting Pull Requests

1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make your changes.
3. Run tests: `python -m pytest tests -q`
4. Ensure no new hard dependencies are added to the core `fnvision/` package without
   discussion.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a PR using the PR template with a clear description of what changed and why.

## Code Style

- Python 3.10+, type hints throughout.
- Docstrings for all public classes and functions (Google style).
- No lines over 100 characters.
- `float32` is the canonical dtype for all tensor outputs - do not change this.

## Testing

- Unit tests live in `tests/`.
- New encoder features must include at least one test that verifies output shape and dtype.
- The calibration tool (GUI) is excluded from automated testing but must launch without errors
  on a static image input.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0,
as described in `LICENSE`.

