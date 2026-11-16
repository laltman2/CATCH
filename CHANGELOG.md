# Changelog

All notable changes to CATCH are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

* `pyproject.toml` (packaging, ruff, pytest and coverage settings), `.gitignore`,
  a pre-commit configuration, a CI workflow (ruff, actionlint and tests on
  Python 3.11 to 3.13), Dependabot for the workflows, and this changelog.
* `tests/` with a test of the packaging metadata.

### Changed

* `TorchEstimator` (formerly `training/torch_estimator_arch.py`) and
  `ParamScale` moved from `training/` to the package root, because inference
  needs them; `training/` and `utilities/` are packages now (`__init__.py`)
  and `import CATCH` loads nothing from either.  The training script imports
  `CATCH.TorchEstimator` instead of a sibling module.

* `import CATCH` no longer imports `yolov5`: `CATCH.Localizer` and
  `CATCH.CATCHapp` load on first use (`from CATCH import Localizer` still
  works), so the package imports, and the tests run, without YOLOv5.
