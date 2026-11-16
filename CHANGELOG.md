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

### Fixed

* `train_yolov5.py` resumed from an undefined name (`prev_weights_path`); it
  now resumes from the `last.pt` that it just checked for.
* The data generators wrote `filenames.txt` through a file that they never
  closed; they write it once at the end.
* `Estimator` called the deprecated `logger.warn`.
* The empty-list defaults of `Localizer`, `Estimator` and `CATCHapp` (mutable
  defaults, which failed on an empty list in any case): `images` is required
  for `Localizer` and `Estimator`, and `CATCHapp.analyze(None)` returns an
  empty DataFrame.
* The lint findings of the library, `training/` and `utilities/` (unused
  imports and loop variables, `zip` without `strict`, long lines).
