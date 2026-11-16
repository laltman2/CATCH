# Changelog

All notable changes to CATCH are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

* Tests of the Localizer (against a stub of `yolov5`), Estimator (random
  weights, and a round trip through a saved checkpoint), CATCHapp, the network,
  `ParamScale` and the training helpers: 45 tests, 99% coverage, with a floor
  of 98% in `pyproject.toml`.  Training and data generation are omitted from
  coverage until they are rebuilt.
* `pyproject.toml` (packaging, ruff, pytest and coverage settings), `.gitignore`,
  a pre-commit configuration, a CI workflow (ruff, actionlint and tests on
  Python 3.11 to 3.13), Dependabot for the workflows, and this changelog.
* `tests/` with a test of the packaging metadata.

### Changed

* The example holograms (`test_image_crop.png`, `test_image_large.png`) and
  `scale_float_eval.png` moved to `docs/images/`, where media files belong;
  the `example()` functions and the tutorial read them from there.
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
* `Localizer` returned the row number of a detection as its `label`; it now
  returns the class name (`'hologram'`).
* `Localizer.find_center` raised `ValueError` for a grayscale (2-D) image.
* `Estimator` compared a `torch.device` with the string `'cpu'`, so it always
  moved the model and the images to the device; it compares the device type.
* `from CATCH import Localizer` could return the submodule instead of the
  class once `CATCH.CATCHapp` had been imported; the lazy loader binds the
  classes after it imports them.

### Removed

* `eval/` and `experiments/`: scripts that imported APIs that no longer exist
  (`CATCH.CATCHobject`, `LMHologram`) and could not run.  They remain in the
  git history; the rebuild writes new evaluation code.
