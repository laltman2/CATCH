# Changelog

All notable changes to CATCH are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

* Tests of data generation (`tests/test_datagen.py`) against the installed
  pylorenzmie; they skip without it, and CI installs the `train` extra.

* A pyright check (CI job, pinned to 1.1.414; configuration in
  `pyproject.toml`) of the library and the tests, with a stub for the optional
  `yolov5` in `typings/`.  The training scripts, `utilities/` and `examples/`
  are not checked until they are rebuilt.
* Sphinx documentation in `docs/` (quickstart, conventions, API reference),
  built with warnings as errors by a CI job and by Read the Docs
  (`.readthedocs.yaml`; the project must be enabled on readthedocs.org).  The
  `docs` extra and `docs/requirements.txt` list what it needs.
* A new `README.md`: what CATCH does, installation (the `yolo` extra), usage
  and the columns of the result, training status and development.
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

* Data generation (`utilities/mtd.py`, `training/YOLO_data_generator.py`,
  `training/Torch_DataLoader.py`) is ported to pylorenzmie 3.x, which the
  `train` extra now requires (`pylorenzmie>=3.7`): `make_model` builds the
  calculator from the shape and the instrument (`model.meshgrid` replaces the
  removed `coordinates`), holograms are computed with `model.particle = ...`
  and `model.hologram()`, and `particle_properties` replaces
  `Sphere.dumps`.  The default Estimator, trained on the old data, has median
  errors of 2.9% (z), 2.9% (a) and 2.1% (n) on 150 holograms that the new code
  generates.

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

* `feature_extent` took the zero crossings from a flattened 2-D grid, so for
  a particle far from the focus (too few fringes in the 300 px range) it
  returned an index in the next row of the grid, 348 to 414 px in 5 of 300
  random cases in the training ranges (z_p of 416 to 598 px), and the boxes
  and crops of those particles were 20-40% too large.  It looks along one row,
  returns the range when there are too few fringes, and is about 25 times
  faster (0.2 ms instead of 5 ms).

* `make_sample` (in `utilities/mtd.py` and `training/YOLO_data_generator.py`)
  looped forever when the field of view was too small to keep the spheres
  apart; it tries `max_attempts` (1000) positions for each sphere and then
  raises `RuntimeError`.

* `scale_int` set the particle to `None` (it assigned the result of
  `list.append`) and `scale_float` assigned through a `lorenzmie` attribute
  that is a method now.
* `make_value` of several ranges (excluded regions) passed probabilities that
  did not sum to 1 to `numpy.random.choice`, so it always raised.

* `from CATCH import Localizer` (or `CATCHapp`) returned the submodule once
  `CATCH.CATCHapp` had been imported first, which broke `Localizer()`; the two
  names are properties of the package module now, which no import order can
  hide.  `__all__` lists only `Estimator`, so `from CATCH import *` works
  without yolov5.
* `CATCHapp.analyze([])` raised `ValueError` (it returns an empty DataFrame),
  and the DataFrame of several frames had a repeated index: it is numbered
  `0..n-1`; `framenum` still says which frame a row came from.
* `packaging`, which the Localizer imports, is a dependency of the `yolo`
  extra; the `example()` functions say that the example images are in the
  repository, not in an installed package.
* `CLAUDE.md` described the repository as it was before this branch.

* `Localizer.directory` and `Estimator.directory` were annotated as returning
  `str` but return a `Path`; `Localizer._model_path` was annotated `None`.
* The docstrings of the public classes are NumPy style; the `freeze_*` methods
  of `TorchEstimator` loop over the layers instead of the weights and biases.
* The repository URL in `pyproject.toml`, `CITATION.cff` and
  `CONTRIBUTING.md` is the actual remote, `laltman2/CATCH`.
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
  returns the class name (`'holo'` for the default model).
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
