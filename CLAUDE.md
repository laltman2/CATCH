# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CATCH (Characterizing And Tracking Colloids Holographically) is a two-stage ML pipeline for in-line holographic microscopy images: a YOLOv5 **Localizer** finds holograms of colloidal particles, then a PyTorch CNN **Estimator** infers each particle's `z_p`, `a_p`, `n_p` from its cropped hologram. Models are trained on synthetic data generated with Lorenz-Mie theory (`pylorenzmie`).

There is no build system, test suite, `requirements.txt` or linter config (see `ToDo.md`). Dependencies per README: pytorch, yolov5, numpy, cv2, matplotlib, pandas; optional cupy, wandb, pylorenzmie (needed for training/eval).

## Package layout quirk

The repo root is itself the Python package `CATCH` (it has a root `__init__.py`, and all internal imports are `from CATCH.xxx import ...`). The checkout directory must therefore be importable as `CATCH` (e.g. its parent on `PYTHONPATH`, or symlinked/installed under that name). Scripts are run with the parent directory on the path, e.g. `python -m CATCH.CATCHapp` (runs `example()` on `examples/test_image_large.png`); `Localizer.py` and `Estimator.py` also have `example()` mains.

## Architecture

- `CATCHapp.py` — `CATCHapp(lmodel, emodel)`: `analyze(images)` runs Localizer on all images, crops each detected feature (`crop`, a square of side `max(w, h)` around `x_p, y_p`, shifted inward when `edge` is True), feeds crops to Estimator, and concatenates features + estimates into one DataFrame with a `framenum` column.
- `Localizer.py` — subclasses `yolov5.YOLOv5`; **requires `yolov5 <= 6.0.7`** (import raises otherwise). Loads `cfg_yolov5/<model>/weights.pt` (default `small_noisy`). Images with max < 100 are multiplied by 100 (normalized holograms are expected to have mean ≈ 100 in 8-bit scale). Output DataFrame columns: `label, confidence, x_p, y_p, bbox, edge`; `find_center` shifts centers for boxes crossing image edges.
- `Estimator.py` — subclasses `TorchEstimator` (`TorchEstimator.py`). Loads `cfg_estimator/<model>/weights.pt` + `configuration.json` (default `default`); the config supplies the input `shape` and parameter scaling via `ParamScale.py`. Crops are expected square, are normalized to mean 100, resized, and outputs are un-normalized to physical units. `import CATCH` loads only inference code: `Localizer` and `CATCHapp` load lazily (they need the optional `yolov5`), and nothing under `training/` or `utilities/` is imported (a test checks this).
- `training/` — data generation and training scripts, driven by JSON configs (`torch_train_config.json`, `yolov5_train_config.json`). See `training/HowToTrain.md` for config options. Run from within `training/`: `python3 train_yolov5.py` (Localizer) and `python3 EstimatorTraining.py` (Estimator). `Classify.py` defines class labels for multi-class Localizers; `EarlyStopping.py` implements auto-training (`epochs: 0`).
- `cfg_yolov5/`, `cfg_estimator/` — trained model weights and configs. A model is either a directory (`<name>/weights.pt` [+ `configuration.json`]) or older flat `<name>.json` + `<name>_checkpoints/` files; the loaders only handle the directory form.
- `docs/images/` — the example holograms used by the `example()` functions and the tutorial, and an old evaluation plot. Media files live only under `docs/`.
- `examples/`, `tutorials/` — example scripts and notebooks (`tutorials/Using_CATCH.ipynb`); not part of the library API.
- `utilities/mtd.py` — helpers such as `feature_extent`.

## Style and quality standards

CATCH follows the conventions of the other Grier-group projects (QVideo,
pylorenzmie; their `CLAUDE.md` files are the reference). Apply them to new
and modified code; do not reformat untouched code as a side effect.

- Single quotes for strings, including docstrings.
- Maximum line length is 79 characters (PEP 8).
- American English spelling everywhere: *color*, *normalize*, *center*,
  *gray*, *behavior*.
- NumPy-style docstrings (`Parameters`, `Returns`, `Raises`, `Notes`), with
  default values and units in `Parameters`. Lengths are in micrometers
  except pixel coordinates.
- `snake_case` for attributes (CATCH has no Qt classes, so the QVideo
  camelCase rule does not apply). Type-annotate public signatures.
- Linting is `ruff check .` with the rule set used by QVideo
  (`F`, `B`, `E501`, `SIM`, `RUF012`, `PLW2901`, `UP037`, `PERF`, `PIE`;
  ignore `SIM905`). Fix findings rather than silencing them; a class-level
  list or dict is a tuple or a `ClassVar`. Scratch directories named
  `devel/` are git-ignored and never committed.
- Tests use `unittest` run by `pytest`, with `tests/` as `testpaths`.
  Always mock hardware and heavy models (YOLOv5, trained weights), mark
  `if __name__ == '__main__':` guards `# pragma: no cover`, and keep test
  data under `tests/data/`. The group's coverage floor is 95%, set just
  below the measured value so that regressions are visible.
- Media files (PNG, JPG, GIF, AVI, MP4) belong only under `docs/`.
- Version lives in one place (`pyproject.toml` in the group's projects);
  CATCH currently has `version.py`, so do not duplicate the version
  elsewhere.

CATCH does not yet meet these standards: there is no `pyproject.toml`,
`tests/` directory, CI or docs, and tracked media and weights sit outside
`docs/`. Adding `pyproject.toml` (ruff, pytest, coverage config modeled on
pylorenzmie's) and a `tests/` directory is the natural first step.
