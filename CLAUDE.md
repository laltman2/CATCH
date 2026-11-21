# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CATCH (Characterizing And Tracking Colloids Holographically) finds colloidal particles in in-line holographic microscopy images and proposes starts for Lorenz-Mie fits of `z_p`, `a_p`, `n_p` good enough to lock on. Version 3: a heatmap **Localizer** (CenterNet-style, with sub-cell offsets) and a profile **Estimator** (radial profile -> 64 network candidates -> each checked against the profile with a batched GPU Lorenz-Mie model -> ranked starts); `CATCHapp.analyze(fit=True)` also fits from every start (pylorenzmie) and chooses with `selection.select`. Networks work in pixels of a reference instrument (0.447 um, 0.048 um/px, n_m 1.34) and serve instruments whose pixel is 1-2.5 reference pixels (`lib.instruments.pixel_ratio`). Trained on synthetic data.

`pyproject.toml` holds the packaging, ruff, pytest, coverage and pyright settings; extras are `fit` (pylorenzmie), `train` (pylorenzmie), `dev` and `docs`. Tests are in `tests/` (run `python -m pytest --cov`), CI is `.github/workflows/test.yml` (ruff, actionlint, pyright, Sphinx, tests on Python 3.11-3.13), and the documentation is in `docs/`. CATCH 2 (YOLOv5 Localizer, crop Estimator) was removed in 3.0; its baseline numbers are in `evaluation/baselines/catch-2.0.0`.

## Package layout quirk

The repo root is itself the Python package `CATCH` (it has a root `__init__.py`, and all internal imports are `from CATCH.xxx import ...`). The checkout directory must therefore be importable as `CATCH` (e.g. its parent on `PYTHONPATH`, or symlinked/installed under that name). Scripts are run with the parent directory on the path, e.g. `python -m CATCH.CATCHapp` (runs `example()` on `docs/images/test_image_large.png`, so only from a checkout). Pyright and CI resolve `CATCH` through the parent of the checkout, so the checkout directory must be called `CATCH` (`extraPaths = [".."]`). `CATCH` exports the classes `CATCHapp`, `Localizer` and `Estimator`, which shadow their modules as attributes: get a module with `importlib.import_module('CATCH.Localizer')` (the tests do).

## Architecture

- `CATCHapp.py` — `CATCHapp(lmodel, emodel, instrument, device, nprocs, compile, rings=10, npixels=4000)`: `analyze(images, fit=False, nrange=None, explore=0, noise=None)` returns one row per particle: `label, confidence, x_p, y_p, bbox, edge, z_p, a_p, n_p, framenum, misfit, starts` (+ `selection.COLUMNS` with `fit=True`). `bbox` and the crops of the fits are squares of twice the radius of the `rings`-th (default 10) bright ring of the best start (CATCH 2 used about 9; fewer rings make fits less sensitive to background errors, aberrations and neighbors, see the CHANGELOG); the fit uses a random subset of `npixels` pixels of its crop; `z_p` in camera pixels. Images: background 1, or 100 (divided by 100).
- `Localizer.py` — `Localizer(model, device)`: `networks.DetectorNet` peaks above `threshold`, then `symmetry_center` (masked FFT autocorrelation, 256 px) kept only within `guard` (1 px) of the detector's center; frames outside the trained `scales` are resized. Also `normalize`, `default_device`, `model_directory` (a model is a name in `cfg_<kind>/` or a directory path).
- `Estimator.py` — `Estimator(model, device, compile)`: `profiles.Profiler` (polar sampling, nearest pixel, nanmedian) -> `networks.ProfileNet` candidates -> `misfits` (robust, every 4th radius, `lorenzmie.RadialModel`) -> top 3 + best 2 in the preferred `n_p` range (standard <= 1.8, or a prior `nrange`; a scan of `a_p` x `n_p` supplies them if none; `explore` adds distinct-`a_p` starts). Outputs in the camera's units via `instruments.from_reference`.
- `lib/` — the internals: `networks.py` (both networks), `profiles.py`, `lorenzmie.py` (Mie coefficients vectorized in numpy, radial field in torch, one `torch.compile`d step per partial wave, phase factored out; float64 except on MPS; ported from pylorenzmie with its author's permission for MIT use), `selection.py` (the choice among fits and the ambiguity, a lower bound), `fitter.py` (pylorenzmie fits from starts; `evaluation.fitting` re-exports `fit`, `fit_particle`) and `instruments.py` (conversion between instruments, `REFERENCE`). `CATCH.instruments` and `CATCH.selection` are aliases of the `lib` modules, for now (set in `__init__.py`). Shapes are padded to fixed chunks so compiled kernels are not rebuilt.
- `models/localizer/`, `models/estimator/` — the trained models: `<name>/weights.pt` + `configuration.json` (the default is `default`).
- `training/` — `frames.py` (synthetic frames at a per-frame pixel scale, log-uniform heights, faint particles, imperfect normalization), `localizer.py` and `estimator.py` (`python -m CATCH.training.localizer|estimator OUT --steps N [--init default]`; write a model directory, resume from `state.pt`, log `eval.jsonl`), `synth.py` (`Distribution`, `Renderer`, `sample_particles`, `FORBIDDEN_SEEDS`). Set `PYLORENZMIE_BACKEND=torch` to render on a GPU.
- `evaluation/` — the scene harness (`scene_run.py`): pipelines `truth`, `oracle` (`estimator_on_truth`, the Estimator at true centers; `--estimator analytic|de` uses `crops_on_truth` with the pylorenzmie references), `catch` (`CATCHapp`), `recorded` (`detect_images.py` output); fits from the estimate (`--starts`, `--grid`, `--loss`, `--scan`), lock-on = radius within 5 nm. `sets.py` defines `dev` and the frozen `test` set (compare finished versions only; `sets.verify` is a test). `paired.py`, `centers.py`, `scan.py`, `matching.py`, `cases.py`, `extent.py` (`feature_extent`: the 20th zero crossing, about 10 bright rings, of a hologram). Baselines in `evaluation/baselines/`.
- `docs/images/` — example holograms. Media files live only under `docs/`.
- `tutorials/` — the notebook `tutorials/Using_CATCH.ipynb` (CATCH 2 era); not part of the library API.
- `docs/` — Sphinx sources (`conf.py` mocks torch, cv2 and pylorenzmie; built with warnings as errors).
- `devel/` — git-ignored research scripts and runs (the experiments behind 3.0).

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
  Always mock hardware and heavy models (trained weights: use
  `tests/random_models.py`), mark
  `if __name__ == '__main__':` guards `# pragma: no cover`, and keep test
  data under `tests/data/`. The group's coverage floor is 95%, set just
  below the measured value so that regressions are visible.
- Media files (PNG, JPG, GIF, AVI, MP4) belong only under `docs/`.
- Version lives in one place (`pyproject.toml` in the group's projects);
  CATCH currently has `version.py`, so do not duplicate the version
  elsewhere.
- Pyright (pinned in CI) checks the library and the tests; fix findings
  rather than silencing them.

Still open against these standards: there is no publish workflow.
