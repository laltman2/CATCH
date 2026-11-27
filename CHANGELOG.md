# Changelog

All notable changes to CATCH are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [3.0.0] - unreleased

CATCH 3 keeps the interface (`CATCHapp.analyze` returns a DataFrame with one
row per particle and the columns of CATCH 2) and replaces the models, the
training and the data.  On the frozen test set of 400 simulated scenes
(1381 particles), a fit from CATCH's estimate locks on for 67.8% of the
particles (72.8% from all five starts), against 15.0% for CATCH 2 and 76.8%
from the true particles; the detector finds 98.8% with a precision of 98.9%
(CATCH 2: 78.8%, 85.2%).

### Added

**The pipeline**

* `Localizer`: a heatmap detector after CenterNet (`networks.DetectorNet`)
  with the position of the center inside its cell, polished by the inversion
  symmetry of the hologram within a guard of 1 pixel.  Works on the pixels of
  the camera for pixel sizes of 1 to 2.5 reference pixels and resizes other
  frames into that range.
* `Estimator`: the radial profile of each hologram (`profiles.Profiler`,
  batched on the GPU), 64 candidates from `networks.ProfileNet`, each checked
  against the profile with a batched Lorenz-Mie model (`lorenzmie`), and the
  ranked starts of a fit, with starts in the standard range of the index or
  in a prior range (`nrange`).  A frame takes about 60 ms plus 2 to 3 ms for
  each particle in it, on the GPU of an Apple M1 Pro.
* `CATCHapp(instrument=...)`: results in the units of the microscope;
  `analyze(fit=True)` fits from every start (`fitter`, needs pylorenzmie) and
  chooses with `selection.select`, which reports the ambiguity
  (`nsolutions`, `a_spread`, `n_spread`, `z_spread`, the best alternative).
  `CATCHapp` and `Fitter` have `close()` and are context managers, which shut
  down the processes of the fits.
* `lorenzmie`: Mie coefficients vectorized over spheres, and the hologram on
  the axes of many spheres at once in torch (ported from pylorenzmie, whose
  author permits the use under the MIT license).
* `CATCHapp(rings=10, npixels=4000)` and `Fitter(npixels=...)`: the box and
  the crop of each fit reach the 10th bright ring of the particle's own
  hologram (the radius follows from its height), and each fit uses a random
  subset of `npixels` pixels of the crop.  There is no cap on the size of a
  crop but the frame.  On 717 simulated particles with neighbors and edges
  (the dev set), fits from five starts lock on for 77.4% with 10 rings,
  69.7% with 20 and 70.7% with 30; with 8,000 pixels, 80.6% (10 rings) and
  78.5% (20 rings).  In simulation the effect of the ring count is the
  density of the pixels in the crop; on 235 real isolated particles
  (1 and 3 um polystyrene and silica, 0.120 um per pixel) fits with 10 rings
  have fewer ambiguous solutions than fits with 20 rings, also at the same
  pixel density, except for 3 um polystyrene, which none of it changes.
* Models are chosen by name or by the path of a directory.

**Saturation and the quality of a particle**

* `CATCHapp` leaves the pixels that a camera saturated out of the profiles
  and the fits, and reports the `saturated` fraction of each box.
  `saturation=` takes a level (the brightest pixel that the camera records,
  in units of the background) or a boolean mask of the pixels to leave out,
  one for all the images or one for each; the default for an image of
  `uint8` is the mask of its pixels at 255, which is exact whatever the
  background.  After the division by a background that is not uniform the
  saturated pixels are not at one level: with a background varying by 25% to
  either side, the fits lock on for 73% of the particles with the mask, 65%
  with a level of 2.55 and 55% without masking.
* The default Estimator is fine-tuned for saturated profiles (10,000 steps on
  profiles of which half are as a saturating 8-bit camera records them).  On
  the dev set recorded by an 8-bit camera the full pipeline locks on 68% of
  the time (55% before), and 64% with the camera's clipping seen by the fits
  too (47% before).
* `Estimator.signal_to_noise` (the energy of the hologram in the box over the
  noise, from the profile, counting the measured pixels only) and the columns
  `snr` and `quality` of `CATCHapp`: `'low snr'` below 100, `'saturated'`
  above 5% of the box (`LOW_SNR`, `MAX_SATURATED`).  Calibrated with
  `scene_run` on scenes of five noise levels and four clipping levels.  The
  fits tolerate 40% of the box saturated when the saturated pixels are left
  out, the starts of the Estimator do not past 5%.

**Input**

* A frame whose median is not near 1 is divided by its median (an 8-bit image
  with a background of 100, a 12-bit one with 1000, ...), which costs 0.8 ms
  and not 11 ms.  Pixels that are NaN or infinite are left out of the
  profiles and the fits (one used to make the Localizer find nothing,
  silently) and counted in `saturated`.  An array of shape (rows, columns, 3
  or 4) is one color image and not a stack.  The noise of a frame is
  estimated from every 8th row, with a floor of 0.1% of the background.  A
  frame takes about 80 ms, was 115.

**Training and evaluation** (`CATCH.customization`)

* `python -m CATCH.customization.training.localizer` and `.estimator`:
  training for a new instrument, or fine-tuning (`training.frames` makes the
  frames).  `training.estimator --saturated` makes part of the examples as a
  saturating camera records them, and `--levels LOW HIGH` draws the camera
  limit of each (default: 2.55, as for the shipped model).  A model trained
  with `--levels 1.33 4` gains 10 points of lock-on at the limits 1.3 and
  1.15 and loses 6 at 2.55, so the shipped model stays the one for the usual
  exposure; the Localizer needs no such training (99.0% recall at 2.55, 98.9%
  at 1.3).
* The evaluation harness (`scene_run`, `centers`, `paired`, `detect_images`):
  simulated scenes of several particles, matching to the truth, fits from
  the estimate (`--starts`, `--grid`, `--loss`, `--scan`), and the criterion
  that matters (the radius within 5 nm).  `--instrument WAVELENGTH
  MAGNIFICATION N_M` simulates another microscope, `--camera [LEVEL]` records
  the scenes as a camera does, and `--mask-saturated` leaves the saturated
  pixels out; the results have the `saturated` fraction of each particle.  A
  development set and a frozen test set, whose stored truth (package data)
  `sets.verify` checks, and the results on the frozen test set in
  `benchmarks/catch-3.0.0`.

**Models and package**

* The distribution is named `catch-python` (the package is still imported as
  `CATCH`).  A tag `vX.Y.Z` runs `publish.yml` (build, wait for the tests of
  that commit, publish to PyPI with trusted publishing, create the GitHub
  release) and `update-citation.yml`, as in pylorenzmie and QVideo.
* The shipped models have a model card (`models/<kind>/default/MODEL_CARD.md`:
  what the model does, its training data, how it was trained (the committed commands reproduce the
  Estimator: 67.1% lock-on from a run from scratch, 67.8% shipped), its evaluation, and its known
  limitations), and their configuration records the SHA-256 of the weights
  (`weights_sha256`; `CATCH.Localizer.weights_digest`), which a training run
  records for the weights it makes.  `Localizer` and `Estimator` check the
  weights against the hash when they load (`CATCH.Localizer.verify_weights`)
  and raise `ValueError` for changed, damaged or mismatched weights; a
  configuration without a hash is not checked.
* `py.typed`: the package ships its type annotations (PEP 561).
* The tutorial notebook (`tutorials/Using_CATCH.ipynb`), written for CATCH 3:
  the pipeline on a real hologram (with a false positive, an imaging
  artifact, that its confidence gives away), the stages one at a time, the
  radial profile, the fits and their ambiguity.

### Changed

* The repository is laid out as `lib/` (networks, profiles, the Lorenz-Mie
  model, the fitter, the choice among fits, instruments), `models/<kind>/<name>`
  (the trained models) and `customization/` (`training/` and `evaluation/`:
  what a user does to adapt CATCH to an instrument, not to analyze
  holograms), with the three classes at the top.  `instruments` and
  `selection` are `CATCH.lib.instruments` and `CATCH.lib.selection`.
* `z_p` is in pixels of the camera of the given instrument.  `bbox` is sized
  from the estimate.
* A fit that fails on arithmetic, a rejected start or a singular matrix
  (`FIT_FAILURES`) is NaN with a `RuntimeWarning`; any other error is no
  longer hidden (none occurred in 3,585 fits of the dev set).
* The extent of a hologram has one implementation (`evaluation.extent.Extent`,
  cached, ten times faster than the one that made a calculator for each call),
  the crop of a scene is `lib.fitter.crop_square`, and the reference
  instrument is one read-only constant, `lib.instruments.REFERENCE`.  The long
  functions of the harness, the training and the Estimator are in steps; the
  results, and the trained weights, are identical.
* Dependencies: `torchvision` is no longer needed; extras `fit` and `train`
  (pylorenzmie).

### Removed

* The YOLOv5 Localizer and the crop Estimator of CATCH 2 (`TorchEstimator`,
  `ParamScale`), their weights (`cfg_yolov5/`, the old `cfg_estimator`
  models), training scripts and configurations, the `yolo`, `gpu` and
  `wandb` extras; the learned `Refiner`; the single-crop harness
  (`evaluation.run`), `evaluation.equivalence`, `evaluation.diagnose` and
  `evaluation.cost`.  Use CATCH < 3 for the 2.x models.

### Fixed

* `DetectorNet` pads a frame whose sides are not multiples of 32 (by
  repeating its right and bottom edges) and trims its outputs.  Before, its
  decoder was misaligned for such a frame, whether from the camera or
  resized into the range of pixel sizes: centers were off by several pixels
  and particles were missed.  On simulated frames of 1000 x 1250 pixels at
  the reference pixel, recall rose from 0.69 to 0.91 and precision from 0.70
  to 0.97.  Frames whose sides are multiples of 32, such as 1024 x 1280, are
  analyzed as before.
* A `Profiler` keeps the sampling offsets of the last few scales only.
  Before, its memory grew with every new scale, about 10 MB of GPU memory
  for each: measuring a training library of a few thousand frames ran out of
  memory.
* A fit that starts below the index of the medium stays below it (down to an
  index of 0.95).  Before, every fit was held at least 0.02 above the index
  of the medium, so a particle of lower index could not be fitted.  The
  shipped Estimator proposes no such start; one has to be given to the fit.

### The way to 3.0

The 2.x code was first brought up to the standards of the other Grier-group
projects (pull request #2): `pyproject.toml` (packaging, ruff, pytest,
coverage and pyright settings), continuous integration for Python 3.11 to
3.13, Sphinx documentation, a new README, tests, and fixes to bugs that lint,
tests and review found.  An evaluation harness was written that judges an
estimator by whether a fit that starts from it locks on, with scenes of several
particles, a frozen test set, and a baseline of the original CATCH on it
(precision 85%, recall 79%, 15.0% of the fits lock on).  Most of that code was
replaced by the rewrite above; the harness and the baseline remain.  The
detailed log of that work is in the git history of the `catch3` branch.
