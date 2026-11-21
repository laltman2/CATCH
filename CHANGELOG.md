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
from the true particles; the
detector finds 98.8% with a precision of 98.9% (CATCH 2: 78.8%, 85.2%).

### Added

* `Localizer`: a heatmap detector after CenterNet (`networks.DetectorNet`)
  with the position of the center inside its cell, polished by the inversion
  symmetry of the hologram within a guard of 1 pixel.  Works on the pixels of
  the camera for pixel sizes of 1 to 2.5 reference pixels and resizes other
  frames into that range.
* `Estimator`: the radial profile of each hologram (`profiles.Profiler`,
  batched on the GPU), 64 candidates from `networks.ProfileNet`, each checked
  against the profile with a batched Lorenz-Mie model (`lorenzmie`), and the
  ranked starts of a fit, with starts in the standard range of the index or
  in a prior range (`nrange`).  About 7 to 11 ms per particle on an Apple
  M-series GPU in frames of ten or more particles.
* `CATCHapp(instrument=...)`: results in the units of the microscope;
  `analyze(fit=True)` fits from every start (`fitter`, needs pylorenzmie) and
  chooses with `selection.select`, which reports the ambiguity
  (`nsolutions`, `a_spread`, `n_spread`, `z_spread`, the best alternative).
* `lorenzmie`: Mie coefficients vectorized over spheres, and the hologram on
  the axes of many spheres at once in torch (ported from pylorenzmie, whose
  author permits the use under the MIT license).
* Training commands for new instruments or fine-tuning:
  `python -m CATCH.customization.training.localizer` and
  `python -m CATCH.customization.training.estimator` (`training.frames` makes the frames).
* Models are chosen by name or by the path of a directory.
* The evaluation harness simulates any instrument: `--instrument WAVELENGTH MAGNIFICATION N_M` for `scene_run`, `centers` and `detect_images` (and `--lmodel`); the scenes, the fits, the scan and the crops use it.  The Estimator drops candidates beyond the bounds of the fits before checking them (an absurd candidate made the Mie coefficients run for ever).
* `CATCHapp` leaves the pixels that a camera saturated out of the profiles and the fits (`saturation=`, 255 of 100 for an image of `uint8`) and reports the `saturated` fraction of each box.  The default Estimator is the shipped one fine-tuned for 10,000 steps on profiles of which half are as a saturating camera records them: on the dev set recorded by an 8-bit camera, the full pipeline locks on 68% of the time (it was 55%), and 64% with the camera's clipping seen by the fits too (47% before).
* `saturation=` takes a boolean mask of the pixels to leave out (one for all the images, or one for each) as well as a level, and the default for an image of `uint8` is the mask of its pixels at 255 (exact, whatever the background).  After the division by a background that is not uniform the saturated pixels are not at one level: with a background varying by 25% to either side, the fits lock on for 73% of the particles with the mask, 65% with a level of 2.55 and 55% without masking.
* Signal-to-noise: `Estimator.signal_to_noise` (the energy of the hologram in the box over the noise, from the profile, counting the measured pixels only) and the columns `snr` and `quality` of `CATCHapp` (`'low snr'` below 100, `'saturated'` above 5% of the box; `LOW_SNR`, `MAX_SATURATED`), calibrated with `scene_run` on scenes of five noise levels and four clipping levels.
* Saturation: `scene_run --camera` records the scenes as an 8-bit camera does, `--mask-saturated` leaves the saturated pixels out of the fits (`fit(..., saturation=)`) and of the profiles of the Estimator (`Estimator(..., saturation=)`; the profiler leaves out NaN pixels), and the particles have a `saturated` fraction.  A fit can ignore the saturated pixels at no cost.  The Estimator is trained for them with `training.estimator --saturated` (half of the examples by default): fine-tuning the shipped weights for 10,000 steps (about 75 minutes) restores the oracle lock-on on clipped scenes from 50% to 69%, as on unclipped ones.
* The stored truth of the evaluation sets (`customization/evaluation/sets/*.csv`) is package data, so `sets.verify` works in an installed package too.
* `py.typed`: the package ships its type annotations (PEP 561), so that type checkers use them in code that uses CATCH.
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
* `customization/evaluation/baselines/catch-3.0.0`: the results on the frozen test set.
* The tutorial notebook (`tutorials/Using_CATCH.ipynb`) is rewritten for CATCH 3: the pipeline on a real hologram (with a false positive, an imaging artifact, that its confidence gives away and the threshold of the Localizer removes), the stages one at a time, the radial profile, the fits and their ambiguity.

### Changed

* `training/` and `evaluation/` are subpackages of `customization/` (`CATCH.customization.training`, `CATCH.customization.evaluation`): what a user does to adapt CATCH to an instrument, not to analyze holograms.
* `z_p` is in pixels of the camera of the given instrument; images may be
  normalized to 1 or to 100.  `bbox` is sized from the estimate.
* The fit procedure of the harness lives in `CATCH.lib.fitter`
  (`evaluation.fitting` re-exports it); `pipelines.estimator_on_truth` gives
  the image and the true centers to the Estimator, and `crops_on_truth`
  keeps the crops for the references of pylorenzmie.
* Dependencies: `torchvision` is no longer needed; extras `fit` and `train`
  (pylorenzmie).

### Removed

* The YOLOv5 Localizer and the crop Estimator of CATCH 2 (`TorchEstimator`,
  `ParamScale`), their weights (`cfg_yolov5/`, the old `cfg_estimator`
  models), training scripts and configurations, the `yolo`, `gpu` and
  `wandb` extras; the learned `Refiner`; the single-crop harness
  (`evaluation.run`), `evaluation.equivalence`, `evaluation.diagnose` and
  `evaluation.cost`.  Use CATCH < 3 for the 2.x models.

### Earlier entries of the 3.0 work

Some of what follows was removed again before the release (see above).

### Added

* The yardstick of the harness: `close` (the radius within 1%, the criterion
  that takes the size of the particle into account) next to `ok` (5 nm) and
  the table by radius in `scene_run`; `redchi0`, the reduced chi-squared of
  the start before the fit, as a measure of its quality that does not depend
  on the fit; `evaluation.paired` and `python -m CATCH.evaluation.paired A.csv
  B.csv` compare two versions on the same particles (McNemar's exact test and
  the interval of the difference).
* Fits from several starts (`scene_run --starts N`): the estimate and radii
  scaled along the degeneracy of the radius and the index; the best by
  chi-squared is the fit.  The starts of a particle run in separate
  processes, which took the 10 scenes of a test from 100 to 31 seconds (one
  particle of 4.5 um had taken 90 seconds by itself).
* `training/synth.py`: synthetic training data made on the fly from a seed,
  not read from disk: frames of 0-10 particles with edges, corners and
  overlaps (`Distribution`), crops as `CATCHapp` makes them with a
  Localizer-like error of the center (`CropPolicy`, `pad` or `shift` at the
  edge) and the stream of them (`Stream`).  The preprocessing is that of
  `Estimator._load_image` (tested), the seeds of the evaluation sets are
  refused, and the axial position is sampled at least `1.2 a_p / magnification`
  from the plane of the image, because the Lorenz-Mie field diverges inside
  the sphere (the old generator drew 2.7% of its particles there).  Frames
  take 13 ms (768 x 768) with the torch backend.
* `training/loop.py`: the training loop of the Estimator on that data
  (`python -m CATCH.training.loop`): worker processes, checkpoints evenly
  spaced in the logarithm of the examples, the cost log
  (`evaluation.cost.TrainingLog`), a fixed validation set and optional fits
  from the estimates.  `CropPolicy.repeats` makes several crops of each
  particle of a frame.  The axial position of the generated particles
  starts at 104 pixels (5 um), as in the instrument.
* `ParamScale` can scale the radius by its logarithm and the index by the
  logarithm of its contrast `n_p - n_m` (`config['scaling']`, absent for the
  shipped network, which is linear), so that relative errors weigh the same
  at every size; `training.loop --scaling log`.  The validation also reports
  the error of the contrast (`c_med`).
* `training.loop --init CHECKPOINT` resumes the training of a network with a new
  schedule of the learning rate; it refuses a checkpoint trained with another
  scaling or ranges.
* `training.loop`: a running average of the weights (`--ema`, 0.999) that
  the checkpoints keep and score (the raw weights are kept under `raw`),
  the loss as an option (`--loss smooth_l1|l1|mse`), and the fraction of
  good estimates by radius (`good_r1` to `good_r4`).  `scene_run` and
  `diagnose` take `--weights` for the checkpoint of a run.
* `evaluation.centers` (`python -m CATCH.evaluation.centers`): the measurement
  of the centers that a Localizer finds, by distance to the edge, with the
  boxes that the border cuts.  The correction of `find_center` is right for a
  box cut on one axis (median error 2.9 px) and wrong at a corner (47 px): 62%
  of the particles within 100 px of an edge are found within 15 px, 84% within
  100 px.
* `evaluation.diagnose` (`python -m CATCH.evaluation.diagnose`): the estimates
  for the same particles in the crop of the scene, padded, alone, and alone at
  the center, to tell neighbors from edges from the network itself.  On the
  dev set the network is accurate on clean crops (median `z_p` error 2.6%) and
  the shift of the crop at an edge, which leaves the particle off center, is
  the largest cause of the errors in scenes.  `pipelines.pad_around` and
  `scene_run --pad` give the network crops that are centered and padded with
  background: the fits that start from it lock on 27.2% instead of 17.4%.
* The loss of the fits (`scene_run --loss linear|cauchy|both`, `fitting.LOSSES`):
  least squares, the robust Cauchy loss, or least squares and then Cauchy.
  The Cauchy loss raises the ceiling (the true particle as the start) from
  73.6% to 78.9%, above all near neighbors, but does worse from a poor
  start; both in turn is the best from the network's estimate (19.1% against
  16.9%), which is the default of `scene_run` (`fitting.SCENE_LOSS`; the
  single-particle `run.py` keeps least squares).  Baselines in
  `evaluation/baselines/catch-2.0.0/oracle/`.
  The 25 starts of the grid with this loss lock on 28.5% (25.0% with least
  squares).
* Measured: the torch backend of pylorenzmie (the Metal kernel) fits the dev
  scenes from the truth in 34 s instead of 78 s with Numba (8 processes)
  with the same lock-on rate (77.4% and 76.3%, p = 0.23); set
  `PYLORENZMIE_BACKEND=torch`.
* `evaluation.references`, `scene_run --estimator analytic|de`: the analytic
  and the differential-evolution estimators of pylorenzmie as starts that
  need no network.
* `evaluation/baselines/catch-2.0.0/oracle/`: the shipped network from the
  true centers, with one start and with nine, and the analytic start, on the
  dev set (16.9%, 22.6% and 14.8% lock on), with 25 starts that also vary
  `z_p` (`scene_run --grid`, 25.0%), and the true particles as the start
  (73.6%, the ceiling).

* `CATCH.lib.instruments` and `Estimator(instrument=...)`: use a network with
  another wavelength, magnification or medium.  The hologram depends on the
  instrument only through the pixel size in wavelengths of the medium, so
  the crop is described in pixels of the network and the estimates are
  converted back (`z_p` in pixels of the instrument, `a_p` in proportion to
  the wavelength in the medium, `n_p` at the same relative index).
  `python -m CATCH.evaluation.equivalence` checks it: on 180 particles the
  shipped network has the same errors on two other instruments as on its
  own (median error of the radius 5.4%, 5.4%, 5.3%).

* A frozen test set and a development set of scenes
  (`CATCH.evaluation.sets`, `scene_run --set`), with the truth about their
  particles stored in `evaluation/sets/` and checked by `sets.verify` (and a
  test), so that a change to the code that makes the scenes is noticed.
* The cost of training as a metric (`CATCH.evaluation.cost`): a log of the
  time, epochs and examples of a run, the cost of reaching a target of the
  lock-on rate or a loss, the lock-on rate of saved checkpoints
  (`Estimator(weights=..., config=...)`), and the description of the
  machine.  `EstimatorTraining` writes the log and takes a `max_epochs`.

* The comparison of versions of CATCH: `scene_run --write-images` writes the
  images of the scenes, `detect_images.py` runs the `CATCHapp` of any checkout
  on them (the checkouts are all called `CATCH`, so it runs in a process of
  its own), and `scene_run --pipeline recorded` scores what it found.  The
  baseline of the original CATCH on 200 scenes is in
  `evaluation/baselines/catch-2.0.0/`: precision 85%, recall 79%, and a fit
  locks on for 15.6% of the particles.  The current code gives identical
  detections.

* Scenes with several particles in the evaluation harness
  (`CATCH.evaluation.scenes`, `matching`, `pipelines`, `scene_run`;
  `python -m CATCH.evaluation.scene_run`): simulated scenes whose fields add,
  matching of detections to the truth, a fit of each matched particle on a
  crop of the scene, and summaries by the number of particles, the distance
  to the edge and the distance to the nearest neighbor.  Baselines: a fit
  that starts from the true values locks on 97% of the time for one particle
  and 73% for 1 to 6 (25-28% with a neighbor within 150 pixels); the default
  Estimator, given the true centers and crop sizes, 30% and 23% (0% within 25
  pixels of an edge).

* `CATCH.evaluation`, the harness that judges an estimator by whether a fit
  that starts from it locks on (the radius within 5 nm), on simulated crops
  of 180 particles, with `python -m CATCH.evaluation.run`; it is the baseline
  of the pylorenzmie handoff, ported from its scratch scripts and tested
  (`tests/test_evaluation.py`).  It reproduces the baseline of the default
  network exactly (median errors 3.7%, 3.4% and 1.1%, 58% of the fits lock
  on) and the page `docs/evaluation.rst`.

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

* `evaluate_scenes` keeps the images of all the scenes and the crops for all
  their fits in memory, which grows with the number of scenes; it fits `chunk`
  scenes at a time (`--chunk`, 50 by default on the command line) and numbers
  the scenes by their own `index`.
* `EarlyStopping` counted a loss that equals the best so far as an
  improvement, so a plateau that is exactly flat never ran out of patience; an
  improvement has to be strictly better by `delta`.
* `EstimatorTraining`: the automatic training had a typo
  (`self.trainloder`) and the test loss of the manual training was computed
  only on a GPU.  It also records the cost of the run (see
  `CATCH.evaluation.cost`) and can be given a budget (`max_epochs`); it
  stops only when all three outputs have stopped improving, which takes
  hundreds of epochs.

* The Localizer could not load its weights with PyTorch 2.6 or later, which
  unpickles only the weights by default, and the checkpoints of YOLOv5 6.0.x
  hold a whole model; it loads its own, trusted, weights with
  `weights_only=False` (`trusted_checkpoints`).

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
* `Localizer.find_center` unpacked three values from the shape of the image,
  so it raised `ValueError` for the shape of a grayscale (2-D) image.  The
  Localizer worked on grayscale images anyway, because yolov5 turns the images
  into three channels in place; `find_center` takes the first two numbers of
  the shape.
* `Estimator` compared a `torch.device` with the string `'cpu'`, so it always
  moved the model and the images to the device; it compares the device type.
* `from CATCH import Localizer` could return the submodule instead of the
  class once `CATCH.CATCHapp` had been imported; the lazy loader binds the
  classes after it imports them.

### Removed

* `eval/` and `experiments/`: scripts that imported APIs that no longer exist
  (`CATCH.CATCHobject`, `LMHologram`) and could not run.  They remain in the
  git history; the rebuild writes new evaluation code.
