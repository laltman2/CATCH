# To Do

1. Migrate the Localizer from `yolov5 <= 6.0.7` to `ultralytics`.
   - On hold until the experiments C and E of the CATCH 3.0 plan decide the
     detector; a heatmap detector would be written in PyTorch and would drop
     the ultralytics dependency (and the AGPL question below) altogether.
   - Do it after the tests exist, and keep the `Localizer` output
     (`label, confidence, x_p, y_p, bbox, edge`) and the `bbox` convention
     used by `CATCHapp.crop` unchanged.
   - The YOLOv5 weights in `cfg_yolov5/` cannot be loaded by ultralytics, so
     the Localizer must be retrained; fold this into the training rebuild.
   - License decided: CATCH keeps its MIT license (an open-source community
     project that relies on YOLO), and the AGPL-3.0 of ultralytics is
     accepted.  Users who distribute or serve a combination of CATCH and
     ultralytics are bound by the AGPL for that combination.
   - Compare with the pylorenzmie Localizer before choosing a detector.
2. Fix yolov5 training script default options (moot after item 1).
3. Clean up imports; the dependencies are in `pyproject.toml` now.
4. Test `EstimatorTraining.py` and `train_yolov5.py`; they are omitted from
   coverage and have not been run since the port of data generation.
5. `examples/pylm_fit.py` imports modules that no longer exist
   (`CATCH.Estimator_Torch`, `LMHologram`, `coordinates`); port it to
   pylorenzmie 3.x or delete it.

# CATCH 3.0 plan

CATCH keeps its name and its interface: images in, a DataFrame out with one
row per particle (`label`, `confidence`, `x_p`, `y_p`, `bbox`, `edge`,
`z_p`, `a_p`, `n_p`, `framenum`), `CATCHapp.analyze` as the entry point, and
models chosen by name.  The networks, the training and the data change.  The
requirements are (1) detecting individual particles in crowded scenes,
(2) accurate centers, especially near the edges of the field of view, and
(3) estimates good enough for a full fit to lock on, especially for large,
high-index particles.  The measure of success is the fraction of fits that
lock on after the network (success = fitted `a_p` within 5 nm of the truth),
not RMSE; it is judged against the number of particles, their separation and
their distance to the edge.

## What the review of the 2.x networks found

Measured, with the default weights on crops sized from the true extent:

* Changing only the `scale` input by +5% moves the estimated `z_p` by 9.8%
  (median; `a_p` by 4.6%, `n_p` by 1.5%), against 2.8% error with the correct
  scale.  `z_p` is carried by the crop size (the crop side is twice the
  radius of the 20th fringe, which grows like the square root of `z_p`), so a
  box that is 5% off gives about 10% error in `z_p`.
* The Estimator has 25,755 parameters and about 120 M multiply-adds per crop,
  nearly all in the first two convolutions at near-full resolution.
* The Estimator is as accurate in relative terms for large spheres as for
  small ones (a_p error 2.3% for a_p >= 2.5 um), yet the success of a fit
  from its estimates falls with size (77, 97, 57, 53, 40, 27% for a_p of 0.5
  to 5 um, from the pylorenzmie harness): the basin of a large sphere is
  narrower, so the estimates must be tighter, which more data of the same
  kind does not give.
* Clipping at 2.55 times the background (8-bit storage) matters only for a
  few large, low-contrast spheres that focus (a_p = 5 um, n_p = 1.59:
  peak 85 times the background).

From the code:

* The optimizer is RMSprop at a fixed 1e-5 with no normalization layers and no
  schedule; the data are 10,000 fixed PNGs; auto-training stops after a
  patience of 300 epochs, which is why runs take thousands of epochs.
* `_train_auto` freezes the shared layers as soon as ANY head stops
  improving, so the other heads can no longer improve the representation.
* `EstimatorTraining.py` has a typo in `_train_auto` (`self.trainloder`) and,
  in `_train_manual`, a test loop nested under `if device != 'cpu'`, so the
  test loss is always 0 on a CPU.  Fix both before using it.
* YOLO boxes (twice the extent, hundreds of pixels) overlap heavily in a
  crowded scene, and the center of a truncated feature is derived from the
  box.

## Design intent to keep or test: the narrow "address" layer

The 401-to-20 dense layer is deliberate: the 20-unit layer holds about as
much information as is needed to address a point of the (a_p, n_p, z_p) space
at the desired resolution, and three independent regression heads read the
address out as `z_p`, `a_p` and `n_p`.  Resolving each parameter to 1%
(logarithmically; n_p - n_m for the index) takes about 25 bits (about 250 x
320 x 340 cells), which 20 continuous units can hold, so the size of the
layer is not the limit.  What the experiments below should test is whether
the limit is upstream of it (three max-pooling steps and a final 4 x 4
max-pool keep which features are present but discard where the fringes are,
which is what the parameters are encoded in), in the narrow heads (20 to 20,
40 and 100 units), or in the crop-size coupling.  See experiment D.

## Order of work

0. Merge the packaging PR and the port of data generation; tag the last 2.x
   release (pretrained models plus bug fixes) so that published results can
   be reproduced.
1. Evaluation harness.  Done: the single-particle harness
   (`CATCH.evaluation.run`, ported from `pylorenzmie/devel/`, reproducing the
   baseline of the default network) and the harness for scenes with several
   particles (`CATCH.evaluation.scene_run`: detection, matching, a fit of
   each match, results by the number of particles, the distance to the edge
   and the distance to the neighbor).  Baselines (in `docs/evaluation.rst`):
   a fit from the true values locks on 97% of the time for one particle but
   73% for 1 to 6, and 25-28% when a neighbor is within 150 pixels, so a fit
   of one particle on its crop cannot be the end of the pipeline (the fit of
   the neighbors together is part of experiment E); the default Estimator on
   the true centers and crop sizes locks on 63% in the interior of the image
   and 0-23% within 250 pixels of an edge.  Still to do: (a) done:
   the baseline of the original CATCH, with the Localizer, is in
   `evaluation/baselines/catch-2.0.0/` (precision 85%, recall 79%, 15.6% of
   the fits lock on for `dev`, 15.0% for the frozen `test`); every new version is to be compared with it; (b) the starting points that need no
   network (the analytic estimator and the differential-evolution estimator
   of pylorenzmie, which locked on 35%) as references (done for the analytic
   start, `scene_run --estimator analytic|de`: from the true centers on the
   dev scenes it locks on 14.8% against 16.9% for the network, not a
   significant difference, p = 0.21; the DE search takes seconds per
   particle and has not been run on the set); (c) scenes with
   particles outside the image (`margin`), where a truncated feature has its
   center out of view; (d) a joint fit of the particles in a scene as the
   second measure of success; (e) stress sets: scenes with variations of the
   instrument (aberrations, the level and the statistics of the noise,
   errors of the normalization, other wavelengths, magnifications and media),
   to measure how a version holds up across instruments.  Synthetic data are
   the strength of CATCH, because experiments cannot cover the range of
   particles and instruments, so this is what to measure and what to train
   against (domain randomization); a few real holograms of known particles
   serve only to check that the simulator describes an instrument, and are
   neither training data nor a test of coverage.  Fix the two bugs in
   `EstimatorTraining.py`.
1a. Metrics (done before the experiments): a frozen test set and a
   development set (`evaluation/sets.py`), and the cost of training
   (`evaluation/cost.py`: seconds and examples to reach a target, measured on
   a stated machine).  Done: `close` (radius within 1%) and the table by
   radius in the scene tables; paired comparisons of two versions on the same
   particles (`evaluation/paired.py`); `redchi0`, the quality of the start
   that does not depend on the fit.  Still missing: the stress sets of the
   instrument (item 1, (e)).  What the yardstick showed, on the dev scenes
   from the true centers (`evaluation/baselines/catch-2.0.0/oracle/`): fits
   from nine starts along the degeneracy of the radius and the index lock on
   22.6% against 16.9% from one (+5.7 points, p = 9e-13), and the best
   choice among the nine would get 23.8%, so more starts along that
   direction are worth little.  The error of `z_p` decides: half of the
   estimates are off by more than 20% (4% for one particle in a clean crop),
   because the crop holds the fringes of neighbors and is shifted at the
   edges; with `z_p` and `a_p` within 5% 71% lock on (79% with no neighbor
   within 400 pixels), otherwise 15%.  So the effort belongs to the estimate
   in a crowded crop (the refiner, experiment B, and estimators that are
   trained on crops with neighbors and edges), and starts that
   also vary `z_p` (a grid of 25: five radii by five axial positions) lock
   on 25.0% against 16.9% from one start (+8.1 points, p = 7e-18) and 22.6%
   from nine radii, while the true particle as the start locks on 73.6% (90%
   when isolated; 45% for radii under 1 um, 87% above 4 um, independent of the
   distance to the edge): the ceiling of the setup is far above anything
   that more starts reach (the best choice among the 25 would get 29.8%).
   The network's start does the opposite of the ceiling with the radius
   (25% under 1 um, 10% above 3 um), so the estimate of large particles in
   crowded and edge crops is the target.  The loss of the fit matters too:
   with the robust Cauchy loss (`--loss cauchy`) the true particle locks on
   78.9% against 73.6% (17% to 58% with a neighbor under 50 pixels), but from
   the network's estimate it loses (14.4% against 16.9%); least squares then
   Cauchy (`--loss both`) gets 19.1% from the estimate and 76.3% from the
   truth.  `both` is the default of `scene_run`; the earlier baselines use
   least squares (`--loss linear`).  The grid of 25 starts with `both` locks on 28.5%
   (one start: 19.1%; best possible choice 33.1%; the ceiling 76.3%), the
   baseline for the new estimator.  Open: the scale of the Cauchy loss.
   Diagnosis of the estimate (`evaluation/diagnose.py`,
   `evaluation/baselines/catch-2.0.0/diagnosis/`): the weights are not the
   problem.  On clean centered crops the network is accurate (median `z_p`
   error 2.6%, 98% within 10%); on the scene crops it is 21% (37% within 10%).
   The largest effect is the shift of the crop at an edge, which leaves the
   particle off center (67% of the dev particles; 27% within 10% for the
   particle alone), then the fringes of neighbors (99% to 54% in unshifted
   crops).  Padding the crop with background, keeping the particle centered
   (`scene_run --pad`), lifts the lock-on from the estimate from 17.4% to 27.2%
   (+9.8 points, p = 2e-14) with no retraining.  So: change what
   `CATCHapp.crop` does at an edge (a change of behavior: decide), and train
   with neighbors and truncated crops (item 3).
   The centers of the Localizer at the edges (`evaluation/centers.py`):
   `find_center` is right for boxes cut on one axis (median error 2.9 px
   against 55 px for the middle of the box) and wrong at corners (76 of 397
   cut boxes, median error 47 px), where `max(w, h)` underestimates the size;
   that is about 22 points of recall within 100 px of an edge (62% to 84% with a
   right center).  16% of the particles there are not detected at all.  This
   is the target of the CenterNet alternative (item 6): it needs a center and
   a size head and training frames with particles at the edges and corners.
   Two fixes of `find_center` at corners were tried on the dev set (a prior
   on the size, `max(w, h) / 0.84`, and the estimator's extent): the center
   is within 15 px for 24-27% of the corner particles against 10.8%, about 2
   points of all particles, and with the true size it would be 58%, because
   the inner edges of a cut box scatter by 15 px.  Not shipped; the
   detector that reads the center from the fringes is the fix.
   Backend of the fits (Apple-silicon laptop, `PYLORENZMIE_BACKEND`; the
   torch backend is the Metal kernel, there is no torch CPU calculator): one
   hologram of 201 x 201 pixels and a = 2 um takes 37.6 ms with NumPy, 33 ms
   with JAX (on the CPU), 10.2 ms with Numba on one thread, 2.3 ms with Numba
   on all cores and 0.74 ms with torch (0.96 ms for 5 um); at 4000 pixels, as
   in the fits, 4.6, 3.7, 1.1, 0.39 and 0.46 ms.  Fits of 4000 pixels per
   second, 8 processes (as the harness runs them), Numba against torch: 374
   against 314 for 0.5 um, 79 against 153 for 2 um and 89 against 137 for
   5 um; one process, 11 against 33 and 13 against 24 for 2 and 5 um, and
   JAX loses 0.7 s to compiling for each fit.  So no backend is best for every
   workload: Numba for small spheres, torch for large ones and large images.
   The truth start on the dev scenes with `PYLORENZMIE_BACKEND=torch` takes
   34 s instead of 78 s and locks on 77.4% instead of 76.3% (p = 0.23: single
   precision moves 34 fits either way), so the harness can use it
   (`PYLORENZMIE_BACKEND=torch python -m CATCH.evaluation.scene_run ...`);
   for pylorenzmie, propose a benchmark helper that times the backends on a
   workload and a choice that depends on it, not a global default.
   The cost of the old recipe is about 10^8 examples (two runs on
   the same data trained for 10,000 and 9,428 epochs of 10,000 images; the
   history of the shipped model is not in the repository) and 74 examples/s on
   a laptop CPU (16 days), which every new recipe is to be compared with, at
   equal lock-on on the `dev` set.
2. Training infrastructure, shared by everything below.  Started:
   `training/synth.py` makes frames and the crops of them on the fly from a
   seed (0-10 particles anywhere, with `edge_bias` and `corner_bias` to put
   them at the edges and corners, overlapping holograms, noise drawn for each
   frame; the crops as `CATCHapp` makes them, with a Localizer-like error of
   the center and `pad` or `shift` at the edge), through the same
   preprocessing as `Estimator._load_image` (a test checks it), and refuses
   the seeds of the evaluation sets.  On the laptop with the torch backend a
   768 x 768 frame takes 13 ms (126 ms with Numba) and a 1024 x 1280 frame
   24 ms, and a stream makes about 170 crops per second in one process (the
   crops and the preprocessing are on the CPU now: the next bottleneck, and
   workers are the way).  Found on the way: the Lorenz-Mie model diverges when
   the plane of the image is inside the sphere, `z_p * magnification < a_p`
   (values of 1e20 and more in double precision, NaN in single), and the old
   generator draws `a_p` and `z_p` independently from their ranges, so 2.7%
   of its particles (4.3% with a clearance of 1.2) were of that kind; the
   new one draws `z_p` above `1.2 * a_p / magnification`.  This is not a
   fault of the torch backend: the NumPy, Numba and torch calculators
   agree to 1e-3 wherever the field is finite.  Still to build: the training
   loop (log-spaced checkpoints scored by lock-on, the cost log), workers,
   float inputs, a fixed validation set, and the detector's targets.
   The plan: generate data on the
   fly on the GPU (pylorenzmie's torch calculators) instead of reading PNGs;
   float inputs without clipping, with realistic noise and aberrations;
   sample `a_p` and `z_p` logarithmically and `n_p - n_m` in contrast;
   weight the sampling toward large, high-index particles; keep a fixed
   validation set; normalization layers, AdamW with a one-cycle schedule,
   mixed precision.  Target: tens of epochs, not thousands (a guess to
   check).  Measured on the Apple-silicon laptop (torch 2.15, pylorenzmie's
   `TorchLorenzMie`, which has a Metal kernel in single precision): one
   hologram takes 0.6-1.4 ms at 201 x 201 and 0.9-1.8 ms at 475 x 475 (the
   median training crop) on `mps`, against 1.6-5.5 ms and 6.7-26 ms with
   Numba on the CPU, 3-4x and 7-14x faster (17x at 1024 x 1024); the
   difference from the double-precision Numba hologram is at most 8e-4 of the
   background (rms 3e-5 of the signal), well below the noise of 0.02-0.05 and
   the 8-bit step, for a_p up to 5 um and z_p up to 600 pixels.  A training
   step of the shipped network with a batch of 64 takes 92 ms on `mps` and
   623 ms on the CPU, 692 against 103 examples/s.  The 29 frames/s of the
   old data generation is therefore mostly PNG writing, reading and resizing
   rather than the physics; generating on the GPU and resizing there should
   leave the training step as the limit.  Not measured: both on the one GPU
   at once, so the end-to-end rate is an estimate (several hundred
   examples/s, 10^8 examples in a few days rather than 16), a batch
   generator (the batch kernel is Triton, for CUDA and XPU), and
   accuracy for crops much larger than 1024 pixels.  A Mac is enough for
   the ablations; a CUDA machine would matter for the long runs.
3. Experiments, each with a baseline from step 1:
   * A. Representation.  Fixed-size crops at native pixel sampling (no
     resize, no `scale` input) with a validity mask for the region outside
     the field of view and pixel-coordinate channels, against a polar
     resampling about the center estimate and a 1D network on the azimuthal
     profile (compare pylorenzmie's `RadialEstimator` and `MLPEstimator`).
   * B. Refiner.  A second network that takes the data and the hologram of
     the current parameters and returns corrections to (x_p, y_p, log z_p,
     log a_p, n_p - n_m), trained from the truth plus a random perturbation
     of up to about 10% and applied two or three times.  Measure the success
     of the fit against the size of the starting error; if it locks on from
     5 to 10%, the single-shot network does not have to reach 1%.
   * C. Detector.  A center-point heatmap network (CenterNet style) on a
     padded canvas, with sub-pixel offsets and a coarse `z_p` and size for
     each peak, against Ultralytics YOLO, on synthetic scenes with up to many
     particles, overlaps and truncation.  Centers outside the field of view
     are allowed.
   * D. Heads and the address layer.  Ablate the width of the address layer
     (20, 64, 256), the max-pooling (strided or average convolutions instead)
     and the scalar `scale` input.  Compare the three regression heads
     against heads that classify into logarithmic bins (a soft-label
     cross-entropy over about 100 to 300 bins per parameter, read out by the
     expectation or the mode and sharpened by the refiner): it is the same
     "address" idea made explicit, and it gives a distribution rather than a
     single value, which matters because `a_p`/`n_p` and the distance to the
     focus are partly degenerate.  Train the heads jointly with the shared
     layers (do not freeze the shared layers when one head stops).
   * E. Scene-level approach.  A dense network on the whole frame (a light
     U-Net or FPN backbone with CenterNet-style heads, written in PyTorch:
     the original CenterNet is not maintained and the algorithm is small) that
     gives, at each center peak, the sub-pixel offsets and a coarse `z_p`,
     `a_p` and `n_p`; and a scene-level refiner that renders all the detected
     particles together (fields add, so a crop sees its neighbors only as
     contamination) and works on the whole-scene residual, or peels: detect,
     render, subtract, detect again.  Compare with YOLO plus crops on
     crowded synthetic scenes (overlaps, truncation, many particles) by
     detection precision and recall, the error of the center against the
     distance to the edge, and the success of the fit.  The receptive field
     must cover several hundred pixels without downsampling away the outer
     fringes (skip connections).
   * F. Physics inputs and several hypotheses.  (i) Back-propagate the
     hologram to a stack of planes at logarithmically spaced `z_p` and give
     the slices to the network as extra channels, so that a scatterer is a
     compact spot at its depth (pad for the FFT; large spheres, a_p near
     5 um or about 100 px, may focus poorly).  (ii) Take the top k modes of
     the bin-classification heads of D, run the fit from each, keep the one
     with the best reduced chi-squared, and measure the success against k.
   * G. Instruments.  The shipped models are for one instrument (wavelength
     0.447 um, 0.048 um per pixel, a medium of index 1.34).  Because the data
     are synthetic, a model can be trained across instruments, but most of
     the variation need not be learned:
     (i) Wavelength, magnification and medium are one change of scale.  A
     hologram in pixels depends on the instrument only through the pixel
     size in wavelengths of the medium, and on the particle through
     `a_p/lambda`, `z_p*mpp/lambda` and `n_p/n_m`, so one network serves all
     instruments if the crop is described in units of the network's pixel
     and the outputs are converted (`CATCH.instruments`,
     `Estimator(instrument=...)`).  DONE and measured
     (`python -m CATCH.evaluation.equivalence`): for 180 particles, the
     errors of the shipped network on an instrument with finer pixels in
     another medium (0.532 um, 0.03 um/px, n_m 1.33) and on one with
     coarser pixels in air (0.532 um, 0.10 um/px, n_m 1.0) match those on
     its own instrument (median |da/a| 0.054, 0.054, 0.053; radii within 5%:
     46%, 46%, 46%).  Limits: coarse pixels undersample the outer fringes
     (information is lost, not rescaled); the particle must fall in the
     ranges of `a_p/lambda` and `z_p/lambda` of the training; dispersion of
     `n_p` and `n_m` is not included.  So wavelength and magnification are
     not stress sets, and are not trained as inputs.
     (ii) Aberrations are not a change of scale.  Defocus and tilt are
     offsets of `z_p` and `x_p, y_p` that a calibration removes; spherical
     aberration biases the estimates along the degenerate direction of
     `a_p` and `n_p` (positive: small particles seem smaller and of higher
     index, negative the opposite), depending on `a_p`, `n_p` and `z_p`, so
     the correction is a map `(a_p, n_p, z_p, coefficient) -> (da, dn, dz)`.
     Plan: sweep the spherical coefficient with the `Aberrated` model of
     pylorenzmie, measure the bias map of the network, fit a low-order
     correction (a table or a surface) on one set and judge it on another,
     and report the lock-on of the fit both with the plain model and with
     the model that matches the data, so that the cost of the mismatch is
     separate from the quality of the start.  Alternative if the map is not
     smooth: give the coefficient to the network as an input, like `scale`.
     The coefficient comes from a fit of a reference sphere.
     (iii) What remains for the stress sets (item 1, (e)) is what scaling
     does not touch: the level and the statistics of the noise, errors of
     the normalization (a background that does not divide out to 1, loss of
     contrast from partial coherence or pixel integration), 8-bit
     quantization and saturation, illumination gradients, and the
     aberrations that the correction does not remove.  Randomize these in
     training (domain randomization) and measure the loss of lock-on against
     them.
4. Choose from the results and build the 3.0 pipeline behind the same
   interface.  The Localizer moves to Ultralytics only if it survives step
   3C.  Implement pylorenzmie's `BaseEstimator`
   (`estimate(hologram) -> pandas.Series`) in the CATCH estimator so that
   pylorenzmie never imports CATCH.
5. Release.  Version 3.0.0 with a changelog that lists what changed in the
   models and how to select the 2.x models; update the documentation and the
   README; retrain the shipped weights (they are not compatible with the
   new networks).

