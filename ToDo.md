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
   the fits lock on); every new version is to be compared with it; (b) the starting points that need no
   network (the analytic estimator and the differential-evolution estimator
   of pylorenzmie, which locked on 35%) as references; (c) scenes with
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
   a stated machine).  Still missing: the radius-aware lock-on in the scene
   tables; paired comparisons of two versions on the same particles; a measure
   of the quality of the start that does not depend on the fit; the stress
   sets of the instrument (item 1, (e)).  The cost of the old recipe is about 10^8 examples (two runs on
   the same data trained for 10,000 and 9,428 epochs of 10,000 images; the
   history of the shipped model is not in the repository) and 74 examples/s on
   a laptop CPU (16 days), which every new recipe is to be compared with, at
   equal lock-on on the `dev` set.
2. Training infrastructure, shared by everything below: generate data on the
   fly on the GPU (pylorenzmie's torch calculators) instead of reading PNGs;
   float inputs without clipping, with realistic noise and aberrations;
   sample `a_p` and `z_p` logarithmically and `n_p - n_m` in contrast;
   weight the sampling toward large, high-index particles; keep a fixed
   validation set; normalization layers, AdamW with a one-cycle schedule,
   mixed precision.  Target: tens of epochs, not thousands (a guess to
   check).
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
     are synthetic, a model can be trained across instruments: randomize the
     wavelength, the magnification, the medium, the noise and the aberrations
     in training, and either give the instrument to the network as an input or
     work in dimensionless units so that one model serves many.  Measure it
     on the stress sets of step 1, (e): the loss of lock-on away from the
     instrument that the model was trained for, against what the training for
     many instruments costs.
4. Choose from the results and build the 3.0 pipeline behind the same
   interface.  The Localizer moves to Ultralytics only if it survives step
   3C.  Implement pylorenzmie's `BaseEstimator`
   (`estimate(hologram) -> pandas.Series`) in the CATCH estimator so that
   pylorenzmie never imports CATCH.
5. Release.  Version 3.0.0 with a changelog that lists what changed in the
   models and how to select the 2.x models; update the documentation and the
   README; retrain the shipped weights (they are not compatible with the
   new networks).

