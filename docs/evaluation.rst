Evaluation
==========

CATCH is judged by what it is for: whether a fit of the Lorenz-Mie model
that starts from its estimates finds the particle.  A fit *locks on* if the
fitted radius is within 5 nm of the true radius.  The error of the
estimates is reported too, but it is not the measure of success: an
estimate within a few percent can still start a fit outside the valley of
a large sphere.

The harness needs pylorenzmie (``pip install "./CATCH[train]"``).

Scenes with several particles
-----------------------------

A hologram has fringes far from the particle that make it, so in a crowded
scene the fringes of neighbors overlap, and a feature near the edge of the
image is cut off.  ``CATCH.customization.evaluation.scene_run`` evaluates a *pipeline*
(something that finds the particles in an image and estimates each) on
simulated scenes of 1024 x 1280 pixels with several particles:

1. the pipeline finds the particles; they are matched to the true particles
   (the closest pairs first, within 15 pixels, which is how far a fit can
   move the center), which gives the precision and recall of the detection
   and the error of the center;
2. a fit starts from each match, on a crop of the scene around the detection
   (twice the extent of the estimated feature).  The neighbors are in the
   crop, as extra fringes, and a feature at the edge is cut off, so both count
   against the fit;
3. the results are summarized overall and by the number of particles in the
   scene, the distance of the center from the nearest edge, and the distance
   to the nearest neighbor.

A pipeline is a function of a ``Scene`` that returns a DataFrame with a row
for each particle that it found and the columns ``x_p``, ``y_p``, ``z_p``,
``a_p`` and ``n_p``.  ``catch_pipeline`` makes one of ``CATCHapp``, and two
pipelines cheat to isolate a part of CATCH: ``truth_pipeline`` returns the
true particles, which is the best that any pipeline can do, and
``estimator_on_truth`` gives the true centers to an Estimator
(``crops_on_truth`` does the same for estimators of crops, such as the
references of pylorenzmie, with crops of twice the true extent).

.. code-block:: sh

   python -m CATCH.customization.evaluation.scene_run --pipeline oracle --nscenes 40 \
       --nparticles 1 6 --nprocs 4 --output scenes

With the Estimator of CATCH 2 (40 scenes, 146 particles; 60 and 80 scenes
of a single particle), the fraction of the particles for which the fit
locked on was:

.. list-table::
   :header-rows: 1

   * -
     - true particles (the best possible)
     - CATCH 2 Estimator on the true centers
   * - one particle in the scene
     - 97%
     - 30%
   * - 1 to 6 particles
     - 73%
     - 23%

By the distance from the center to the nearest edge:

==================  ================  =======================
distance            true particles    CATCH 2 Estimator
                    (1 particle)      (1 particle)
==================  ================  =======================
within 25 pixels    89%               0%
25 to 100 pixels    100%              6%
100 to 250 pixels   94%               23%
250 pixels or more  100%              63%
==================  ================  =======================

and, with 1 to 6 particles, for the best possible pipeline, by the distance to
the nearest neighbor: 25% (within 50 pixels), 28% (50 to 150), 71% (150 to
400) and 92% (400 or more).  The fit of one particle on a crop is spoiled by
its neighbors even from the true values, and the Estimator of CATCH 2
could not handle a feature cut off by the edge even when it was given the
exact center and crop size.

Development and test sets
-------------------------

Two sets of scenes are defined in ``CATCH.customization.evaluation.sets``, each made from a
seed, so that anyone can make the same scenes:

* ``dev`` (200 scenes, 717 particles) is for development: try ideas on it and
  look at the failures.  These are the scenes of the baseline of the original
  CATCH.
* ``test`` (400 scenes, 1381 particles) is **frozen**.  Compare finished
  versions on it, and do not use it to choose among ideas, or the numbers stop
  being an honest measure of how a version does on scenes that it has not
  been tuned to.  It is twice the size of ``dev`` so that it can show small
  differences (the rate of locking on has a standard error of about 1
  point).

.. code-block:: sh

   python -m CATCH.customization.evaluation.scene_run --set dev --pipeline oracle
   python -m CATCH.customization.evaluation.scene_run --set test --nscenes 40 ...

``--nscenes`` asks for the first scenes of a set, which are the same however
many are asked for.  The truth about every particle is stored in
``customization/evaluation/sets/``, and ``sets.verify`` makes the first scenes again and
compares: a change to the code that makes the scenes would invalidate every
comparison, and a test checks for it.

Another instrument
------------------

``--instrument WAVELENGTH MAGNIFICATION N_M`` (in ``scene_run``, ``centers``
and ``detect_images``) simulates another microscope: the wavelength and the
size of a pixel in micrometers, and the index of the medium.  The scenes
of a set are the same particles (positions, radii, indexes) as that
instrument sees them, with the heights converted so that they are the same in
wavelengths of the medium; the stored truth and ``sets.verify`` are those of
the reference instrument.  The fits, the crops and the scan use the
instrument too, and ``--lmodel`` and ``--model`` take the models of a
network that was trained for it.  The estimators of ``--estimator
analytic|de`` are for the reference instrument only.

.. code-block:: sh

   python -m CATCH.customization.evaluation.scene_run --set dev \
       --pipeline oracle --instrument 0.532 0.1 1.335

A word of caution about the heights: the sets draw them at random, and a
sphere of a few micrometers focuses the light, so that some scenes have a
spot that is tens or hundreds of times brighter than the background, far
beyond what a camera records.  Such a scene is as hard for the neighbors of
the particle as it is for the particle, and a few of them can move the rate
of locking on by tens of points when the instrument changes slightly.
Compare instruments on enough scenes, and look at the scenes that fail.

Saturation
----------

A camera records at most 2.55 times the background (255 of an 8-bit image
with a background of 100), but the hologram of a sphere that focuses the
light has spots that are many times brighter.  ``--camera`` records the
scenes the way such a camera does, ``saturated`` in the table of particles
is the fraction of the crop of a particle that is at the limit (also for
scenes that were not recorded: what would be lost), and the report has a
table by that fraction.  ``--mask-saturated`` leaves the saturated pixels
out of the fits and, with ``--pipeline oracle``, out of the profiles of the
Estimator.

On the 717 particles of the ``dev`` set, 215 have no saturated pixel, 402
have less than 1% of their crop, 99 have 1 to 5% and 1 has more.  The fits
that start from the truth lock on for 76% of them when the scenes are not
clipped and for 55% when they are, and for 78% when the saturated pixels
are left out of the fit: the fit can ignore what the camera did not record.
An Estimator that was not trained for it cannot (the shipped model before
version 3.0 was finished): leaving the saturated pixels out of its profiles
did not help, the oracle pipeline locking on for 69% of the particles of
unclipped scenes, 50% of clipped ones, and 50% with the pixels left out,
because it had not seen profiles whose inner rings are missing.  The shipped
model was fine-tuned on examples of which half (``--saturated``) are made as
a camera that saturates records them, and then the oracle pipeline locks on
for 69% of the particles of the clipped scenes with the saturated pixels left
out of the profiles and the fits, and for 68% of those of the unclipped
scenes: nothing is lost where the camera does not saturate.

Signal to noise
---------------

``snr`` of a particle (``Estimator.signal_to_noise``) is the energy of its
hologram in the box over the variance of the noise, square-rooted, from the
radial profile, counting the pixels that were measured.  It is calibrated by
scenes of five noise levels (0.05 to 0.8 of the background; 150 scenes of 1 to
5 particles each, recorded by an 8-bit camera, saturated pixels left out, the
Estimator at the true centers, a fit from its best start):

.. code-block:: sh

   python -m CATCH.customization.evaluation.scene_run --nscenes 150 \
       --nparticles 1 5 --noise 0.2 --camera --mask-saturated \
       --pipeline oracle

The median ratio falls as the noise grows (2,000, 1,000, 500, 250 and 120), and
the lock-on depends on the ratio, not on the noise that made it: at 400 to 800
it is 45, 56, 55 and 48% for the noise levels that reach it.  Pooled:

==========  =====================  ==================
``snr``     radius within 1%       radius within 5 nm
==========  =====================  ==================
below 100   9-16%                  8-9%
100-200     45%                    19%
200-400     70%                    34%
400-800     82%                    54%
800-1600    88%                    68%
1600-3200   94%                    82%
==========  =====================  ==================

There is no sharp edge, only a region where the data cannot decide.  Below
100 the particle cannot be analyzed, which is what ``quality`` calls
``'low snr'``.  A faint pattern that is not a particle, such as the artifact of
the tutorial, has a ratio of hundreds: it is signal, and ``confidence`` is
what tells it from a particle.

**Saturation** has no sharp edge either, and it is different for the Estimator
and for the fits.  Scenes clipped at 2.55, 1.6, 1.3 and 1.15 times the
background (``--camera LEVEL``), with the saturated pixels left out, have a
median of 0.3, 1.9, 7 and 18% of the box saturated.  A fit that starts from the
truth still locks on for 72 to 78% of the particles up to 40% saturated, because
the outer rings carry the information.  The starts of the Estimator do not
survive: with the Estimator at the true centers the lock-on is 78% below 2% of
the box saturated, 65% from 2 to 5%, 41% from 5 to 10%, 26% from 10 to 20%
and 6% above 20%.  Part of that is the training, which used the limit of an
8-bit camera (2.55) only, so it is worse for a camera whose background is
brighter, where the limit is lower.  ``quality`` calls ``'saturated'`` more than
5% of the box; a model trained for the limit of its camera should do better.

**The limit of the camera.**  The shipped model was trained for the limit of
an 8-bit camera with a background of 100 (2.55 times the background).  A model
trained with ``--levels 1.33 4`` draws the limit for each example (a
background of 25% to 75% of the full scale).  The Estimator at the true
centers, on 150 scenes clipped at each limit, with the saturated pixels left
out (lock-on within 5 nm):

============== ========= ==== ==== ==== ==== ====
limit          unclipped 4.0  2.55 1.6  1.3  1.15
============== ========= ==== ==== ==== ==== ====
shipped        0.73      0.72 0.75 0.65 0.43 0.25
``--levels``   0.72      0.70 0.69 0.68 0.55 0.34
============== ========= ==== ==== ==== ==== ====

It gains at the limits that a camera would hardly be set to (a background
above 150 of 255) and loses 6 points at the limit that it is, so the shipped
model is the one for the usual exposure.  The second model also does better
with many saturated pixels: with 5 to 10% of the box saturated 56% lock on
against 41%, and with 10 to 20% 41% against 26%, so it would allow a
threshold of about 10% for ``'saturated'`` where the shipped model has 5%.

The Localizer is not affected: on the 200 scenes of the ``dev`` set it finds
99.2% of the particles unclipped, 99.0% at the limit 2.55 and 98.9% at 1.3,
with a median error of the center of 0.08 to 0.09 pixels at every limit.  Only
the precision falls with heavy clipping (97.9% at 1.6, 95.8% at 1.3) from
false detections on the clipped blobs.  Even where 10 to 20% of the box is
saturated, 98% are found.

On real frames of the lens-test videos (0.120 um/pixel, none saturated), the
weights before the saturation fine-tuning, the shipped ones and the ones of
``--levels`` agree with each other and with the streaming analysis to within
the scatter: for 3 um silica, a median radius of 1.52, 1.53 and 1.53 um and
an interquartile range of 0.07, 0.06 and 0.05 um (the nominal radius is
1.5 um).

Comparing versions of CATCH
---------------------------

Two versions of CATCH cannot be imported together (both are the package
``CATCH``; a 3.x checkout has it in ``src``, a 2.x checkout is the package
itself, so its parent is the directory to put on the path).  ``scene_run --write-images`` writes the images of the
scenes, ``customization/evaluation/detect_images.py`` runs the ``CATCHapp`` found
on ``PYTHONPATH`` on them and records what it finds, and
``scene_run --pipeline recorded --detections`` scores the record against the
same truth.  The original CATCH (YOLOv5 and the default Estimator) on 200
scenes of 1 to 6 particles has a precision of 85% and a recall of 79%, a
median error of the center of 1.9 pixels, and a fit locks on for 15.6% of the
particles (19.7% of those that were found), with the worst results near an
edge and near a neighbor.  The files and the procedure are in
``benchmarks/catch-2.0.0/``, with the same measurements on the
frozen test set (1381 particles): a precision of 85%, a recall of 79%, and a fit
locks on for 15.0% of the particles.

CATCH 3.0.0, by the same procedure on the frozen test set
(``benchmarks/catch-3.0.0``): a precision of 98.9%, a recall of
98.8%, a median error of the center of 0.08 pixels, and a fit from the best
start locks on for 67.8% of the particles (76.4% of the standard ones),
against 76.8% for a fit from the true particles.  Up to a radius of 2 um it
is at that limit; for 3 to 5 um it reaches 64, 77 and 78% against 88-90%.
The first defaults of the 3.0 work (crops of 20 rings, an Estimator that was
not yet trained for saturation) gave 52.9%, and 56, 46 and 29%.

.. autofunction:: CATCH.customization.evaluation.cases.make_cases

.. autofunction:: CATCH.customization.evaluation.cases.simulate

.. autofunction:: CATCH.customization.evaluation.fitting.refine

.. autofunction:: CATCH.customization.evaluation.fitting.locks_on

.. autoclass:: CATCH.customization.evaluation.cases.Case

.. autoclass:: CATCH.customization.evaluation.cases.Crop

.. autoclass:: CATCH.customization.evaluation.fitting.Job

.. autofunction:: CATCH.customization.evaluation.scenes.make_scenes

.. autoclass:: CATCH.customization.evaluation.scenes.Scene

.. autofunction:: CATCH.customization.evaluation.matching.match

.. autoclass:: CATCH.customization.evaluation.matching.Match

.. autofunction:: CATCH.customization.evaluation.pipelines.truth_pipeline

.. autofunction:: CATCH.customization.evaluation.pipelines.estimator_on_truth

.. autofunction:: CATCH.customization.evaluation.pipelines.crops_on_truth

.. autofunction:: CATCH.customization.evaluation.pipelines.catch_pipeline

.. autofunction:: CATCH.customization.evaluation.scene_run.evaluate_scenes

.. autoclass:: CATCH.customization.evaluation.scene_run.SceneResults

.. autofunction:: CATCH.customization.evaluation.scene_run.summarize_scenes

.. autofunction:: CATCH.customization.evaluation.scene_run.binned

.. autoclass:: CATCH.customization.evaluation.sets.SceneSet

.. autofunction:: CATCH.customization.evaluation.sets.verify

Fits from several starts, and comparing versions
------------------------------------------------

Besides ``ok`` (the radius within 5 nm) the scene tables have ``close`` (within
1% of the radius, which takes the size of the particle into account) and
``redchi0``, the reduced chi-squared of the start before the fit.
``scene_run --starts 9`` fits each particle from the estimate and from other
radii along the degeneracy of the radius and the index, and keeps the fit with
the least chi-squared.  ``--estimator analytic`` starts from the analytic
estimator of pylorenzmie instead of the network.  ``--loss cauchy`` fits
with the robust Cauchy loss, which down-weights the fringes of neighbors, and
``--loss both`` fits least squares first and then refines with the Cauchy loss
(the default of the scene harness; the older baselines use least
squares).  ``python -m CATCH.customization.evaluation.centers`` measures how well CATCH
finds the centers, by distance to the edge of the image.  Two versions
evaluated on the same scenes are compared on the particles that differ:

.. code-block:: sh

   python -m CATCH.customization.evaluation.paired A_particles.csv B_particles.csv
