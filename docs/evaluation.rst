Evaluation
==========

A network is judged by what it is for: whether a fit of the Lorenz-Mie model
that starts from its estimates finds the particle.  A fit *locks on* if the
fitted radius is within 5 nm of the true radius.  The error of the estimates
is reported too, but it is not the measure of success: a network that is
within a few percent can still start most fits outside the valley of a large
sphere.

The harness needs pylorenzmie (``pip install "./CATCH[train]"``).

.. code-block:: sh

   python -m CATCH.evaluation.run --ncases 180 --nprocs 4 \
       --model default --output results.csv

or, to evaluate any estimator:

.. code-block:: python

   from CATCH.evaluation.cases import make_cases
   from CATCH.evaluation.run import evaluate, summarize, by_radius

   def estimate(crops):            # list of 2-D arrays -> DataFrame
       ...                         # with columns z_p, a_p and n_p

   results = evaluate(estimate, make_cases(180), 'training', nprocs=4)
   print(summarize(results))
   print(by_radius(results))

The cases are the same number of simulated polystyrene and silica spheres in
water (radius 0.5 to 5 um, 150 to 600 pixels from the focus, centered or 3
pixels off) for each radius.  There are two conditions: ``'fixed'`` crops of
201 x 201 pixels with 2% noise, and ``'training'`` crops sized as CATCH
trains on them (twice the extent of the feature, with a 5% error, and 5%
noise).  The default network gives, with 180 cases:

=========================  =======  ========
                           fixed    training
=========================  =======  ========
median error of ``z_p``    60%      3.7%
median error of ``a_p``    51%      3.4%
median error of ``n_p``    50%      1.1%
within 5% in all three     0%       41%
fits that lock on          4%       58%
=========================  =======  ========

and, for the ``'training'`` crops, the fraction of the fits that lock on for
radii of 0.5, 1, 2, 3, 4 and 5 um is 77, 97, 57, 53, 40 and 27%.  The
``'training'`` crops use the true extent of the feature, which a Localizer
does not know.

Scenes with several particles
-----------------------------

A hologram has fringes far from the particle that make it, so in a crowded
scene the fringes of neighbors overlap, and a feature near the edge of the
image is cut off.  ``CATCH.evaluation.scene_run`` evaluates a *pipeline*
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
``a_p`` and ``n_p``.  ``catch_pipeline`` makes one of ``CATCHapp`` (which needs
yolov5), and two pipelines cheat to isolate a part of CATCH: ``truth_pipeline``
returns the true particles, which is the best that any pipeline can do, and
``estimator_on_truth`` takes the true centers, with crops of twice the true
extent, and asks an Estimator for the rest.

.. code-block:: sh

   python -m CATCH.evaluation.scene_run --pipeline oracle --nscenes 40 \
       --nparticles 1 6 --nprocs 4 --output scenes

With the default Estimator (40 scenes, 146 particles; 60 and 80 scenes of a
single particle), the fraction of the particles for which the fit locks on is:

.. list-table::
   :header-rows: 1

   * -
     - true particles (the best possible)
     - default Estimator on the true centers
   * - one particle in the scene
     - 97%
     - 30%
   * - 1 to 6 particles
     - 73%
     - 23%

By the distance from the center to the nearest edge:

==================  ================  =======================
distance            true particles    default Estimator
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
its neighbors even from the true values, and the default Estimator cannot
handle a feature that is cut off by the edge even when it is given the exact
center and crop size.  These are the effects that a better network, and a
fit of the neighbors together, must overcome.

Development and test sets
-------------------------

Two sets of scenes are defined in ``CATCH.evaluation.sets``, each made from a
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

   python -m CATCH.evaluation.scene_run --set dev --pipeline oracle
   python -m CATCH.evaluation.scene_run --set test --nscenes 40 ...

``--nscenes`` asks for the first scenes of a set, which are the same however
many are asked for.  The truth about every particle is stored in
``evaluation/sets/``, and ``sets.verify`` makes the first scenes again and
compares: a change to the code that makes the scenes would invalidate every
comparison, and a test checks for it.

Training cost
-------------

A network that is more accurate but as slow to train is not an improvement in
what this project is after.  The cost is the **time and the number of
examples that a training run needs to reach a target**, most usefully a
rate of locking on (``cost.validate``), and also a loss.  The examples are the
number of images that the network was trained on, which does not depend on the
machine; the time is in seconds on the machine that
``cost.describe_hardware`` records, and leaves out the time spent on
validation.  The cost of making the data is counted separately
(``data_seconds``).

``EstimatorTraining`` writes the record of a run: ``<savefile>_cost.csv``
(the seconds, epoch and examples at the end of each epoch, with the losses) and
``<savefile>_cost.json`` (the configuration, the time to make the data, and
the machine).  ``cost.cost_to_reach`` reads the first time that a run got to
a target, and ``cost.evaluate_checkpoints`` measures the rate of locking on of
the checkpoints that a run saved (``Estimator(weights=..., config=...)``), so
that a learning curve of lock-on against examples can be made for a run that
did not validate as it went.

.. code-block:: python

   from CATCH.evaluation import cost
   record, info = cost.TrainingLog.load('model_cost.csv')
   cost.cost_to_reach(record, 'test_loss', 0.01, lower_is_better=True)
   # {'seconds': 6412.0, 'epoch': 240, 'examples': 2400000}

**The old recipe.**  The repository holds the loss histories of two runs on
the same data (``overlaps_fullrange``, 10,000 images in an epoch), which trained
for 10,000 and 9,428 epochs: about 10\ :sup:`8` examples, in a mode that stops
only when all three outputs have stopped improving.  The shipped ``default``
model has the configuration of the same recipe (``autotrain_300p``), whose own
history is not in the repository.  The time was not recorded.  On a CPU of an
Apple laptop (the trainer has no GPU path for this machine) the current code
makes data at 29 images per second and trains at 74 images per second, so that
10\ :sup:`8` examples would take about 16 days there; the original runs used
GPUs, which are faster, so this is a scale and not a figure for them.

Comparing versions of CATCH
---------------------------

The checkouts of different versions are all called ``CATCH``, so they cannot
be imported together.  ``scene_run --write-images`` writes the images of the
scenes, ``evaluation/detect_images.py`` runs the ``CATCHapp`` of the checkout
on ``PYTHONPATH`` on them and records what it finds, and
``scene_run --pipeline recorded --detections`` scores the record against the
same truth.  The original CATCH (YOLOv5 and the default Estimator) on 200
scenes of 1 to 6 particles has a precision of 85% and a recall of 79%, a
median error of the center of 1.9 pixels, and a fit locks on for 15.6% of the
particles (19.7% of those that were found), with the worst results near an
edge and near a neighbor.  The files and the procedure are in
``evaluation/baselines/catch-2.0.0/``, with the same measurements on the
frozen test set (1381 particles): a precision of 85%, a recall of 79%, and a fit
locks on for 15.0% of the particles.  The current code gives the same
detections, because the models have not changed.

.. autofunction:: CATCH.evaluation.run.evaluate

.. autofunction:: CATCH.evaluation.run.summarize

.. autofunction:: CATCH.evaluation.run.by_radius

.. autofunction:: CATCH.evaluation.cases.make_cases

.. autofunction:: CATCH.evaluation.cases.simulate

.. autofunction:: CATCH.evaluation.fitting.fit

.. autofunction:: CATCH.evaluation.fitting.refine

.. autofunction:: CATCH.evaluation.fitting.locks_on

.. autoclass:: CATCH.evaluation.cases.Case

.. autoclass:: CATCH.evaluation.cases.Crop

.. autoclass:: CATCH.evaluation.fitting.Job

.. autofunction:: CATCH.evaluation.scenes.make_scenes

.. autoclass:: CATCH.evaluation.scenes.Scene

.. autofunction:: CATCH.evaluation.matching.match

.. autoclass:: CATCH.evaluation.matching.Match

.. autofunction:: CATCH.evaluation.pipelines.truth_pipeline

.. autofunction:: CATCH.evaluation.pipelines.estimator_on_truth

.. autofunction:: CATCH.evaluation.pipelines.catch_pipeline

.. autofunction:: CATCH.evaluation.scene_run.evaluate_scenes

.. autoclass:: CATCH.evaluation.scene_run.SceneResults

.. autofunction:: CATCH.evaluation.scene_run.summarize_scenes

.. autofunction:: CATCH.evaluation.scene_run.binned

.. autofunction:: CATCH.evaluation.fitting.fit_particle

.. autoclass:: CATCH.evaluation.sets.SceneSet

.. autofunction:: CATCH.evaluation.sets.verify

.. autoclass:: CATCH.evaluation.cost.TrainingLog

.. autofunction:: CATCH.evaluation.cost.cost_to_reach

.. autofunction:: CATCH.evaluation.cost.validate

.. autofunction:: CATCH.evaluation.cost.evaluate_checkpoints

.. autofunction:: CATCH.evaluation.cost.describe_hardware

Other instruments
-----------------

``python -m CATCH.evaluation.equivalence`` simulates the same particles as
the reference instrument and as another one sees them, runs the
``Estimator`` on both (``Estimator(instrument=...)``, see
``CATCH.instruments``), and prints the errors side by side.  A change of
wavelength, magnification or medium is a change of scale, so they should
agree; the shipped network's errors do (median error of the radius 5.4% on
each of three instruments).

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
(the default of the scene harness; the single-particle ``run.py`` and the
older baselines use least squares).  ``python -m CATCH.evaluation.centers`` measures how well a Localizer finds the
centers, by distance to the edge of the image, with the boxes that the border
cuts (``find_center`` is right for a box cut on one axis and wrong at a
corner).  ``python -m CATCH.evaluation.diagnose`` estimates the same particles in the
crop of the scene, in a crop padded with background so that the particle stays
at the center (``scene_run --pad``), alone in the scene, and alone at the
center, which tells the effect of neighbors from that of the shift at an edge
and from the network itself.  Two versions evaluated on
the same scenes are compared on the particles that differ:

.. code-block:: sh

   python -m CATCH.evaluation.paired A_particles.csv B_particles.csv
