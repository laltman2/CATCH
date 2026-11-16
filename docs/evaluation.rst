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
