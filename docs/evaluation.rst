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
