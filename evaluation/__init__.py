'''Evaluation of CATCH by what matters: does a full fit lock on?

A network is judged by whether a fit of the Lorenz-Mie model that starts from
its estimates finds the particle, not by the error of the estimates.  A fit
*locks on* if the fitted radius is within 5 nm of the true radius.

The modules need pylorenzmie (the ``train`` extra):

``cases``
    Simulated particles and their holograms.
``fitting``
    The fit and the criterion for locking on.
``run``
    Runs an estimator on the cases, fits, and summarizes
    (``python -m CATCH.evaluation.run --help``).
'''
