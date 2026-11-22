Pipeline
========

.. autoclass:: CATCH.CATCHapp.CATCHapp
   :members: analyze, scale, sides

.. autofunction:: CATCH.CATCHapp.quality_of

.. autodata:: CATCH.CATCHapp.LOW_SNR

.. autodata:: CATCH.CATCHapp.MAX_SATURATED

.. autoclass:: CATCH.Localizer.Localizer
   :members: localize

.. autofunction:: CATCH.Localizer.symmetry_center

.. autofunction:: CATCH.Localizer.normalize

.. autoclass:: CATCH.Estimator.Estimator
   :members: estimate, misfits, signal_to_noise

.. autofunction:: CATCH.Estimator.ring_radius

Fits and the choice among them
------------------------------

.. automodule:: CATCH.lib.selection
   :members: select, distinct, COLUMNS, NMAX, DELTA, PAD, TOL, SAME_A

.. autoclass:: CATCH.lib.fitter.Fitter
   :members: __call__, close

.. autofunction:: CATCH.lib.fitter.fit_particle

.. autofunction:: CATCH.lib.fitter.fit

.. autodata:: CATCH.lib.fitter.NPIXELS

.. autodata:: CATCH.lib.fitter.FIT_FAILURES

.. autodata:: CATCH.lib.fitter.LIMITS

Instruments
-----------

.. automodule:: CATCH.lib.instruments
   :members: REFERENCE, pixel_ratio, from_reference, to_reference,
             wavelength_in_medium
