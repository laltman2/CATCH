Pipeline
========

.. autoclass:: CATCH.CATCHapp.CATCHapp
   :members: analyze, scale, sides

.. autoclass:: CATCH.Localizer.Localizer
   :members: localize

.. autofunction:: CATCH.Localizer.symmetry_center

.. autofunction:: CATCH.Localizer.normalize

.. autoclass:: CATCH.Estimator.Estimator
   :members: estimate, misfits

.. autofunction:: CATCH.Estimator.ring_radius

Fits and the choice among them
------------------------------

.. automodule:: CATCH.selection
   :members: select, distinct, COLUMNS, NMAX, DELTA, PAD, TOL, SAME_A

.. autoclass:: CATCH.fitter.Fitter
   :members: __call__, close

.. autofunction:: CATCH.fitter.fit_particle

.. autofunction:: CATCH.fitter.fit

.. autodata:: CATCH.fitter.NPIXELS

.. autodata:: CATCH.fitter.LIMITS

Instruments
-----------

.. automodule:: CATCH.instruments
   :members: REFERENCE, pixel_ratio, from_reference, to_reference,
             wavelength_in_medium
