Conventions
===========

Images
------
The models expect a *normalized* hologram: the recorded image divided by
the background, scaled so that the mean is 100 in 8-bit units.
:class:`~CATCH.Localizer.Localizer` multiplies an image whose maximum is below
100 by 100, and :class:`~CATCH.Estimator.Estimator` rescales each crop to a
mean of 100.

Units
-----
Lengths are in micrometers, except coordinates in an image, which are in
pixels.  The default models were trained on spheres in water
(:math:`n_m = 1.34`) imaged at a wavelength of 0.447 µm and a magnification
of 0.048 µm/pixel, with radii of 0.2 to 5 µm and refractive indexes of 1.38
to 2.5.

Models
------
A model is a directory in ``cfg_estimator/`` (for the Estimator) or
``cfg_yolov5/`` (for the Localizer) that holds ``weights.pt`` and, for the
Estimator, ``configuration.json``, which records the input shape and the
ranges that scale the parameters (see
:class:`~CATCH.ParamScale.ParamScale`).
