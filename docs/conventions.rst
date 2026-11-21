Conventions
===========

Images
------
The models expect a *normalized* hologram: the recorded image divided by
the background, with a background of 1.  An image on the scale of CATCH 2
(a background of 100 in 8-bit units, as a normalized hologram is often
stored) is divided by 100.  If the camera has a black level, subtract it
from the image and from the background before dividing.

Units
-----
Lengths are in micrometers, except coordinates in an image and the axial
position ``z_p``, which are in pixels of the camera.  ``z_p`` is positive:
the particles are above the focal plane.

The networks work in pixels of a *reference instrument* (a wavelength of
0.447 um in water, n_m = 1.34, and 0.048 um per pixel;
:data:`CATCH.instruments.REFERENCE`).  A hologram depends on the instrument
only through the size of a pixel in wavelengths of the medium, so CATCH
converts between the two (:mod:`CATCH.instruments`): the index relative to
the medium is what is kept, and the standard range of the index (up to 1.8
in water) is a range of the relative index.

Ranges
------
The shipped networks were trained on spheres of radius 0.2 to 5.5 um, index
1.38 to 2.5 and height 5 to 320 um (at the reference wavelength), on frames
with pixels of 1 to 2.5 reference pixels, noise of 1.5 to 7%, overlapping
holograms, particles at the edges, and imperfect normalization.  They are
most reliable for *standard* particles: heights of 10 um or more, index up
to 1.8, at least 100 pixels from an edge and 150 pixels from a neighbor.

Models
------
A model is a directory that holds ``weights.pt`` and
``configuration.json``: ``models/localizer/<name>`` for the Localizer
(the threshold of the detections, the guard of the centers, the range of
scales) and ``models/estimator/<name>`` for the Estimator (the reference
instrument, the number of starts, the standard range of the index, the
range of scales).  The training commands write models in this form.
