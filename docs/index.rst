CATCH
=====

**CATCH** (Characterizing And Tracking Colloids Holographically) finds
colloidal particles in in-line holographic microscopy images and estimates
each particle's axial position, radius and refractive index, well enough
for a fit of the Lorenz-Mie theory to lock on.

A :class:`~CATCH.Localizer.Localizer` (a heatmap detector, after CenterNet)
finds the center of each hologram.  An :class:`~CATCH.Estimator.Estimator`
measures the radial profile of each hologram, lets a network propose 64
candidates, checks every candidate against the profile with the Lorenz-Mie
theory, and returns the best as the starts of a fit.
:class:`~CATCH.CATCHapp.CATCHapp` runs both and, if asked, the fits.  The
networks are trained on synthetic holograms and serve any instrument whose
pixel, in wavelengths of the medium, is 1 to 2.5 times that of the
reference instrument.

.. code-block:: python

   import cv2
   from CATCH import CATCHapp

   image = cv2.imread('docs/images/test_image_large.png',
                      cv2.IMREAD_GRAYSCALE)
   results = CATCHapp()(image)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   conventions
   training
   evaluation
   api/catch
   api/models
