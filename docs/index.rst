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
networks are trained on synthetic holograms and are meant for instruments whose
pixel, in wavelengths of the medium, is 1 to 2.5 times that of the
reference instrument.  They have been tested on simulated scenes of the
reference instrument and of a few others, and on real frames of one
microscope (0.120 um per pixel); for another instrument, check the results
before relying on them (:doc:`evaluation`).

.. code-block:: python

   import numpy as np
   from PIL import Image
   from CATCH import CATCHapp

   image = np.asarray(
       Image.open('docs/images/test_image_large.png').convert('L'))
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
