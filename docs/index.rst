CATCH
=====

**CATCH** (Characterizing And Tracking Colloids Holographically) detects
colloidal particles in in-line holographic microscopy images and estimates
each particle's axial position, radius and refractive index.

A :class:`~CATCH.Localizer.Localizer` (a YOLOv5 detector) finds the hologram
of each particle, and an :class:`~CATCH.Estimator.Estimator` (a convolutional
neural network) analyzes a crop around each hologram.
:class:`~CATCH.CATCHapp.CATCHapp` runs both.  The models are trained on
synthetic holograms computed with the Lorenz-Mie theory of light scattering.

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
   api/catch
   api/models
