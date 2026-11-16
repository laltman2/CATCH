Quickstart
==========

Installation
------------
CATCH requires Python 3.11 or later and is installed from a checkout:

.. code-block:: sh

   git clone https://github.com/laltman2/CATCH.git
   pip install ./CATCH

The Localizer needs YOLOv5, an optional extra because this version of CATCH
requires ``yolov5 <= 6.0.7``:

.. code-block:: sh

   pip install "./CATCH[yolo]"

``import CATCH`` does not import YOLOv5.  :class:`~CATCH.Estimator.Estimator`
works without it, and ``CATCH.Localizer`` and ``CATCH.CATCHapp`` load it on
first use.

The whole pipeline
------------------
.. code-block:: python

   import cv2
   from CATCH import CATCHapp

   image = cv2.imread('docs/images/test_image_large.png',
                      cv2.IMREAD_GRAYSCALE)
   results = CATCHapp()(image)     # a list of images works too

``results`` is a :class:`pandas.DataFrame` with one row per detected particle:

==============================  ============================================
column                          meaning
==============================  ============================================
``label``, ``confidence``       class name and confidence of the detection
``x_p``, ``y_p``                center of the feature in the image, pixels
``bbox``                        ``((x_0, y_0), w, h)``, in pixels
``edge``                        ``True`` if the box crosses an image edge
``z_p``, ``a_p``, ``n_p``       estimated axial position, radius and
                                refractive index
``framenum``                    index of the image in the list analyzed
==============================  ============================================

The center of a feature that crosses an edge can lie outside the image.

One stage at a time
-------------------
.. code-block:: python

   from CATCH import Localizer, Estimator, CATCHapp

   features = Localizer()(image)               # DataFrame of detections
   crops = CATCHapp.crop(image, features)      # one square crop per feature
   estimates = Estimator()(crops)              # DataFrame of z_p, a_p, n_p

The Estimator takes square crops of any size.  It resizes them to the size
of the network input and gives the network the crop size as a second input.
Another model is chosen by name, as in ``Estimator(model='boltzmann')``.
Models are directories in ``cfg_estimator/`` and ``cfg_yolov5/`` that hold
``weights.pt`` and, for the Estimator, ``configuration.json``.
