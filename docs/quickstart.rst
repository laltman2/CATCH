Quickstart
==========

Installation
------------
CATCH requires Python 3.11 or later and PyTorch, and is installed from a
checkout:

.. code-block:: sh

   git clone https://github.com/laltman2/CATCH.git
   pip install ./CATCH

The fits (``analyze(fit=True)``) need pylorenzmie:

.. code-block:: sh

   pip install "./CATCH[fit]"

CATCH runs on a CUDA GPU, on Apple's GPU, or on the CPU (much slower), and
chooses the fastest by itself.

The whole pipeline
------------------
.. code-block:: python

   import cv2
   from CATCH import CATCHapp

   image = cv2.imread('docs/images/test_image_large.png',
                      cv2.IMREAD_GRAYSCALE)
   catch = CATCHapp()
   results = catch(image)          # a list of images works too

``results`` is a :class:`pandas.DataFrame` with one row per particle:

==============================  ============================================
column                          meaning
==============================  ============================================
``label``, ``confidence``       ``'hologram'``, and the score of the detector
``x_p``, ``y_p``                center of the hologram in the image, pixels
``bbox``                        ``((x_0, y_0), w, h)``: a square of twice the
                                radius of the 10th bright ring of the
                                estimate
``edge``                        ``True`` if the box crosses an image edge
``z_p``, ``a_p``, ``n_p``       the best start: axial position [pixels],
                                radius [um] and refractive index
``framenum``                    index of the image in the list analyzed
``misfit``                      robust misfit of the best start to the
                                radial profile
``starts``                      all the starts, ``(z_p, a_p, n_p)``, best
                                first
==============================  ============================================

The best start is the most plausible estimate, but for large spheres far
from the focal plane several sizes and indexes make nearly the same
hologram, and the best by the profile is often not the right one.  A fit
from every start decides:

.. code-block:: python

   catch = CATCHapp(nprocs=6)              # fits in 6 processes
   results = catch(image, fit=True)

The processes of the fits start afresh, so a script that uses more than one
must run under ``if __name__ == '__main__':``.

adds the reported fit (``x_fit``, ``y_fit``, ``z_fit``, ``a_fit``,
``n_fit``, ``redchi``), a ``flag``, and how well the data constrain it:
``nsolutions``, the number of distinct solutions (by radius) within 0.5% of
the reduced chi-squared of the reported one, the spread of their ``a_p``,
``n_p`` and ``z_p`` (``a_spread``, ``n_spread``, ``z_spread``), and the best
other solution (``alt_a``, ``alt_n``, ``alt_redchi``).  See
:func:`CATCH.lib.selection.select` for the rule.  The ambiguity is a lower
bound: it counts only the solutions that the starts found.

The crop and the pixel budget
-----------------------------
The box, and the crop that each fit uses, reach the 10th bright ring of the
particle's own hologram (``rings=10``; the radius follows from its height).
Fewer rings make a fit less sensitive to background errors, aberrations and
the fringes of neighbors, which grow with the distance from the particle.
Each fit uses a random subset of 4,000 pixels of its crop (``npixels``); more
pixels fit better and take longer:

.. code-block:: python

   catch = CATCHapp(rings=10, npixels=8000)

On 717 simulated particles with neighbors and edges, fits from five starts
locked on for 77.4% with 10 rings, against 69.7% with 20, and 80.6% with
10 rings and 8,000 pixels.  In simulation the gain from fewer rings is the
density of the pixels (20 rings with twice the pixels do as well); on real
frames of 1 um polystyrene and 3 um silica the 10-ring fits had fewer
ambiguous solutions and agreed better with the streaming analysis, also at
the same pixel density.

Your instrument
---------------
Give the microscope, and the outputs are in its units:

.. code-block:: python

   catch = CATCHapp(instrument=dict(wavelength=0.447,     # um, in vacuum
                                    magnification=0.120,  # um per pixel
                                    n_m=1.34))            # the medium

The shipped networks serve an instrument whose pixel is 1 to 2.5 times the
reference pixel (0.048 um at 0.447 um in water) measured in wavelengths of
the medium, ``CATCHapp.scale``; a frame outside that range is resized for
the detector.  No training is needed for such an instrument (see
:doc:`training` for the others).

A prior for the sample
----------------------
If the material is known, a range of the index makes the choice among
near-degenerate solutions:

.. code-block:: python

   results = catch(image, fit=True, nrange=(1.55, 1.70))  # polystyrene

The Estimator then adds starts inside the range, and the reported fit is
the best inside it (``flag`` is ``'outside prior'`` if no fit is).  Without
a prior, the standard range of the index (up to 1.8 relative to water) is
preferred unless a fit of higher index is clearly better.

One stage at a time
-------------------
.. code-block:: python

   from CATCH import Localizer, Estimator

   features = Localizer()(image)                # label, confidence, x_p, y_p
   estimates = Estimator()(image, features.x_p, features.y_p)

Models are chosen by name (a directory in ``models/localizer/`` or
``models/estimator/``), or by the path of a directory such as the output of a
training run: ``CATCHapp(lmodel='runs/localizer', emodel='runs/estimator')``.
