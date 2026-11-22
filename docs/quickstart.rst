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
``saturated``                   fraction of the box at the limit of the camera
``snr``                         signal-to-noise ratio of the hologram in the box
``quality``                     ``''``, or why not to trust the particle:
                                ``'low snr'``, ``'saturated'``
``z_p``, ``a_p``, ``n_p``       the best start: axial position [pixels],
                                radius [um] and refractive index
``framenum``                    index of the image in the list analyzed
``misfit``                      robust misfit of the best start to the
                                radial profile
``starts``                      all the starts, ``(z_p, a_p, n_p)``, best
                                first
==============================  ============================================

A camera records at most its limit (255 of a background of 100, for an
8-bit image), and the hologram of a sphere that focuses the light has spots
that are brighter: the pixels saturate.  CATCH leaves them out of the
profiles and of the fits, and ``saturated`` is the fraction of each box that
was lost.  Up to a few percent costs little; the more that is lost, the less
to trust the particle.  An image of ``uint8`` is taken to be from such a
camera: its pixels at 255 are the saturated ones.  If you divide the frames
by a background yourself, tell CATCH which pixels were saturated in the raw
frame with a boolean mask (``True`` for the pixels to leave out):

.. code-block:: python

   raw = ...                        # the frame as the camera recorded it
   image = raw / background         # normalized, median near 1
   results = catch(image, saturation=(raw >= 255))

That is exact, which a level cannot be: when the background is not uniform the
saturated pixels are at ``255 / background`` after the division, a level that
changes across the frame.  On simulated scenes with a background that varies
by 25% to either side of its mean, the fits lock on for 73% of the particles
with the mask, 65% with a single level of 2.55 and 55% without masking.
``saturation`` also takes that level, in units of the background (a number,
``catch(image, saturation=2.55)``), a single mask for all the images or one
for each (a sequence), and ``numpy.inf`` switches it off.

``snr`` is the energy of the hologram in the box (the deviation of the
pixels from the background, squared and summed, over the variance of the
noise, which is estimated from the image or given as ``noise``), square-rooted,
counting only the pixels that were measured, so that what is outside the
frame or saturated does not count.  A ratio of 100 is a hologram that can
hardly be analyzed; 400 to 800 is enough for the radius to be within 1% for
four particles in five, and only above about 1,000 is the radius within 5 nm
for most.  ``quality`` collects the reasons for doubt: ``'low snr'`` (below
100), ``'saturated'`` (more than 5% of the box at the limit) or both.  The
thresholds are :data:`CATCH.CATCHapp.LOW_SNR` and
:data:`CATCH.CATCHapp.MAX_SATURATED`, calibrated on simulated scenes
(:doc:`evaluation`).  A faint pattern that is not a particle can have a high
``snr``: ``confidence`` is what tells them apart.

The best start is the most plausible estimate, but for large spheres far
from the focal plane several sizes and indexes make nearly the same
hologram, and the best by the profile is often not the right one.  A fit
from every start decides:

.. code-block:: python

   catch = CATCHapp(nprocs=6)              # fits in 6 processes
   results = catch(image, fit=True)

The processes of the fits start afresh, so a script that uses more than one
must run under ``if __name__ == '__main__':``.  They are kept for the next
frames; use the app in a ``with`` statement, or call ``catch.close()``, to shut
them down when you are done:

.. code-block:: python

   with CATCHapp(nprocs=6) as catch:
       for frame in frames:
           results = catch(frame, fit=True)

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

The shipped networks are meant for an instrument whose pixel is 1 to 2.5 times
the reference pixel (0.048 um at 0.447 um in water) measured in wavelengths of
the medium, ``CATCHapp.scale``; a frame outside that range is resized for
the detector.  They have been tested on simulated scenes of a few instruments
and on real frames of one, so check the results for yours (``scene_run
--instrument``, :doc:`evaluation`); see :doc:`training` to train for an
instrument outside the range or that they do not serve well.

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
