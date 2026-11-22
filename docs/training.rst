Training
========

**An instrument may need no training.**  The shipped networks work in
pixels of a reference instrument and are meant for any microscope whose pixel,
in wavelengths of the medium, is 1 to 2.5 times the reference pixel
(``CATCHapp(instrument=...).scale``), but they have been tested only on
simulated scenes of a few instruments and on real frames of one (0.120 um per
pixel): measure how well they do for yours with the evaluation harness
(``scene_run --instrument``, :doc:`evaluation`).  Train when an instrument is
outside that range, when they do not do well enough, or to fine-tune for the
noise and backgrounds of a particular setup.

Training needs pylorenzmie (``pip install "./CATCH[train]"``) and, for
reasonable times, a GPU; set ``PYLORENZMIE_BACKEND=torch`` to render the
frames on it.  Both commands make their data on the fly from a seed (the
seeds of the evaluation sets are refused), write a model directory that
``CATCHapp`` loads, and resume an interrupted run when they are run again.

The Localizer
-------------
.. code-block:: sh

   python -m CATCH.customization.training.localizer runs/localizer --steps 20000
   # or fine-tune the shipped weights for another range of scales
   python -m CATCH.customization.training.localizer runs/localizer --steps 8000 \
       --init default --scales 2.5 5

The detector learns a heatmap of the centers and the position of each
center in its cell of 4 pixels, on frames of 0 to 10 particles with
heights of 5 to 330 um, faint particles, overlapping holograms, particles
at the edges and imperfect backgrounds (:mod:`CATCH.customization.training.frames`).
``eval.jsonl`` records, at both ends of the range of scales, the recall,
the false detections and the error of the centers.  The shipped model took
20,000 steps from scratch and 8,000 to fine-tune the offsets, about four
hours on an Apple M-series GPU.

The Estimator
-------------
.. code-block:: sh

   python -m CATCH.customization.training.estimator runs/estimator --steps 40000
   python -m CATCH.customization.training.estimator runs/estimator --steps 10000 \
       --init default --scales 2.5 5

First a library of what measurement adds to a radial profile is made from
synthetic frames (noise, neighbors, edges, the sampling of the camera);
then every batch is made of clean profiles of random particles, computed by
the batched Lorenz-Mie model, plus residuals from the library.  The network
proposes 64 candidates and only the best one learns from each example, so
the candidates spread over the near-degenerate solutions.  ``eval.jsonl``
records the fraction of standard particles with a candidate within 3% and
1% of the truth, by height and scale.  The shipped model took 40,000 steps
of 512 profiles, about four to five hours on an Apple M-series GPU, and then
10,000 steps (``--init default --lr 3e-4``, about 75 minutes) in which half
of the examples (``--saturated``) are made as a camera that saturates
records them: the rings of the profile that reach the limit of the camera are
marked as not measured, as they are when ``CATCHapp`` leaves the saturated
pixels out.  A model that is trained for another instrument should be
trained that way too.  The limit of the camera is that of an 8-bit camera with
a background of 100, 2.55 times the background; ``--levels LOW HIGH`` draws
it for each example instead, which makes a model for any exposure (the
background at 1/HIGH to 1/LOW of the full scale), at a price
(:doc:`evaluation`).

Use the new models with
``CATCHapp(lmodel='runs/localizer', emodel='runs/estimator')`` and check
them with the evaluation harness (:doc:`evaluation`).

.. automodule:: CATCH.customization.training.frames

.. autoclass:: CATCH.customization.training.frames.Ranges

.. autoclass:: CATCH.customization.training.frames.Frames
   :members: frame, particles, imperfect, scale

.. autofunction:: CATCH.customization.training.localizer.train

.. autofunction:: CATCH.customization.training.estimator.train
