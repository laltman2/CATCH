# Model card: Estimator `default`

| | |
|---|---|
| Used by | `CATCH.Estimator.Estimator` (and `CATCHapp`) |
| Version | CATCH 3.0.0 |
| Weights | `weights.pt`, 21,100,617 bytes, SHA-256 `52dd44892694a0eea0ec0c812254bdbbe3e3e6aa0e597b23a23323eda6ac8c1c` |
| Network | `CATCH.lib.networks.ProfileNet`, 5,267,472 parameters |
| License | MIT, as the package |

## What it does

It reads the radial profile of a hologram (the median of the normalized
image on rings around the center, at 640 radii from 2 to 1,280 pixels of the
reference instrument) and proposes 64 candidates for the height `z_p`, the
radius `a_p` and the index `n_p` of the sphere, each with a logit.  A
candidate is the logarithm of the three properties; the loss lets only the
best candidate of each example learn (winner takes all), so that the 64
spread over the near-degenerate solutions.  The Estimator then checks every
candidate against the profile with a batched Lorenz-Mie model and keeps the
best as the starts of a fit.  The network is not a measurement by itself.

## Intended use

Starts for fits of the Lorenz-Mie theory to the holograms of single colloidal
spheres in water-like media, imaged by in-line holographic microscopy, with
the sizes and pixels below.  Not for particles that are not spheres, for
clusters, or for holograms of several overlapping particles taken as one.

## Training data

Synthetic only, rendered with the Lorenz-Mie theory (pylorenzmie): spheres of
radius 0.2 to 5.5 um, index 1.38 to 2.5 and height 5 to 320 um (at the
wavelength 0.447 um, in a medium of index 1.34, with 0.048 um per pixel: the
*reference instrument*), pixels of 1 to 2.5 reference pixels, noise of 1.5
to 6%.  What measurement adds to a profile (noise, neighbors, edges, the
sampling of the camera) comes from the profiles of 1,500 synthetic frames of
one to six particles, which are added to clean profiles of random particles.
Half of the examples of the second stage below are made as a camera that
saturates at 2.55 times the background (an 8-bit camera with a background of
100) records them.  No real image was used for training.

## Training

Two stages: 40,000 steps of 512 profiles (about 5 hours on an Apple M-series
GPU), then 10,000 steps (about 75 minutes) from those weights at a learning
rate of 3e-4 with half of the examples saturated.

**Provenance.** The shipped first stage was trained by development scripts
that are not part of this repository; the second stage was made with the
committed command, `python -m CATCH.customization.training.estimator OUT
--steps 10000 --init STAGE1 --lr 3e-4`.  The committed commands reproduce the
model: stage 1 is `python -m CATCH.customization.training.estimator OUT
--steps 40000 --saturated 0` (no saturated examples), followed by the stage-2
command above.  A run of exactly these two commands, from scratch, gave
weights (SHA-256 `7432a1b3...`, not identical to the shipped ones, since the
runs are not bit-for-bit repeatable) whose fits from the best start lock on
for 67.1% of the 1,381 particles of the frozen test set, against 67.8 +- 1.3%
for the shipped weights, with the same precision and recall of detection
(98.8%).  The five-start fits were not rerun.

## Evaluation (simulated scenes of the reference instrument, unless said)

A fit from the best start locks on (radius within 5 nm of the truth) for
67.8% of the 1,381 particles of the frozen test set (a fit from the true
particles: 76.8%), 72.8% from the five starts of `CATCHapp(fit=True)`.  By
radius, 0.5 to 5 um, from the best start: 42, 68, 79, 64, 77 and 78%
(from the truth 43, 69, 84, 88, 88 and 90%).  With the Estimator given the
true centers on 200 scenes: 69% unclipped, 69% when the scenes are clipped as
an 8-bit camera records them and the saturated pixels are left out.

On 80 real frames of lens-test videos (0.120 um per pixel; none saturated) it
agrees with the streaming analysis: a median radius of 1.53 um for 3 um
silica (nominal 1.5) and 0.48 um for 1 um polystyrene (nominal 0.5).

## Known limitations

* Near-degenerate solutions: nearly the same hologram belongs to other sizes
  and indexes, most for polystyrene of 3 um, and the best start by the
  profile is often not the right one.  A prior on the index (`nrange`) and
  the fits from several starts help; `nsolutions` counts what the starts
  found.
* Index above 1.8 is outside the standard range that is preferred.
* The saturation training used the limit 2.55 only.  For a camera whose
  limit is lower (a brighter background) the starts degrade, from 75% lock-on
  at 2.55 to 65% at 1.6 and 43% at 1.3 in simulation; the fits do not.
* Only simulated scenes of a few instruments and real frames of one
  microscope have been tested.  Check the results for another instrument.
* Particles within 100 pixels of an edge or 150 pixels of a neighbor, or
  closer than 10 um to the focal plane, are harder.
* A hologram whose signal-to-noise ratio (`snr`) is below 100 cannot be
  analyzed; see `CATCHapp.quality_of`.

## Citation

See `CITATION.cff`.  The original CATCH: L. E. Altman and D. G. Grier, *The
Journal of Physical Chemistry B* **124**, 1602 (2020),
[doi:10.1021/acs.jpcb.9b10463](https://doi.org/10.1021/acs.jpcb.9b10463).
