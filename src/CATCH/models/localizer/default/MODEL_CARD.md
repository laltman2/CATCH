# Model card: Localizer `default`

| | |
|---|---|
| Used by | `CATCH.Localizer.Localizer` (and `CATCHapp`) |
| Version | CATCH 3.0.0 |
| Weights | `weights.pt`, 2,839,511 bytes, SHA-256 `9ba144999d713c998edff31ec8f4606c792b2a579ef7c8ae1358e992d3b39f35` |
| Network | `CATCH.lib.networks.DetectorNet`, 694,491 parameters |
| License | MIT, as the package |

## What it does

It finds the holograms of particles in a whole frame.  The network marks, for
each cell of 4 x 4 pixels, the logit that the center of a hologram falls in
it and the position of the center inside the cell (a heatmap with offsets,
after CenterNet), through an encoder to 1/32 of the resolution with dilated
blocks that see the whole of a large hologram.  Peaks above the threshold of
0.35 are the detections (at most 40 for a frame); the center of each is
polished with the inversion symmetry of the hologram (a masked
autocorrelation, which is not part of the weights) and kept only within 1
pixel of the network's.  A frame whose pixel is outside 1 to 2.5 reference
pixels is resized into that range.

## Intended use

Centers of the holograms of colloidal spheres in in-line holographic
microscopy frames of about 1,000 x 1,300 pixels, as the first stage of
`CATCHapp`.  It does not tell a particle from an imaging artifact that looks
like one; the `confidence` is the score of the detector, and the detections
with a confidence below 0.5 of the tutorial's example are artifacts.

## Training data

Synthetic only: frames of 1,024 x 1,280 pixels with 0 to 10 spheres of radius
0.2 to 5.5 um, heights of 5 to 330 um (log-uniform), index contrast 0.04 to
1.16 (log-uniform, so that many are faint), noise of 1.5 to 7%, pixels drawn
for each frame between 1 and 2.5 times the reference pixel (0.048 um at 0.447
um in a medium of index 1.34), and imperfect normalization: a black level, a
smooth gain, and faint straight fringes.  No real image was used.

## Training

Two stages, about 4 hours on an Apple M-series GPU in all: 20,000 steps of 4
frames of the heatmap alone at a learning rate of 1e-3, then 8,000 steps from
those weights that add the offset head, at 3e-4.

**Provenance.** Both stages were trained by development scripts that are not
part of this repository.  The committed command, `python -m
CATCH.customization.training.localizer OUT --steps 20000`, implements the
same network, loss and data in one stage, with the offsets learned together
with the heatmap.  A run of the committed command from scratch has not been
made, so these weights are not known to be reproducible from this repository.

## Evaluation (simulated scenes of the reference instrument, unless said)

On the frozen test set (400 scenes, 1,381 particles, found within 15 pixels
of the truth): recall 98.8%, precision 98.9%, median error of the center 0.08
pixels.  On the 200 scenes of the development set: recall 99.2% unclipped,
99.0% when clipped as an 8-bit camera records them and 98.9% at a limit of 1.3
times the background; the median error of the center is 0.08 to 0.09 pixels at
every limit (95th percentile 1.8 pixels, 3.3 where 10 to 20% of the box is
saturated), and the precision falls only under heavy clipping, to 97.9% at
1.6 and 95.8% at 1.3, from false detections on clipped blobs.  A frame of
1,024 x 1,280 pixels takes about 45 to 55 ms on the GPU of an Apple M1 Pro.

On 80 real frames of lens-test videos (0.120 um per pixel) it found every
particle that the streaming analysis fitted.

## Known limitations

* It cannot tell an imaging artifact from a particle; use `confidence`.
* Only simulated scenes of a few instruments and real frames of one
  microscope have been tested.
* Particles far from focus, whose holograms are large and faint, are the
  harder ones, and detections near the edge of a frame are less accurate.

## Citation

See `CITATION.cff`.  The original CATCH: L. E. Altman and D. G. Grier, *The
Journal of Physical Chemistry B* **124**, 1602 (2020),
[doi:10.1021/acs.jpcb.9b10463](https://doi.org/10.1021/acs.jpcb.9b10463).
