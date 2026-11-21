# Where the estimate goes wrong: shifts, neighbors, and the network itself

The shipped network (`default`) on the 717 particles of the `dev` set,
estimated under four conditions (`python -m CATCH.customization.evaluation.diagnose`, about
a minute; `errors_network.csv` has the error of each particle and condition):

* `scene`: the crop of the scene as CATCH makes it, shifted inside the image
  near an edge;
* `padded`: the crop of the scene centered on the particle and padded with
  background where it leaves the image (`pipelines.pad_around`);
* `alone`: the particle alone in an image of the same shape, at the same
  position (same crop and shift, no neighbors);
* `centered`: the particle alone at the center of the image (a clean crop, as
  in training).

## All particles

| condition | median `z_p` error | median `a_p` error | `z_p` within 10% | `a_p` within 5% |
|-----------|-------:|-------:|-------:|-------:|
| scene     | 20.6%  | 23.8%  | 36.8%  | 23.6%  |
| padded    | 10.8%  |  7.0%  | 48.0%  | 41.4%  |
| alone     |  9.9%  | 19.6%  | 50.1%  | 29.4%  |
| centered  |  2.6%  |  3.3%  | 97.9%  | 73.8%  |

## By what is in the crop

A crop is *shifted* if the particle is closer to an edge than half the side
of the crop (67% of the particles of the dev set: the crops, twice the extent
of the feature, are large compared with the images), and *crowded* if another
particle is within a crop side.  Fraction with `z_p` within 10%:

| shifted | crowded | n   | scene | padded | alone | centered |
|---------|---------|----:|------:|-------:|------:|---------:|
| no      | no      |  70 | 97%   | 97%    | 96%   | 96%      |
| no      | yes     | 166 | 54%   | 54%    | 99%   | 99%      |
| yes     | no      | 166 | 28%   | 47%    | 27%   | 97%      |
| yes     | yes     | 315 | 19%   | 35%    | 26%   | 98%      |

* Where the crop is neither shifted nor crowded the network is accurate (median
  error of `z_p` 2.5%); on clean centered crops it is accurate everywhere
  (2.6%, and 94% of the radii within 5% above 3 um).  The weights are not
  the problem: the training distribution is.
* The shift is the largest effect.  The particle alone, with no neighbors, in
  a shifted crop has a median `z_p` error of 30% and 27% within 10%; the error
  grows with the shift (66% within 10% for no shift, 22% for a shift of
  0.1-0.2 of the side, 15% above 0.3).  The network was trained on particles
  at the center and is not told where the particle is.
* Neighbors are the second effect: in unshifted crops they take the `z_p`
  within 10% from 99% to 54%.
* Padding with background keeps the particle at the center and recovers about
  half of the loss from the shift (28% to 47% and 19% to 35%).  A better
  recovery needs a network that is trained on crops with neighbors and
  truncated fringes.
* Small particles (under 2 um) have `a_p` within 5% for only 37-42% of the
  clean crops: that is the degeneracy of the radius and the index, not the
  position.

## Lock-on of the fit that starts from the estimate

One fit, least squares then Cauchy (`--loss both`), `PYLORENZMIE_BACKEND=torch`,
estimates from true centers:

| crop given to the network | locks on | within 1% |
|---------------------------|---------:|----------:|
| shifted inside the image (CATCH) | 17.4% | 22.2% |
| padded (`scene_run --pad`)       | 27.2% | 34.3% |

Padding gains 9.8 points (95% interval +7.2 to +12.3, p = 2.5e-14, 81
particles gained and 11 lost).  By distance to the edge (under 25, 25-100,
100-250 and 250+ pixels): 6, 5, 14, 36% shifted and 16, 19, 29, 36% padded.
(`particles_network_shifted_torch.csv` and `particles_network_padded.csv`.)

## The centers of the Localizer at the edges

The original Localizer (YOLOv5, `find_center`) on the 200 dev scenes, each
particle matched to the nearest detection within 100 pixels
(`python -m CATCH.customization.evaluation.centers`, `centers_yolo.csv`).  Recall by the
distance of the particle to the nearest edge, for a center within 15 pixels
(the gate of the harness) and within 100:

| distance to the edge | 15 px | 100 px | middle of the box, 15 px |
|----------------------|------:|-------:|-------:|
| under 25 px          | 58.7% | 84.1%  |  0.0%  |
| 25-100 px            | 64.4% | 84.6%  |  0.0%  |
| 100-250 px           | 81.8% | 91.1%  | 25.6%  |
| 250 px and more      | 96.2% | 96.2%  | 96.2%  |

* `find_center` does its job.  YOLO's box is cut by the border for a particle
  near an edge, so the middle of the box is far from the particle (median
  error 55 px); the correction brings it to 3 px, and it is better for 99% of
  the boxes that it corrects.  The boxes are the right size: the side is 0.97
  of twice the extent of the feature.
* It fails at the corners.  Of the 397 boxes that the border cuts, a box cut
  on one axis (321) has a median error of 2.9 px (3% over 15 px), and a box
  cut on both (76) 47 px (89% over 15 px): the size of the feature can only
  be read from the dimension that is not cut, so at a corner `max(w, h)` is
  too small (0.84 of twice the extent) and the center is not shifted far
  enough on either axis.  A better centroid alone raises the recall within
  100 px of an edge from about 62% to about 84%.
* The rest is not found at all: 16% of the particles within 100 px of an
  edge, and 4% of the others, have no detection within 100 pixels.

### Fixing the corners

The corner particles are the 120 of the dev set whose true box crosses both
an x and a y border.  For the boxes that the border cuts on both axes, four
ways to set the size `s` of the feature, and so the center (`x_p = x_max -
s/2` for a box cut at the left, and so on); the boxes cut on one axis and the
uncut ones are not touched (`corner_options.csv`, one row for each particle and
rule, matched to the nearest detection within 100 pixels):

* `current`: `find_center`, which takes `s = max(w, h)`.
* `A prior`: `s = max(w, h) / 0.84`, the measured median shortfall of the
  box at a corner (not tuned: 0.78 and 0.90 give 17% and 27% below).
* `B`: the estimator's extent: the crop centered on the center of A (padded),
  estimated, `s = 0.97 * 2 * extent(z_p, a_p, n_p)`, and the center again
  (three iterations).
* `O`: the true size `s = 0.97 * 2 * extent`, which no program knows: what is
  left if the size were right.

| rule             | found within 100 px | center within 15 px | median error |
|------------------|--------------------:|--------------------:|-------------:|
| current          | 73%                 | 10.8%               | 45 px        |
| A prior (0.84)   | 85%                 | 24.2%               | 28 px        |
| B, one iteration | 85%                 | 27.5%               | 26 px        |
| B, three         | 81%                 | 21.7%               | 28 px        |
| O, the true size | 89%                 | 58.3%               | 12 px        |

Within 100 px of an edge (251 particles) the center is within 15 px for 62.9%
with `current`, 67.7% with A, 68.5% with B (one iteration) and 81.7% with O;
nothing changes for the other particles.

* Both options help by about 13-17 points of the corner particles, which is
  about 2 points of all the particles, and they do not differ (the count is
  120, so a difference of 4 points is noise).  A needs nothing but the box and
  can live in `find_center`; B needs the estimator, which is poor for a
  particle that is not at the center of its crop, and does not improve with
  iteration.
* Neither is close to the ceiling, and the ceiling is low.  Even with the true
  size, 42% of the corner centers are more than 15 pixels off: the inner
  edges of YOLO's boxes are unbiased (median error 1 px) but scatter by
  about 15 pixels (10th to 90th percentile -13 to +18) for a feature that is
  cut by the border, and the center inherits that.  A better rule on the
  boxes cannot fix it; a detector that reads the center from the fringes can.

