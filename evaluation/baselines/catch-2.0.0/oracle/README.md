# Fits from the true centers: the network and a start that needs none

The `oracle` pipeline of `scene_run` knows where the particles are (the true
centers and the size of the crop, as in training) and takes the starting
values of the fit from an estimator.  It measures the estimator, with the
effect of neighbors and edges on its crop, apart from the Localizer.  The
scenes are the 200 of the `dev` set (717 particles in the image); the fit is
the one of `evaluation/fitting.py` (5 nm criterion).

Files (the columns are those of `SceneResults.particles`):

* `particles_network_9starts.csv`: the shipped network, with nine fits of
  each particle: from the estimate and from radii 0.95, 1.05, 0.9, 1.1, 0.8,
  1.2, 0.7 and 1.3 times it, the index moving the other way so that
  `(n_p - n_m) a_p` stays the same.  The fit is the start with the least
  reduced chi-squared; `a_fit_i` and `redchi_i` are those of each start.
  `python -m CATCH.evaluation.scene_run --set dev --pipeline oracle
  --starts 9 --nprocs 8` (about 20 minutes on 8 processes of an Apple-silicon
  laptop).
* `particles_analytic.csv`: the analytic estimator of pylorenzmie (azimuthal
  profile; the index is fixed at 1.5) as the start, one fit.
  `--estimator analytic` (about 5 minutes).

* `particles_network_grid25.csv`: the network with 25 fits of each particle,
  from a grid of 5 radii (0.9 to 1.1 times the estimate, the index moving as
  above) and 5 axial positions (0.7 to 1.3 times), the estimate first;
  `a_fit_i`, `z_fit_i`, `redchi_i` of each.  `--grid` (about 25 minutes).
* `particles_truth.csv`: the true particles as the start, one fit (the
  `truth` pipeline): the ceiling of this setup.

## Results (fraction of the 717 particles that lock on)

| start                                   | locks on | radius within 1% |
|-----------------------------------------|---------:|-----------------:|
| network, one fit (the estimate)         |    16.9% |            19.7% |
| network, best of 3 by chi-squared       |    19.8% |            23.3% |
| network, best of 5                      |    20.9% |            25.1% |
| network, best of 9                      |    22.6% |            27.1% |
| network, some start of the 9 locks on   |    23.8% |                  |
| analytic estimator, one fit             |    14.8% |            17.6% |

* Nine starts against one: +5.7 points (95% interval +4.0 to +7.4), no
  particle is lost and 41 are gained (McNemar p = 9e-13).  Choosing by
  chi-squared gets 22.6 of the 23.8% that the best choice among the nine
  would get, so the selection is not the limit.
* The analytic start (14.8%) and the network (16.9%) do not differ
  significantly (p = 0.21, 56 particles only the analytic start locks on and
  71 only the network).  By radius, locking on: analytic 34, 13, 3, 4, 2% for
  radii of 0-1, 1-2, 2-3, 3-4 and 4+ um; the network 25, 25, 22, 10, 10%
  (one fit).
* What decides the fit is the error of the estimate, above all of `z_p`.
  Locking on by the relative error of `z_p` (best of 9): 60% for 0-2%, 54%
  for 2-5%, 26% for 5-10%, 20% for 10-20%, 7% for 20-50% and 2% above that;
  half of the estimates of `z_p` are off by more than 20% (median 21%; 4% for
  a single particle in a clean crop).  When `z_p` and `a_p` are both within
  5%, 71% lock on (79% if the nearest neighbor is 400 pixels or more away),
  and 15% otherwise.
* Failed fits are not near a solution: the median reduced chi-squared is 51,
  and 1.8 for the fits that lock on.

## The grid of radii and axial positions

| start                                    | locks on | within 1% |
|------------------------------------------|---------:|----------:|
| the estimate (1 fit)                     |    16.9% |     19.7% |
| 5 radii (same z)                         |    20.9% |     25.1% |
| 5 axial positions (same radius)          |    20.8% |     25.8% |
| 9 radii (above)                          |    22.6% |     27.1% |
| **grid of 25**                           | **25.0%**|  **31.2%**|
| some start of the 25 locks on            |    29.8% |           |
| **the true particle (the ceiling)**      | **73.6%**|  **85.4%**|

* The grid against one start: +8.1 points (95% interval +6.1 to +10.1,
  p = 7e-18, no particle lost, 58 gained); against the nine radii: +2.4
  points (p = 0.003).  Choosing by chi-squared gets 25.0 of the 29.8% that the
  best choice would.  Varying `z_p` helps as much as varying the radius, and
  the two together add: the estimates are wrong in both.
* The ceiling is 73.6% (90% for particles whose nearest neighbor is 400
  pixels or more away, 17% under 50 pixels), and does not depend on the
  distance to the edge (70, 76, 73, 74%).  It rises with the radius (45, 63,
  76, 82, 87% for radii of 0-1, 1-2, 2-3, 3-4 and 4+ um), whereas the
  network's start does the opposite (25, 25, 22, 10, 10% from one start;
  32, 30, 33, 14, 21% from the grid).  The loss of the network near the
  edges (35% in the interior, 5-10% within 100 pixels, from one start) is
  therefore in its estimates, not in the fit.
* So a start that costs 25 fits recovers 8 of the 57 points between the
  network's estimate and the ceiling.  Failed fits still end at a median
  reduced chi-squared of 49; only 7% of them have one below 3.  Starts spread
  around a poor estimate are not a substitute for a good estimate, above all
  for large particles, which are the ones that lock on from the true values
  and from none of these starts.

## The loss of the fit

The fits above are plain least squares (`--loss linear`, the default).  The
robust Cauchy loss of `Optimizer(robust=True)` (`--loss cauchy`) down-weights
outliers such as the fringes of a neighbor; `--loss both` fits least squares
first and then refines with the Cauchy loss from that result.  One fit, 717
particles; files `particles_{truth,network}_{cauchy,both}.csv`.

| start                | linear | Cauchy | both  |
|----------------------|-------:|-------:|------:|
| the true particle    |  73.6% |  78.9% | 76.3% |
| the network estimate |  16.9% |  14.4% | 19.1% |

* From the true particle the Cauchy loss gains 5.3 points (95% interval +2.1
  to +8.5, p = 0.002), nearly all of it near a neighbor: with the nearest
  neighbor under 50 pixels, 17% lock on with least squares and 58% with the
  Cauchy loss (37% and 52% for 50-150 pixels); isolated particles are the
  same (90% and 89%).  The ceiling of the setup is therefore 79%.
* From the network's estimate the Cauchy loss alone loses 2.5 points: where
  every residual is large the loss is flat and the fit stalls.  Least squares
  first and then the Cauchy loss gains 2.2 points over least squares (27
  particles gained, 11 lost) and is the best fit from an estimate; from the
  truth it is 2.6 points below Cauchy alone (p = 0.003).
* The other baselines of this directory (nine starts, the grid, the analytic
  start) use least squares.

