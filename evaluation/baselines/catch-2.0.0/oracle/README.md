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
