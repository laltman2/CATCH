# CATCH 3.0.0 on the frozen test set

The finished CATCH 3.0.0 (the Localizer `models/localizer/default`, the
Estimator `models/estimator/default`) on the 400 scenes of the frozen `test`
set (1381 particles), with exactly the procedure of the CATCH 2 baseline
(`../catch-2.0.0/test/README.md`): the images written as 8-bit PNGs with a
background of 100, CATCH run on them, and a fit from each detection's
estimate (the best start), on a crop of twice the estimated extent, with
the loss `both`.  A fit locks on if the radius is within 5 nm of the truth.

    python -m CATCH.customization.evaluation.scene_run --set test --write-images IMAGES
    python CATCH/customization/evaluation/detect_images.py IMAGES test/detections.csv
    python -m CATCH.customization.evaluation.scene_run --set test --pipeline recorded \
        --detections test/detections.csv --nprocs 6 --chunk 50 \
        --output test/test
    python -m CATCH.customization.evaluation.scene_run --set test --pipeline truth \
        --nprocs 6 --chunk 50 --output test/truth

Software: Python 3.14.10, torch 2.15.0, pylorenzmie 3.7.0 (numba backend for
the fits), on a Mac with an Apple processor; the networks ran on its GPU.

## Results

Standard errors are binomial.  *Standard* particles are at least 100
pixels from an edge and 150 pixels from a neighbor (787 of the 1381).

| | CATCH 2.0.0 | CATCH 3.0.0 | from the truth |
|---|---|---|---|
| precision | 85.2 +- 1.0% | 98.9% | |
| recall | 78.8 +- 1.1% | 98.8% | |
| median error of the center | 1.9 pixels | 0.08 pixels | |
| **fits that lock on, all particles** | **15.0 +- 1.0%** (207) | **52.9 +- 1.3%** (730) | 76.8 +- 1.1% (1061) |
| fits that lock on, standard particles | 25.0% | 58.6% | 83.5% |
| fits that lock on, the others | 1.7% | 45.3% | 68.0% |
| time for an image (no fits) | 0.20 s (CPU) | 0.10 s (GPU) | |

By radius (lock-on, %):

| a_p [um] | 0.5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| CATCH 2.0.0 | 20 | 32 | 17 | 13 | 7 | 1 |
| CATCH 3.0.0 | 43 | 70 | 80 | 56 | 46 | 29 |
| from the truth | 43 | 69 | 84 | 88 | 88 | 90 |

Up to 2 um CATCH 3 is at the limit that the fits allow; the remaining gap
is in the large spheres (3 to 5 um, which in these scenes are within 7 to
29 um of the focal plane), where nearly the same holograms belong to other
sizes and indexes and the best start by the radial profile is often not the
right one.

**With the fits** (`CATCHapp(fit=True)`: a fit from every start, 5 per
particle, and `selection.select`; `fit_particles.csv`, made by the script
`devel/test_fit.py` of the development tree): 53.0% (732) lock on, the
same as from the best start; 114 of the 1365 particles found are reported
ambiguous (more than one solution within 0.5% of the reduced chi-squared),
and they lock on for 12% against 57% for the others.  The fits took 2.3
hours on 6 processes (the defaults of that commit: crops of 20 rings and up
to 501 pixels, 4,000 pixels per fit, 5 starts; the default is now 10 rings
and no cap, see the CHANGELOG).

Files in `test/`: `detections.csv` and `detections.seconds.csv` (what CATCH
found and the time per image), `test_particles.csv` and `test_scenes.csv`
(scored), `truth_particles.csv` and `truth_scenes.csv` (the fits from the
true particles), `fit_particles.csv` (with the fits and the choice), and the
printed reports.

The scenes are simulated, at the reference instrument, with indexes of
1.45 and 1.59 and heights of 150 to 600 pixels; the numbers say nothing
about other instruments, other ranges or real data (see the documentation
for the tests on real frames).
