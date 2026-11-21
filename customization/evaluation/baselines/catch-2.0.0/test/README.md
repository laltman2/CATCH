# Original CATCH on the frozen test set

The original CATCH (commit `742e0ae`; same software, weights and procedure as
`../README.md`) on the 400 scenes of the frozen `test` set
(`python -m CATCH.customization.evaluation.scene_run --set test`): 1381 particles.  This is
the "before" for the final comparison of a finished version.  The test set is
frozen: compare finished versions on it, and tune on the `dev` set.

Files: `detections.csv` (what it found), `detections.seconds.csv` (time for
each image), `particles.csv` (a row for each true particle) and `scenes.csv`
(a row for each scene).  To score a detections file again:

    python -m CATCH.customization.evaluation.scene_run --set test --pipeline recorded \
        --detections detections.csv --nprocs 6 --chunk 50 --output test_baseline

(Run it from a script or the command line, not from the interactive prompt: the
fits use processes that start afresh, which need `if __name__ == '__main__':`.)

## Results

Standard errors are binomial.

| | test (1381 particles) | dev (717 particles) |
|---|---|---|
| detections | 1277: 1088 true, 189 false | 666: 569 true, 97 false |
| precision | 85.2 +- 1.0% | 85.4 +- 1.4% |
| recall | 78.8 +- 1.1% | 79.4 +- 1.5% |
| median error of the center | 1.9 pixels | 1.9 pixels |
| **fits that lock on, of all the particles** | **15.0 +- 1.0%** (207) | 15.6 +- 1.4% |
| fits that lock on, of those found | 19.0 +- 1.2% | 19.7 +- 1.7% |
| time for an image | 0.20 s | 0.20 s |

The two sets agree within their errors, so the development set is not an odd
one.  By the distance of the center from the nearest edge (recall, lock-on):
under 25 pixels, 49% and 2%; 25 to 100, 63% and 2%; 100 to 250, 85% and 17%;
250 or more, 93% and 27%.  By the distance to the nearest neighbor: under 150
pixels, 50-52% and 0-3%; 150 to 400, 87% and 14%; 400 or more, 79% and 21%.
By the number of particles in the scene: one, 80% and 20%; two, 84% and 25%;
three, 79% and 21%; four or five, 78% and 13%; six or more, 78% and 9%.
