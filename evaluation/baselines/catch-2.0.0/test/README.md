# Original CATCH on the frozen test set (detections only so far)

`detections.csv` is what the original CATCH (commit `742e0ae`, same procedure
and software as `../README.md`) found in the 400 scenes of the frozen `test`
set (`python -m CATCH.evaluation.scene_run --set test`): 1277 particles in 400
images, 0.20 s per image.  The scoring against the truth (`particles.csv`,
`scenes.csv`) is not here yet; make it with

    python -m CATCH.evaluation.scene_run --set test --pipeline recorded \
        --detections detections.csv --nprocs 6 --output test_baseline
