# Baseline: CATCH 2.0.0, before the 3.0 work

How the original CATCH (a YOLOv5 Localizer and the default Estimator) does on
simulated scenes with several particles.  The numbers are the "before" of any
comparison with a later version, so everything needed to reproduce them is
here.

## What was evaluated

* **Code:** the original CATCH, commit `742e0ae` of `laltman2/CATCH`
  (`master` before the packaging, port and evaluation work), with its
  weights: Localizer `cfg_yolov5/small_noisy/weights.pt`, Estimator
  `cfg_estimator/default`.  The current code (`CATCH.CATCHapp`, same weights)
  gives identical detections on these scenes (666 particles, largest
  difference 0.0), because the work since then does not change the models.
* **Scenes:** 200 simulated scenes of 1024 x 1280 pixels with 1 to 6
  particles each (717 particles), noise 0.05, seed 2026.  Radii 0.5, 1, 2, 3,
  4 and 5 um; refractive index 1.59 or 1.45 in water (n_m = 1.34);
  z_p of 150 to 600 pixels; the fields of the particles add.  The centers
  are in the image (`margin` 0) and at least 20 pixels apart.
* **Software:** Python 3.14.10, torch 2.15.0, torchvision 0.29.2, numpy
  2.5.5, pandas 3.1.1, opencv-python 5.0.0.93, scipy 1.18.1, yolov5 6.0.7,
  pylorenzmie 3.7.0, numba 0.68.1, on a Mac with an Apple processor, CPU only.
  The old code needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` with this torch (the
  current code does not).
* **Matching and fit:** a detection belongs to a particle if it is within
  15 pixels; the fit starts from the detection, on a crop of twice the
  extent of the estimated feature, and locks on if the fitted radius is
  within 5 nm of the truth (see `docs/evaluation.rst`).

## How to reproduce

```sh
# 1. the images of the scenes
python -m CATCH.customization.evaluation.scene_run --write-images images \
    --nscenes 200 --nparticles 1 6

# 2. what the code under test finds; PYTHONPATH holds the directory that
#    contains the checkout, which is named CATCH
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=/path/to/old \
    python CATCH/evaluation/detect_images.py images detections.csv

# 3. score it
python -m CATCH.customization.evaluation.scene_run --pipeline recorded \
    --detections detections.csv --nscenes 200 --nparticles 1 6 \
    --nprocs 6 --output baseline
```

The files here are `detections.csv` (what the old code found),
`detections.seconds.csv` (the time for each image), `particles.csv` (a row for
each true particle) and `scenes.csv` (a row for each scene).

## Results

Standard errors are binomial.

| | |
|---|---|
| particles | 717 in 200 scenes |
| detections | 666: 569 true, 97 false |
| precision | 85.4 +- 1.4% |
| recall | 79.4 +- 1.5% |
| median error of the center | 1.9 pixels |
| median relative error of the estimates (z, a, n), of those found | 18%, 17%, 7% |
| **fits that lock on, of all the particles** | **15.6 +- 1.4%** (112) |
| fits that lock on, of those that were found | 19.7 +- 1.7% |
| time for an image (Localizer and Estimator) | 0.20 s |

By the number of particles in the scene (recall, lock-on): one, 87% and 40%;
two, 83% and 19%; three, 82% and 15%; four or five, 80% and 16%; six or more,
76% and 11%.  By the distance of the center from the nearest edge: under 25
pixels, 59% and 10%; 25 to 100, 64% and 3%; 100 to 250, 82% and 15%; 250 or
more, 96% and 30%.  By the distance to the nearest neighbor: under 150
pixels, 50-54% and 0%; 150 to 400, 87% and 14%; 400 or more, 80% and 23%.

For comparison, a fit that starts from the true particles locks on 73% of the
time in scenes of 1 to 6 particles (97% for one), so the loss is in the
network, and in the neighbors for the fit that does not model them.

The scenes are simulated, which is how CATCH is trained and what makes it
possible to cover the range of particles.  They use the instrument that the
shipped models were trained for (wavelength 0.447 um, 0.048 um per pixel, a
medium of index 1.34) and the range of particles of the training, so the
numbers say nothing about other instruments or about particles outside the
range.
