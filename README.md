# CATCH

[![Tests](https://github.com/laltman2/CATCH/actions/workflows/test.yml/badge.svg)](https://github.com/laltman2/CATCH/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/laltman2/CATCH)](LICENSE)

**C**haracterizing **A**nd **T**racking **C**olloids **H**olographically

Machine-learning models that detect colloidal particles in in-line
holographic microscopy images and estimate each particle's three-dimensional
position, radius and refractive index.

## What CATCH does
CATCH is a two-stage pipeline:

1. A **Localizer**, a [YOLOv5](https://github.com/ultralytics/yolov5)
   detector, finds the hologram of each particle in a normalized image and
   reports its center and bounding box.
2. An **Estimator**, a convolutional neural network written in PyTorch,
   analyzes a crop around each hologram and estimates the particle's axial
   position `z_p`, radius `a_p` and refractive index `n_p`.

Both are trained on synthetic holograms computed with the Lorenz-Mie theory
of light scattering (see [pylorenzmie](https://github.com/davidgrier/pylorenzmie)).
The estimates are fast, and they can serve as the starting point for a
least-squares fit of a Lorenz-Mie model to the hologram.

<img src="docs/images/test_image_crop.png" alt="Hologram of a colloidal sphere" width="200"/>

## Installation
CATCH requires Python 3.11 or later. It is not on PyPI yet; install it from
a checkout:

```sh
git clone https://github.com/laltman2/CATCH.git
pip install ./CATCH
```

The Localizer needs YOLOv5, which is an optional extra because this version
of CATCH requires `yolov5 <= 6.0.7`:

```sh
pip install "./CATCH[yolo]"
```

Other extras: `train` (pylorenzmie and matplotlib, for generating training
data), `gpu` (CuPy), `wandb` and `dev` (pytest, coverage and ruff).

`import CATCH` does not import YOLOv5. `CATCH.Estimator` works without it,
and `CATCH.Localizer` and `CATCH.CATCHapp` load it on first use.

## Usage
The models expect a *normalized* hologram: the recorded image divided by
the background, scaled so that the mean is 100 in 8-bit units. Lengths are
in micrometers, except for coordinates in the image, which are in pixels.
The default models were trained on spheres in water
(`n_m = 1.34`) imaged at a wavelength of 0.447 µm and a magnification of
0.048 µm/pixel, with radii of 0.2 to 5 µm and refractive indexes of 1.38 to
2.5.

### The whole pipeline
```python
import cv2
from CATCH import CATCHapp

image = cv2.imread('docs/images/test_image_large.png', cv2.IMREAD_GRAYSCALE)

catch = CATCHapp()
results = catch(image)        # a list of images works too
print(results)
```

`results` is a `pandas.DataFrame` with one row per detected particle:

| column | meaning |
|---|---|
| `label`, `confidence` | class name and confidence of the detection |
| `x_p`, `y_p` | center of the feature in the image, in pixels |
| `bbox` | `((x_0, y_0), w, h)`, the bounding box in pixels |
| `edge` | `True` if the box crosses an edge of the image |
| `z_p`, `a_p`, `n_p` | estimated axial position, radius and refractive index |
| `framenum` | index of the image in the list that was analyzed |

The center of a feature that crosses an edge can lie outside the image.

### One stage at a time
```python
from CATCH import Localizer, Estimator, CATCHapp

features = Localizer()(image)               # DataFrame of detections
crops = CATCHapp.crop(image, features)      # one square crop per feature
estimates = Estimator()(crops)              # DataFrame of z_p, a_p, n_p
```

The Estimator takes square crops of any size; it resizes them to the size
of the network input and gives the network the crop size as a second input.
Another model can be chosen by name: `Estimator(model='boltzmann')`.
Models are directories in `cfg_estimator/` and `cfg_yolov5/` that hold
`weights.pt` and, for the Estimator, `configuration.json`.

A tutorial notebook is in [tutorials/Using_CATCH.ipynb](tutorials/Using_CATCH.ipynb).

## Training
Training scripts and their configuration files are in `training/`; see
[training/HowToTrain.md](training/HowToTrain.md). They need the `train`
extra (pylorenzmie 3.7 or later). Data generation works with the current
pylorenzmie; the training scripts themselves have not been run since the
port. Retraining, and moving the Localizer to Ultralytics, are planned; see
[ToDo.md](ToDo.md).

## Development
```sh
python3 -m venv ~/python/venvs/catch      # keep the venv outside the checkout
~/python/venvs/catch/bin/pip install -e "./CATCH[dev]"
~/python/venvs/catch/bin/python -m pytest --cov
ruff check .
```

The tests mock YOLOv5 and the trained weights. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the style rules and
[CHANGELOG.md](CHANGELOG.md) for what has changed.

## Acknowledgments
This work was primarily supported by the SBIR program of the National
Institutes of Health under Award Number R44TR001590. The Titan Xp and Titan
RTX GPUs used for this work were provided by GPU Grants from NVIDIA.

## License
CATCH is released under the [MIT License](LICENSE).
