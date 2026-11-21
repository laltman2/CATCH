# CATCH

[![Tests](https://github.com/laltman2/CATCH/actions/workflows/test.yml/badge.svg)](https://github.com/laltman2/CATCH/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/laltman2/CATCH)](LICENSE)

**C**haracterizing **A**nd **T**racking **C**olloids **H**olographically

Machine-learning models that find colloidal particles in in-line
holographic microscopy images [2] and estimate each particle's position,
radius and refractive index [3-6].

## What CATCH does
1. A **Localizer**, a heatmap detector based on CenterNet [7], finds the
   center of each holographic feature in a normalized image.
2. An **Estimator** analyzes the pixels comprising each feature and
   proposes values for the associated particle's axial position, `z_p`,
   radius `a_p` and refractive index `n_p`.
   Starting from the feature's radial profile, the network
   proposes 64 candidate estimates for the particle's position and properties.
   Each estimate is checked against the measured profile
   with a generative model based on the Lorenz-Mie theory of light
   scattering [1, 2] and the best is selected for the estimate.
3. Optionally, a **fit** from every start (with
   [pylorenzmie](https://github.com/davidgrier/pylorenzmie)) and a rule that
   chooses among near-degenerate solutions and reports how ambiguous the
   result is.

The networks are trained on synthetic holograms and serve any microscope
whose pixel, in wavelengths of the medium, is 1 to 2.5 times the reference
pixel (0.048 µm at 0.447 µm in water).
Estimates take about 7 to 11 ms per particle on an Apple M-series GPU.

<img src="docs/images/test_image_crop.png" alt="Hologram of a colloidal sphere" width="200"/>

## Installation
CATCH requires Python 3.11 or later and PyTorch. Install it from a checkout:

```sh
git clone https://github.com/laltman2/CATCH.git
pip install ./CATCH
pip install "./CATCH[fit]"      # for the fits (pylorenzmie)
```

Other extras: `train` (pylorenzmie, for training and the evaluation
harness) and `dev` (pytest, coverage and ruff).

## Usage
The models expect a *normalized* hologram: the recorded image divided by
the background (a background of 1, or of 100 as in CATCH 2). Lengths are in
micrometers, except coordinates in the image and `z_p`, which are in pixels.

```python
import cv2
from CATCH import CATCHapp

image = cv2.imread('docs/images/test_image_large.png', cv2.IMREAD_GRAYSCALE)

catch = CATCHapp(instrument=dict(wavelength=0.447, magnification=0.048,
                                 n_m=1.34))
results = catch(image)                    # a list of images works too
fitted = catch(image, fit=True)           # also fit, and choose
polystyrene = catch(image, fit=True, nrange=(1.55, 1.70))   # a prior
```

`results` is a `pandas.DataFrame` with one row per particle:

| column | meaning |
|---|---|
| `label`, `confidence` | `'hologram'`, and the score of the detector |
| `x_p`, `y_p` | center of the hologram in the image, in pixels |
| `bbox` | `((x_0, y_0), w, h)`: twice the radius of the 20th fringe |
| `edge` | `True` if the box crosses an edge of the image |
| `z_p`, `a_p`, `n_p` | the best start: axial position, radius and index |
| `framenum` | index of the image in the list that was analyzed |
| `misfit`, `starts` | misfit of the best start; all the starts, best first |

With `fit=True`: the chosen fit (`x_fit` ... `n_fit`, `redchi`), a `flag`,
and the ambiguity (`nsolutions`, `a_spread`, `n_spread`, `z_spread`, and the
best other solution `alt_a`, `alt_n`, `alt_redchi`).  See the
[documentation](docs/quickstart.rst).

## Training
Most instruments need no training. For the others, or to fine-tune:

```sh
python -m CATCH.training.localizer runs/localizer --steps 20000
python -m CATCH.training.estimator runs/estimator --steps 40000
```

and `CATCHapp(lmodel='runs/localizer', emodel='runs/estimator')`; see
[docs/training.rst](docs/training.rst).

## Development
```sh
python3 -m venv ~/python/venvs/catch      # keep the venv outside the checkout
~/python/venvs/catch/bin/pip install -e "./CATCH[dev]"
~/python/venvs/catch/bin/python -m pytest --cov
ruff check .
```

The tests use models with random weights, not the trained ones. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the style rules and
[CHANGELOG.md](CHANGELOG.md) for what has changed.

## References
1. S.-H. Lee, Y. Roichman, G.-R. Yi, S.-H. Kim, S.-M. Yang, A. van Blaaderen,
   P. van Oostrum and D. G. Grier,
   "Characterizing and tracking single colloidal particles with video
   holographic microscopy,"
   *Optics Express* **15**, 18275 (2007).
   [doi:10.1364/OE.15.018275](https://doi.org/10.1364/OE.15.018275)
2. C. Martin, L. E. Altman, S. Rawat, A. Wang, D. G. Grier and
   V. N. Manoharan,
   "In-line holographic microscopy with model-based analysis,"
   *Nature Reviews Methods Primers* **2**, 83 (2022).
   [doi:10.1038/s43586-022-00165-z](https://doi.org/10.1038/s43586-022-00165-z)
3. A. Yevick, M. Hannel and D. G. Grier,
   "Machine-learning approach to holographic particle characterization,"
   *Optics Express* **22**, 26884 (2014).
   [doi:10.1364/OE.22.026884](https://doi.org/10.1364/OE.22.026884)
4. M. D. Hannel, A. Abdulali, M. O'Brien and D. G. Grier,
   "Machine-learning techniques for fast and accurate feature localization
   in holograms of colloidal particles,"
   *Optics Express* **26**, 15221 (2018).
   [doi:10.1364/OE.26.015221](https://doi.org/10.1364/OE.26.015221)
5. L. E. Altman and D. G. Grier,
   "CATCH: Characterizing and Tracking Colloids Holographically using deep
   neural networks,"
   *The Journal of Physical Chemistry B* **124**, 1602 (2020).
   [doi:10.1021/acs.jpcb.9b10463](https://doi.org/10.1021/acs.jpcb.9b10463)
6. L. E. Altman and D. G. Grier,
   "Machine learning enables precise holographic characterization of
   colloidal materials in real time,"
   *Soft Matter* **19**, 3002 (2023).
   [doi:10.1039/D2SM01283A](https://doi.org/10.1039/D2SM01283A)
7. X. Zhou, D. Wang and P. Krähenbühl,
   "Objects as points,"
   [arXiv:1904.07850](https://arxiv.org/abs/1904.07850) (2019).

## Acknowledgments
This work was initiated with support from the SBIR program of the National
Institutes of Health under Award Number R44TR001590. The Titan Xp and Titan
RTX GPUs used for this work were provided by GPU Grants from NVIDIA.
Development of CATCH v3 was supported by the National Science Foundation
under Award Number DMR-2428983.

## License
CATCH is released under the [MIT License](LICENSE).
