'''Finds the holograms of particles in a frame and their centers

The detector (:class:`CATCH.lib.networks.DetectorNet`) marks the cell of
4 x 4 pixels that holds each center and the position of the center inside
it.  The center is then polished with the inversion symmetry of the
hologram (the peak of its masked autocorrelation in a window of 256
pixels), which is precise for most particles but strays for the large
holograms of particles far above the focal plane; the polished center is
kept only within :attr:`Localizer.guard` of the detector's.
'''
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import cv2
import numpy as np
import pandas as pd
import torch

from CATCH.lib.networks import DetectorNet, peaks

#: columns of the output
COLUMNS = ('label', 'confidence', 'x_p', 'y_p')
#: side of the window of the symmetry [pixels], and the largest shift
WINDOW, LIMIT = 256, 12


def default_device() -> torch.device:
    '''The fastest device: CUDA, Apple's GPU, or the CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def normalize(image: np.ndarray) -> np.ndarray:
    '''A hologram normalized to a background of 1

    Holograms on the scale of CATCH 2 (background 100) are divided by 100.
    '''
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    if np.median(image) > 10.:
        image = image / 100.
    return image


def model_directory(kind: str, model: str | Path | None) -> Path:
    '''Directory of a trained model

    A name is a directory ``models/<kind>/<name>`` of the package; a path to
    a directory (such as the output of ``CATCH.training``) is used as it
    is.  Either holds ``weights.pt`` and ``configuration.json``.
    '''
    if model is not None and Path(model).is_dir():
        return Path(model)
    path = Path(__file__).parent / 'models' / kind / (model or 'default')
    if not path.is_dir():
        raise FileNotFoundError(f'no {kind} model {model!r} ({path})')
    return path


def _subpixel(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0. if d == 0 else float(np.clip(.5 * (a - c) / d, -.5, .5))


def symmetry_center(image: np.ndarray, x: float, y: float,
                    n: int = WINDOW, limit: int = LIMIT
                    ) -> tuple[float, float]:
    '''Center of inversion symmetry of the hologram near (x, y)

    The autocorrelation of the hologram with itself, inverted, peaks at
    twice the displacement of the center; it is normalized by the overlap
    of the pixels inside the frame, so that a hologram cut by an edge is
    not pulled inward.

    Parameters
    ----------
    image : numpy.ndarray
        Normalized hologram.
    x, y : float
        Starting center [pixels].
    n : int
        Side of the window [pixels].  Default: 256
    limit : int
        Largest shift of the center [pixels].  Default: 12
    '''
    rows, columns = image.shape
    x0, y0 = int(round(x)), int(round(y))
    top, left = y0 - n // 2, x0 - n // 2
    window = np.hanning(n)
    window = window[None] * window[:, None]
    yy = np.arange(top, top + n)[:, None]
    xx = np.arange(left, left + n)[None, :]
    visible = (xx >= 0) & (xx < columns) & (yy >= 0) & (yy < rows)
    crop = np.ones((n, n))
    r0, r1 = max(top, 0), min(top + n, rows)
    c0, c1 = max(left, 0), min(left + n, columns)
    if r1 > r0 and c1 > c0:
        crop[r0 - top:r1 - top, c0 - left:c1 - left] = image[r0:r1, c0:c1]
    c = (crop - 1.) * window * visible
    m = window * visible
    A = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(c) ** 2, s=(n, n)))
    B = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(m) ** 2, s=(n, n)))
    C = A / (B + .02 * B.max())
    h, L = n // 2, 2 * limit
    part = C[h - L:h + L + 1, h - L:h + L + 1]
    i, j = np.unravel_index(np.argmax(part), part.shape)
    I, J = i + h - L, j + h - L
    return (x0 + (J - h + _subpixel(C[I, J - 1], C[I, J], C[I, J + 1])) / 2,
            y0 + (I - h + _subpixel(C[I - 1, J], C[I, J], C[I + 1, J])) / 2)


class Localizer:
    '''Finds the holograms of particles and their centers

    Parameters
    ----------
    model : str | pathlib.Path | None
        Name of a trained model of the package (``models/localizer/<model>``)
        or the path of a directory with ``weights.pt`` and
        ``configuration.json``.  Default: None, which is ``'default'``.
    device : torch.device | str | None
        Where the network runs.  Default: None, the fastest available.

    Attributes
    ----------
    threshold : float
        Smallest score of a detection.
    guard : float
        Largest distance [pixels] between the polished center and the
        detector's center for the polished center to be used.
    scales : tuple of float
        Range of the scale (pixel of the camera in pixels of the reference
        instrument) that the detector was trained on; a frame outside it
        is resized into it.
    '''

    def __init__(self, model: str | Path | None = None,
                 device: torch.device | str | None = None) -> None:
        self.model = model or 'default'
        self.device = torch.device(device) if device else default_device()
        path = model_directory('localizer', self.model)
        config = json.loads((path / 'configuration.json').read_text())
        self.threshold = float(config['threshold'])
        self.guard = float(config['guard'])
        self.scales = tuple(config['scales'])
        self.count = int(config.get('count', 40))
        self.net = DetectorNet().to(self.device).eval()
        state = torch.load(path / 'weights.pt', map_location=self.device,
                           weights_only=True)
        self.net.load_state_dict(state)

    @overload
    def __call__(self, images: np.ndarray,
                 scale: float = ...) -> pd.DataFrame: ...

    @overload
    def __call__(self, images: Sequence[np.ndarray],
                 scale: float = ...) -> list[pd.DataFrame]: ...

    def __call__(self, images, scale=1.):
        return self.localize(images, scale)

    def localize(self, images, scale: float = 1.):
        '''Finds the particles in one image or a list of images

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray]
            Normalized hologram(s): background 1, or 100 as in CATCH 2.
        scale : float
            Size of a pixel of the camera in pixels of the reference
            instrument (:func:`CATCH.lib.instruments.pixel_ratio`).
            Default: 1

        Returns
        -------
        features : pandas.DataFrame | list[pandas.DataFrame]
            For each image, one row per particle: ``label``
            (``'hologram'``), ``confidence`` (the score of the detector),
            and the center ``x_p``, ``y_p`` [pixels].
        '''
        if isinstance(images, np.ndarray) and images.ndim == 2:
            return self._one(normalize(images), scale)
        return [self._one(normalize(image), scale) for image in images]

    def _one(self, image: np.ndarray, scale: float) -> pd.DataFrame:
        low, high = self.scales
        # resize into the trained range: a new pixel is ``factor`` pixels
        # of the camera (a scale in reference pixels within the range)
        factor = float(np.clip(scale, low, high)) / scale
        work = image
        if factor != 1.:
            h, w = image.shape
            size = (max(int(round(w / factor)), 32),
                    max(int(round(h / factor)), 32))
            how = cv2.INTER_AREA if factor > 1 else cv2.INTER_CUBIC
            work = cv2.resize(image, size, interpolation=how)
        t = torch.from_numpy(np.ascontiguousarray(work)).to(self.device)
        with torch.no_grad():
            x, y, s = peaks(self.net, t[None], self.count)
        keep = s[0] >= self.threshold
        x, y, s = x[0][keep], y[0][keep], s[0][keep]
        if factor != 1.:
            x, y = (x + .5) * factor - .5, (y + .5) * factor - .5
        rows = []
        for xi, yi, si in zip(x, y, s, strict=True):
            fx, fy = symmetry_center(image, float(xi), float(yi))
            if np.hypot(fx - xi, fy - yi) > self.guard:
                fx, fy = float(xi), float(yi)
            rows.append(dict(label='hologram', confidence=float(si),
                             x_p=fx, y_p=fy))
        return pd.DataFrame(rows, columns=list(COLUMNS))
