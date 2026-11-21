'''CATCH: detect, localize and characterize colloidal particles in
holographic video microscopy images'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from CATCH.Estimator import Estimator, ring_radius
from CATCH.instruments import REFERENCE, pixel_ratio, to_reference
from CATCH.Localizer import Localizer, default_device, normalize
from CATCH.selection import COLUMNS as FIT_COLUMNS
from CATCH.selection import select

Images = np.ndarray | Sequence[np.ndarray]


def noise_of(image: np.ndarray) -> float:
    '''Standard deviation of the noise of a normalized hologram

    From the median absolute deviation of the differences of neighboring
    pixels, which the smooth fringes hardly affect.
    '''
    d = np.diff(np.asarray(image, float), axis=1).ravel()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


class CATCHapp:
    '''Detect, localize and characterize colloidal particles
    in holographic video microscopy images.

    The Localizer finds the holograms of particles and their centers; the
    Estimator proposes, for each, the starts of a fit of the Lorenz-Mie
    theory and reports the best; optionally, the fits are made and the
    best is chosen (``fit=True``, which needs pylorenzmie).

    Parameters
    ----------
    lmodel : str | pathlib.Path | None
        Trained model of the Localizer: a name or a directory.  Default:
        None, the default model.
    emodel : str | pathlib.Path | None
        Trained model of the Estimator: a name or a directory.  Default:
        None, the default model.
    instrument : mapping | None
        ``wavelength`` and ``magnification`` [um] and ``n_m`` of the
        microscope.  Default: None, the reference instrument
        (:data:`CATCH.instruments.REFERENCE`).
    device : torch.device | str | None
        Where the networks run.  Default: None, the fastest available.
    nprocs : int
        Processes for the fits.  Default: 1
    compile : bool
        Compile the inner loops of the Estimator with ``torch.compile``.
        Default: True

    Attributes
    ----------
    localizer : Localizer
        Finds the particles.
    estimator : Estimator
        Proposes the starts of the fits.
    '''

    def __init__(self,
                 lmodel: str | Path | None = None,
                 emodel: str | Path | None = None,
                 instrument: Mapping[str, float] | None = None,
                 device: torch.device | str | None = None,
                 nprocs: int = 1, compile: bool = True) -> None:
        self.device = torch.device(device) if device else default_device()
        self.instrument = dict(instrument or REFERENCE)
        self.localizer = Localizer(lmodel, self.device)
        self.estimator = Estimator(emodel, self.device, compile)
        self.nprocs = int(nprocs)
        self._fitter = None

    @property
    def scale(self) -> float:
        '''Pixel of the camera in pixels of the reference instrument'''
        return pixel_ratio(self.instrument, self.estimator.reference)

    def __call__(self, images: Images | None = None,
                 **kwargs) -> pd.DataFrame:
        return self.analyze(images, **kwargs)

    def analyze(self, images: Images | None = None, fit: bool = False,
                nrange: tuple[float, float] | None = None,
                explore: int = 0,
                noise: float | None = None) -> pd.DataFrame:
        '''Detect the particles in images and characterize them

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray] | None
            Normalized hologram(s): background 1, or 100 as in CATCH 2.
            Default: None, which returns an empty DataFrame.
        fit : bool
            Fit each particle from all of its starts and choose the best
            (:func:`CATCH.selection.select`).  Needs pylorenzmie.
            Default: False
        nrange : tuple of float | None
            Prior range of ``n_p`` for the sample (in its medium).
            Default: None, the standard range (``n_p`` up to 1.8 relative
            to the medium of the reference).
        explore : int
            With a prior, more starts of distinct ``a_p`` inside it.
            Default: 0
        noise : float | None
            Standard deviation of the noise, for the fits.  Default: None,
            estimated from each image.

        Returns
        -------
        results : pandas.DataFrame
            One row for each particle: ``label``, ``confidence``, the
            center ``x_p``, ``y_p`` [pixels], ``bbox`` (``((x0, y0), w,
            h)``, a square of twice the radius of the 20th fringe of the
            estimate), ``edge`` (True if the box crosses an edge of the
            image), the estimates ``z_p`` [pixels], ``a_p`` [um] and
            ``n_p``, their ``misfit``, ``starts`` (all the starts,
            ``(z_p, a_p, n_p)``, best first) and ``framenum``.  With
            ``fit=True``, also the columns of
            :data:`CATCH.selection.COLUMNS`.
        '''
        if images is None:
            return pd.DataFrame()
        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]
        results = []
        for n, raw in enumerate(images):
            image = normalize(raw)
            features = self.localizer(image, self.scale)
            if len(features) == 0:
                continue
            estimates = self.estimator(image, features.x_p, features.y_p,
                                       self.instrument, nrange, explore)
            result = pd.concat([features.reset_index(drop=True),
                                estimates.reset_index(drop=True)], axis=1)
            sides = self.sides(result)
            result['bbox'] = [((int(round(x - s / 2)), int(round(y - s / 2))),
                               s, s) for x, y, s in
                              zip(result.x_p, result.y_p, sides,
                                  strict=True)]
            h, w = image.shape
            result['edge'] = [bool(x0 < 0 or y0 < 0 or x0 + s > w
                                   or y0 + s > h)
                              for (x0, y0), s, _ in result.bbox]
            result['framenum'] = n
            if fit:
                result = self._fit(image, result, sides, nrange, noise)
            results.append(result)
        if not results:
            return pd.DataFrame()
        out = pd.concat(results, ignore_index=True)
        first = ['label', 'confidence', 'x_p', 'y_p', 'bbox', 'edge',
                 'z_p', 'a_p', 'n_p', 'framenum']
        return pd.DataFrame(
            out[first + [c for c in out.columns if c not in first]])

    def sides(self, result: pd.DataFrame) -> list[int]:
        '''Side of the box of each particle [pixels]'''
        sides = []
        for z, a, n in zip(result.z_p, result.a_p, result.n_p, strict=True):
            if not np.isfinite(z):
                sides.append(101)
                continue
            z_ref = to_reference(z, a, n, self.instrument,
                                 self.estimator.reference)[0]
            radius = ring_radius(z_ref, self.scale,
                                 reference=self.estimator.reference)
            sides.append(max(int(2 * radius), 101))
        return sides

    def _fit(self, image: np.ndarray, result: pd.DataFrame,
             sides: list[int], nrange: tuple[float, float] | None,
             noise: float | None) -> pd.DataFrame:
        from CATCH.fitter import Fitter
        if (self._fitter is None or self._fitter.instrument != self.instrument
                or self._fitter.nprocs != self.nprocs):
            if self._fitter is not None:
                self._fitter.close()
            self._fitter = Fitter(self.instrument, nprocs=self.nprocs)
        fitter = self._fitter
        sigma = noise if noise is not None else noise_of(image)
        particles = list(zip(result.x_p, result.y_p, sides, result.starts,
                             strict=True))
        fits = fitter(image, sigma, particles)
        chosen = pd.DataFrame([select(f, nrange) for f in fits],
                              columns=list(FIT_COLUMNS))
        return pd.concat([result, chosen], axis=1)


def example():  # pragma: no cover
    from pathlib import Path

    import cv2

    image_file = (Path(__file__).parent / 'docs' / 'images' /
                  'test_image_large.png')
    image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f'{image_file}: the example images are in the repository, '
            'not in an installed package')
    catch = CATCHapp()
    print(catch(image))


if __name__ == '__main__':  # pragma: no cover
    example()
