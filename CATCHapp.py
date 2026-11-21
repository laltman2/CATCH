'''CATCH: detect, localize and characterize colloidal particles in
holographic video microscopy images'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from CATCH.Estimator import Estimator, ring_radius
from CATCH.lib.instruments import REFERENCE, pixel_ratio, to_reference
from CATCH.Localizer import Localizer, default_device, normalize
from CATCH.lib.selection import COLUMNS as FIT_COLUMNS
from CATCH.lib.selection import select

Images = np.ndarray | Sequence[np.ndarray]

#: a hologram whose signal-to-noise ratio (``snr``) is below this cannot be
#: analyzed; on simulated scenes the radius is within 1% for at most 16% of
#: the particles, and within 5 nm for 9%, against 82% and 54% for a ratio
#: of 400 to 800
LOW_SNR = 100.
#: a box with more of its pixels saturated than this cannot be trusted; the
#: starts of the Estimator, which was trained for the 8-bit limit of the
#: camera, lock on for about 40% of the particles with 5 to 10% saturated
#: and 26% with 10 to 20%, against 75% with less than 5%
MAX_SATURATED = 0.05


def quality_of(snr: float, saturated: float) -> str:
    '''Whether a particle can be trusted, and if not why

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio of the hologram (``nan`` without an estimate).
    saturated : float
        Fraction of the box that the camera saturated.

    Returns
    -------
    quality : str
        ``''`` if there is no reason for doubt; otherwise the reasons,
        separated by commas: ``'no estimate'``, ``'low snr'`` (``snr``
        below :data:`LOW_SNR`) and ``'saturated'`` (``saturated`` above
        :data:`MAX_SATURATED`).
    '''
    if not np.isfinite(snr):
        return 'no estimate'
    reasons = []
    if saturated > MAX_SATURATED:
        reasons.append('saturated')
    if snr < LOW_SNR:
        reasons.append('low snr')
    return ', '.join(reasons)


def noise_of(image: np.ndarray) -> float:
    '''Standard deviation of the noise of a normalized hologram

    From the median absolute deviation of the differences of neighboring
    pixels, which the smooth fringes hardly affect.

    Parameters
    ----------
    image : numpy.ndarray
        Normalized hologram (background 1).

    Returns
    -------
    noise : float
        Standard deviation of the noise, in units of the background.
    '''
    d = np.diff(np.asarray(image, float), axis=1).ravel()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def saturation_of(raw: Any, saturation: Any, index: int = 0,
                  count: int = 1) -> float | np.ndarray | None:
    '''What is saturated in an image: a level, a mask of pixels, or nothing

    Parameters
    ----------
    raw : numpy.ndarray
        The image as it was given.
    saturation : float | numpy.ndarray | sequence | None
        What the caller said (see :meth:`CATCHapp.analyze`): a level, a
        mask of the pixels that are not to be used (a boolean array of the
        shape of the image), several masks (a sequence, or an array with
        one more axis, one for each image), or None.
    index : int
        Number of the image among ``count`` images.  Default: 0
    count : int
        Number of images.  Default: 1

    Returns
    -------
    saturated : float | numpy.ndarray | None
        A level, in units of the background, or a boolean mask whose
        pixels are left out, or None for no limit.  An image of ``uint8``
        with no ``saturation`` given has the mask of its pixels at 255, the
        limit of the camera, which is exact whatever the background.

    Raises
    ------
    ValueError
        If there are several masks and their number is not ``count``.
    '''
    if saturation is None:
        pixels = np.asarray(raw)
        if pixels.dtype != np.uint8:
            return None
        if pixels.ndim == 3:
            pixels = pixels[..., 0]
        return pixels >= 255
    if isinstance(saturation, (list, tuple)) or (
            isinstance(saturation, np.ndarray) and saturation.ndim == 3):
        if len(saturation) != count:
            raise ValueError(f'{len(saturation)} masks of saturated pixels '
                             f'for {count} images')
        return saturation_of(raw, saturation[index])
    value = np.asarray(saturation)
    if value.ndim == 0:
        return None if np.isinf(value) else float(value)
    return value.astype(bool)


def saturated_fraction(image: np.ndarray, bbox: tuple,
                       saturated: float | np.ndarray | None) -> float:
    '''Fraction of a box of the image that a camera saturated

    Parameters
    ----------
    image : numpy.ndarray
        Normalized hologram.
    bbox : tuple
        ``((x0, y0), w, h)``, as in the output of
        :meth:`CATCHapp.analyze`; the part inside the image counts.
    saturated : float | numpy.ndarray | None
        The brightest pixel that the camera records, or the boolean mask of
        the pixels that are left out, or None for no limit (see
        :func:`saturation_of`).

    Returns
    -------
    fraction : float
        The pixels at or above the level, or in the mask; 0 without a limit.
    '''
    if saturated is None:
        return 0.
    (x0, y0), w, h = bbox
    rows, columns = image.shape
    window = (slice(max(y0, 0), min(y0 + h, rows)),
              slice(max(x0, 0), min(x0 + w, columns)))
    if isinstance(saturated, np.ndarray):
        part = saturated[window]
        return float(np.mean(part)) if part.size else 0.
    part = image[window]
    return float(np.mean(part >= saturated - 1e-6)) if part.size else 0.


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
        (:data:`CATCH.lib.instruments.REFERENCE`).
    device : torch.device | str | None
        Where the networks run.  Default: None, the fastest available.
    nprocs : int
        Processes for the fits.  They start afresh, so a script that uses
        more than one must run under ``if __name__ == '__main__':``.
        Default: 1
    compile : bool
        Compile the inner loops of the Estimator with ``torch.compile``.
        Default: True
    rings : int
        Bright rings of each hologram that the box and the fits use: the
        crop of a particle reaches the ``rings``-th ring.  Fewer rings
        make the fit less sensitive to background errors, aberrations and
        neighbors.  Default: 10
    npixels : int
        Pixels in each fit, a random subset of its crop.  Default: 4000

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
                 nprocs: int = 1, compile: bool = True, rings: int = 10,
                 npixels: int = 4000) -> None:
        self.device = torch.device(device) if device else default_device()
        self.instrument = dict(instrument or REFERENCE)
        self.localizer = Localizer(lmodel, self.device)
        self.estimator = Estimator(emodel, self.device, compile)
        self.nprocs = int(nprocs)
        self.rings = int(rings)
        self.npixels = int(npixels)
        self._fitter = None

    @property
    def scale(self) -> float:
        '''Pixel of the camera in pixels of the reference instrument'''
        return pixel_ratio(self.instrument, self.estimator.reference)

    def __call__(self, images: Images | None = None,
                 **kwargs: Any) -> pd.DataFrame:
        '''Detect and characterize the particles in images

        Equivalent to :meth:`analyze`, to which the arguments go.
        '''
        return self.analyze(images, **kwargs)

    def analyze(self, images: Images | None = None, fit: bool = False,
                nrange: tuple[float, float] | None = None,
                explore: int = 0,
                noise: float | None = None,
                saturation: Any = None) -> pd.DataFrame:
        '''Detect the particles in images and characterize them

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray] | None
            Normalized hologram(s): background 1, or 100 as in CATCH 2.
            Default: None, which returns an empty DataFrame.
        fit : bool
            Fit each particle from all of its starts and choose the best
            (:func:`CATCH.lib.selection.select`).  Needs pylorenzmie.
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
        saturation : float | numpy.ndarray | sequence | None
            The pixels that are not to be used, because a camera that
            saturates records its limit, not the hologram.  They are left
            out of the profiles and the fits of the particles.  Either the
            brightest pixel that the camera records, in units of the
            background (a number, 2.55 for 8 bits with a background of 100),
            or, better, a boolean mask of the shape of the image whose
            ``True`` pixels are left out: after the division by a background
            that is not uniform, the saturated pixels are not all at one
            level, but they are exactly those that equal the full scale of
            the raw frame.  A single mask applies to every image; for
            several masks, a sequence (or an array with one more axis) with
            one for each image.  ``numpy.inf`` means that no pixel is
            saturated.  Default: None, which is the mask of the pixels at 255
            for an image of ``uint8`` and no limit for any other.

        Returns
        -------
        results : pandas.DataFrame
            One row for each particle: ``label``, ``confidence``, the
            center ``x_p``, ``y_p`` [pixels], ``bbox`` (``((x0, y0), w,
            h)``, a square of twice the radius of the ``rings``-th bright
            ring of the estimate), ``edge`` (True if the box crosses an
            edge of the image), ``saturated`` (the fraction of the box that
            the camera saturated), ``snr`` (the signal-to-noise ratio of the
            hologram in the box, see
            :meth:`CATCH.Estimator.Estimator.signal_to_noise`), ``quality``
            (``''``, or why not to trust the particle: ``'low snr'``,
            ``'saturated'``; see :func:`quality_of`), the
            estimates ``z_p`` [pixels], ``a_p`` [um] and ``n_p``, their
            ``misfit``, ``starts`` (all the starts,
            ``(z_p, a_p, n_p)``, best first) and ``framenum``.  With
            ``fit=True``, also the columns of
            :data:`CATCH.lib.selection.COLUMNS`.
        '''
        if images is None:
            return pd.DataFrame()
        if isinstance(images, np.ndarray) and images.ndim == 2:
            images = [images]
        images = list(images)
        results = []
        for n, raw in enumerate(images):
            image = normalize(raw)
            level = saturation_of(raw, saturation, n, len(images))
            if isinstance(level, np.ndarray) and level.shape != image.shape:
                raise ValueError('the mask of the saturated pixels has the '
                                 f'shape {level.shape}, the image '
                                 f'{image.shape}')
            sigma = noise if noise is not None else noise_of(image[::4])
            features = self.localizer(image, self.scale)
            if len(features) == 0:
                continue
            estimates = self.estimator(image, features.x_p, features.y_p,
                                       self.instrument, nrange, explore,
                                       level, sigma, self.rings)
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
            result['saturated'] = [saturated_fraction(image, box, level)
                                   for box in result.bbox]
            result['quality'] = [quality_of(s, f) for s, f in
                                 zip(result.snr, result.saturated,
                                     strict=True)]
            result['framenum'] = n
            if fit:
                result = self._fit(image, result, sides, nrange, sigma,
                                   level)
            results.append(result)
        if not results:
            return pd.DataFrame()
        out = pd.concat(results, ignore_index=True)
        first = [c for c in ('label', 'confidence', 'x_p', 'y_p', 'bbox',
                             'edge', 'saturated', 'snr', 'quality', 'z_p',
                             'a_p', 'n_p', 'framenum') if c in out.columns]
        return pd.DataFrame(
            out[first + [c for c in out.columns if c not in first]])

    def sides(self, result: pd.DataFrame) -> list[int]:
        '''Side of the box of each particle [pixels]

        The box reaches the ``rings``-th bright ring of the hologram, but
        is never smaller than 101 pixels.

        Parameters
        ----------
        result : pandas.DataFrame
            Output of :meth:`analyze`, with the columns ``z_p``, ``a_p``
            and ``n_p`` in the units of the instrument.

        Returns
        -------
        sides : list[int]
            One side for each row of ``result``; 101 for a row without a
            finite ``z_p``.
        '''
        sides = []
        for z, a, n in zip(result.z_p, result.a_p, result.n_p, strict=True):
            if not np.isfinite(z):
                sides.append(101)
                continue
            z_ref = to_reference(z, a, n, self.instrument,
                                 self.estimator.reference)[0]
            radius = ring_radius(z_ref, self.scale, self.rings,
                                 reference=self.estimator.reference)
            sides.append(max(int(2 * radius), 101))
        return sides

    def _fit(self, image: np.ndarray, result: pd.DataFrame,
             sides: list[int], nrange: tuple[float, float] | None,
             noise: float | None,
             saturation: float | np.ndarray | None = None) -> pd.DataFrame:
        from CATCH.lib.fitter import Fitter
        if (self._fitter is None or self._fitter.instrument != self.instrument
                or self._fitter.nprocs != self.nprocs
                or self._fitter.npixels != self.npixels):
            if self._fitter is not None:
                self._fitter.close()
            self._fitter = Fitter(self.instrument, nprocs=self.nprocs,
                                  npixels=self.npixels)
        fitter = self._fitter
        sigma = noise if noise is not None else noise_of(image)
        particles = list(zip(result.x_p, result.y_p, sides, result.starts,
                             strict=True))
        fits = fitter(image, sigma, particles, saturation)
        chosen = pd.DataFrame([select(f, nrange) for f in fits],
                              columns=list(FIT_COLUMNS))
        return pd.concat([result, chosen], axis=1)


def example() -> None:  # pragma: no cover
    '''Analyze the example hologram of the repository and print the result

    Raises
    ------
    FileNotFoundError
        If the example image is missing, as it is in an installed package.
    '''
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
