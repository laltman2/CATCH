'''A scan of the chi-squared around an estimate, to choose starts for fits

The estimate of the network is off by 4% in ``z_p`` and ``a_p`` and by 10%
of the contrast of the index, and for large spheres that scatter strongly
the landscape of the chi-squared has many minima, so that a fit from the
estimate, or from a few starts near it, often ends in the wrong one.  The
model costs about half a millisecond to render, so a grid of a few
thousand points around the estimate is cheap: the local minima of the
chi-squared on the grid are the starts of the fits.
'''
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scan:
    '''The grid of a scan, in fractions of the estimate

    Attributes
    ----------
    z_span, a_span, c_span : float
        Half-widths of the grid in ``z_p``, ``a_p`` and the contrast of the
        index ``n_p - n_m``, as fractions of the estimate.  Defaults: 0.25,
        0.12 and 0.35
    spacing : float
        Spacing of the grid points, as a fraction.  Default: 0.025
    keep : int
        Number of local minima that are returned.  Default: 10
    npixels : int
        Number of pixels of the crop that the chi-squared uses, chosen at
        random.  Default: 4000
    clearance : float
        A point is skipped if ``z_p`` times the pixel size is less than
        this many radii, because the model diverges inside the sphere.
        Default: 1.2
    '''
    z_span: float = 0.25
    a_span: float = 0.12
    c_span: float = 0.35
    spacing: float = 0.025
    keep: int = 10
    npixels: int = 4000
    clearance: float = 1.2


def axis(span: float, spacing: float) -> np.ndarray:
    '''Factors from ``1 - span`` to ``1 + span``, with 1 among them'''
    half = max(1, round(span / spacing))
    return 1. + spacing * np.arange(-half, half + 1)


def local_minima(chi: np.ndarray) -> np.ndarray:
    '''Boolean array that marks the points no larger than all their neighbors

    Neighbors are the 26 points that touch a point (fewer at the edge of the
    grid); points where the chi-squared is not finite are not minima.
    '''
    padded = np.pad(chi, 1, constant_values=np.inf)
    low = np.isfinite(chi)
    nx, ny, nz = chi.shape
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            for dz in (0, 1, 2):
                if (dx, dy, dz) != (1, 1, 1):
                    low &= chi <= padded[dx:dx + nx, dy:dy + ny, dz:dz + nz]
    return low


def scan(data: np.ndarray, noise: float, start: Mapping[str, float],
         settings: Scan = Scan(), n_m: float = 1.34,
         pixel: float = 0.048, seed: int = 0
         ) -> tuple[list[dict[str, float]], np.ndarray]:
    '''Starts for fits, from the local minima of the chi-squared of a grid

    Parameters
    ----------
    data : numpy.ndarray
        The normalized hologram (a crop).
    noise : float
        Standard deviation of the noise in the hologram.
    start : mapping
        ``x_p`` and ``y_p`` (pixels, in the crop) and the estimates of
        ``z_p`` (pixels), ``a_p`` [um] and ``n_p``: the center of the grid.
    settings : Scan
        The grid.
    n_m : float
        Refractive index of the medium.  Default: 1.34
    pixel : float
        Size of a pixel [um].  Default: 0.048
    seed : int
        Seed for the choice of the pixels.  Default: 0

    Returns
    -------
    starts : list of dict
        The ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p`` of the local
        minima, the least chi-squared first, at most ``settings.keep``.
    chi : numpy.ndarray
        The reduced chi-squared on the grid, shape ``(nz, na, nc)``
        (infinite where the model was not computed).
    '''
    from pylorenzmie.analysis import Hologram
    from pylorenzmie.theory import LorenzMie, Sphere
    hologram = Hologram(data)
    flat = np.asarray(hologram.flat_data).ravel()
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(flat.size, min(settings.npixels, flat.size),
                               replace=False))
    model = LorenzMie()
    model.parallel = False
    model.instrument.noise = noise
    model.coordinates = np.asarray(hologram.flat_coordinates)[:, index]
    particle = model.single(Sphere)
    particle.x_p, particle.y_p = start['x_p'], start['y_p']
    zs = start['z_p'] * axis(settings.z_span, settings.spacing)
    As = start['a_p'] * axis(settings.a_span, settings.spacing)
    contrast = (start['n_p'] - n_m) * axis(settings.c_span, settings.spacing)
    chi = np.full((zs.size, As.size, contrast.size), np.inf)
    target = flat[index]
    for i, z in enumerate(zs):
        for j, a in enumerate(As):
            if z * pixel < settings.clearance * a:
                continue
            for k, c in enumerate(contrast):
                particle.z_p, particle.a_p, particle.n_p = z, a, n_m + c
                residual = target - np.asarray(model.hologram()).ravel()
                chi[i, j, k] = np.mean((residual / noise) ** 2)
    found = np.argwhere(local_minima(chi))
    order = np.argsort(chi[tuple(found.T)])[:settings.keep]
    starts = [dict(x_p=start['x_p'], y_p=start['y_p'], z_p=float(zs[i]),
                   a_p=float(As[j]), n_p=float(n_m + contrast[k]))
              for i, j, k in found[order]]
    return starts, chi
