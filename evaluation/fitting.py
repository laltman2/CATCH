'''The fit that starts from an estimate, and the criterion for locking on'''
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from CATCH.evaluation.cases import Case, Crop

#: a fit locks on if the radius is within this of the truth [um]
TOLERANCE = 0.005
#: a fit is *close* if the radius is within this fraction of the truth: the
#: radius-aware criterion, because 5 nm is 0.1% of a radius of 5 um
RELATIVE = 0.01
#: factors of the radius of the starts of a fit from several starts, in the
#: order that they are used: the estimate first, then alternately smaller and
#: larger (the estimates are off by 5% in the median)
SPREAD = (1., 0.95, 1.05, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3)
#: starts that vary the axial position too, as pairs of factors of the
#: radius and of ``z_p``, the estimate first: the estimates of ``z_p`` are
#: off by 20% in the median and often by much more
GRID = tuple((a, z)
             for a, z in sorted(((a, z)
                                 for a in (1., 0.95, 1.05, 0.9, 1.1)
                                 for z in (1., 0.85, 1.15, 0.7, 1.3)),
                                key=lambda f: (f != (1., 1.),
                                               abs(f[0] - 1) + abs(f[1] - 1))))
#: number of pixels in the fit
NPIXELS = 4000
#: the losses of a fit: plain least squares, the robust Cauchy loss, and
#: both in turn (least squares to reach the basin, then Cauchy to refine it
#: without the pull of outliers such as the fringes of a neighbor)
LOSSES = ('linear', 'cauchy', 'both')
#: the loss of the fits of the scene harness: the best from an estimate (the
#: single-particle harness, ``run.py``, keeps least squares, which
#: reproduces its baseline)
SCENE_LOSS = 'both'


@dataclass(frozen=True)
class Job:
    '''A fit to be made

    Attributes
    ----------
    case : Case
        The particle.
    crop : Crop
        The hologram of the particle.
    estimate : tuple of float
        Starting values of ``(z_p, a_p, n_p)``.
    '''
    case: Case
    crop: Crop
    estimate: tuple[float, float, float]


def locks_on(a_fit: float, a_p: float,
             tolerance: float = TOLERANCE,
             relative: float | None = None) -> bool:
    '''True if the fitted radius is within ``tolerance`` of the truth

    Parameters
    ----------
    a_fit, a_p : float
        The fitted and the true radius [um].
    tolerance : float
        Largest error [um].  Default: ``TOLERANCE``
    relative : float or None
        If given, the largest error as a fraction of ``a_p`` in place of
        ``tolerance``.  Default: None
    '''
    if relative is not None:
        tolerance = relative * a_p
    return bool(abs(a_fit - a_p) < tolerance)


def spread_start(start: Mapping[str, float], factor: float,
                 n_m: float, z_factor: float = 1.) -> dict[str, float]:
    '''A start with the radius scaled by ``factor``, along the degeneracy

    The hologram of a small sphere constrains mostly the product of its
    radius and the contrast of its index, so the index moves the other way:
    ``(n_p - n_m) * a_p`` stays the same.

    Parameters
    ----------
    start : mapping
        The ``a_p``, ``n_p`` and the other starting values.
    factor : float
        Scale of the radius.
    n_m : float
        Refractive index of the medium.
    z_factor : float
        Scale of the axial position.  Default: 1
    '''
    out = dict(start)
    out['z_p'] = start['z_p'] * z_factor
    out['a_p'] = start['a_p'] * factor
    out['n_p'] = n_m + (start['n_p'] - n_m) / factor
    return out


def fit(model, hologram, radius: float = np.inf, seed: int = 0,
        loss: str = 'linear'):
    '''Least-squares fit of a model to a hologram, with bounds

    Fits the pixels within ``radius`` of the particle, at most ``NPIXELS`` of
    them.  The bounds keep a fit from running off to unphysical values: the
    radius, the index and the height are positive, and the center stays
    within 15 pixels of where it started.

    Parameters
    ----------
    model : pylorenzmie.theory.LorenzMie
        Model with one particle, set to the starting values.
    hologram : pylorenzmie.analysis.Hologram
        The data.
    radius : float
        Distance from the particle [pixels].  Default: all the pixels.
    seed : int
        Seed for the choice of the pixels.  Default: 0
    loss : str
        ``'linear'`` (least squares), ``'cauchy'`` (robust) or ``'both'``:
        a least-squares fit, then a robust fit from its result.  Default:
        ``'linear'``

    Returns
    -------
    result : pandas.Series
        The fitted properties and the reduced chi-squared (``redchi``).
    '''
    if loss not in LOSSES:
        raise ValueError(f'loss must be one of {LOSSES}, not {loss!r}')
    from pylorenzmie.analysis import Mask, Optimizer
    particle = model.single()
    x0, y0 = particle.x_p, particle.y_p
    x, y = hologram.coordinates
    exclude = np.hypot(x - x0, y - y0) > radius
    npix = int((~exclude).sum())
    mask = Mask(shape=hologram.shape,
                fraction=min(1., NPIXELS / max(npix, 1)),
                exclude=exclude, seed=seed)
    optimizer = Optimizer(model=model, mask=mask)
    limits = dict(x_p=(x0 - 15, x0 + 15), y_p=(y0 - 15, y0 + 15),
                  z_p=(20., 3000.), a_p=(0.1, 10.),
                  n_p=(model.instrument.n_m + 0.02, 2.6))
    variables = list(optimizer.variables)
    start = [model.properties[p] for p in variables]
    lower = [limits[p][0] for p in variables]
    upper = [limits[p][1] for p in variables]
    settings = optimizer.settings
    settings['method'] = 'trf'
    settings['loss'] = 'cauchy' if loss == 'cauchy' else 'linear'
    settings['bounds'] = (lower, upper)
    for p, value, low, high in zip(variables, start, lower, upper,
                                   strict=True):
        if not low < value < high:              # start inside the bounds
            model.properties = {p: min(max(value, low + 1e-6),
                                       high - 1e-6)}
    result = optimizer.optimize(hologram)
    if loss == 'both':
        fitted = result.to_dict()
        model.properties = {p: float(fitted[p]) for p in variables}
        settings['loss'] = 'cauchy'
        result = optimizer.optimize(hologram)
    return result


def fit_particle(data: np.ndarray, noise: float,
                 start: Mapping[str, float],
                 loss: str = 'linear') -> dict[str, float]:
    '''Fits a model of one particle to a hologram, from a starting point

    Parameters
    ----------
    data : numpy.ndarray
        The normalized hologram (a crop of the image).
    noise : float
        Standard deviation of the noise in the hologram.
    start : mapping
        Starting values of ``x_p`` and ``y_p`` (pixels, in the crop),
        ``z_p`` (pixels), ``a_p`` (micrometers) and ``n_p``.
    loss : str
        ``'linear'``, ``'cauchy'`` or ``'both'`` (see ``fit``).  Default:
        ``'linear'``

    Returns
    -------
    result : dict
        The fitted ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p`` and the
        reduced chi-squared ``redchi``.  All are ``nan`` for a fit that
        fails.
    '''
    from pylorenzmie.analysis import Hologram
    from pylorenzmie.theory import LorenzMie, Sphere
    hologram = Hologram(data)
    model = LorenzMie()
    model.parallel = False          # the jobs run in parallel instead
    model.instrument.noise = noise
    model.coordinates = hologram.flat_coordinates
    particle = model.single(Sphere)
    particle.x_p, particle.y_p = start['x_p'], start['y_p']
    particle.z_p, particle.a_p, particle.n_p = (start['z_p'], start['a_p'],
                                                start['n_p'])
    keys = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'redchi')
    try:
        redchi0 = start_redchi(model, hologram, noise)
        result = fit(model, hologram, loss=loss).to_dict()
        out = {key: float(result[key]) for key in keys}
        out['redchi0'] = redchi0
        return out
    except Exception:           # a fit that fails is a result
        return dict.fromkeys((*keys, 'redchi0'), np.nan)


def start_redchi(model, hologram, noise: float,
                 npixels: int = NPIXELS) -> float:
    '''Reduced chi-squared of the model as it is, before any fit

    A measure of the quality of a start that does not depend on the fit: it
    is large for a start far from the data and about 1 for the truth.  It
    uses at most ``npixels`` pixels, evenly spaced.
    '''
    data = np.asarray(hologram.flat_data).ravel()
    step = max(1, data.size // npixels)
    model.coordinates = hologram.flat_coordinates[:, ::step]
    residual = data[::step] - np.asarray(model.hologram()).ravel()
    model.coordinates = hologram.flat_coordinates
    return float(np.mean((residual / noise) ** 2))


def refine(job: Job) -> tuple[float, float, float]:
    '''Fits from an estimate; returns the fitted ``(a_p, n_p, redchi)``

    The fit starts at the center of the crop, with the estimated ``z_p``,
    ``a_p`` and ``n_p``.  It returns ``nan`` for a fit that fails.
    '''
    z_p, a_p, n_p = job.estimate
    start = dict(x_p=job.crop.center, y_p=job.crop.center,
                 z_p=z_p, a_p=a_p, n_p=n_p)
    result = fit_particle(job.crop.data, job.crop.noise, start)
    return result['a_p'], result['n_p'], result['redchi']
