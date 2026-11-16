'''The fit that starts from an estimate, and the criterion for locking on'''
from dataclasses import dataclass

import numpy as np

from CATCH.evaluation.cases import Case, Crop

#: a fit locks on if the radius is within this of the truth [um]
TOLERANCE = 0.005
#: number of pixels in the fit
NPIXELS = 4000


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
             tolerance: float = TOLERANCE) -> bool:
    '''True if the fitted radius is within ``tolerance`` of the truth'''
    return bool(abs(a_fit - a_p) < tolerance)


def fit(model, hologram, radius: float = np.inf, seed: int = 0):
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

    Returns
    -------
    result : pandas.Series
        The fitted properties and the reduced chi-squared (``redchi``).
    '''
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
    settings['bounds'] = (lower, upper)
    for p, value, low, high in zip(variables, start, lower, upper,
                                   strict=True):
        if not low < value < high:              # start inside the bounds
            model.properties = {p: min(max(value, low + 1e-6),
                                       high - 1e-6)}
    return optimizer.optimize(hologram)


def refine(job: Job) -> tuple[float, float, float]:
    '''Fits from an estimate; returns the fitted ``(a_p, n_p, redchi)``

    The fit starts at the center of the crop, with the estimated ``z_p``,
    ``a_p`` and ``n_p``.  It returns ``nan`` for a fit that fails.
    '''
    from pylorenzmie.analysis import Hologram
    from pylorenzmie.theory import LorenzMie, Sphere
    hologram = Hologram(job.crop.data)
    model = LorenzMie()
    model.parallel = False          # the jobs run in parallel instead
    model.instrument.noise = job.crop.noise
    model.coordinates = hologram.flat_coordinates
    particle = model.single(Sphere)
    particle.x_p = particle.y_p = job.crop.center
    particle.z_p, particle.a_p, particle.n_p = job.estimate
    try:
        result = fit(model, hologram).to_dict()
        return (float(result['a_p']), float(result['n_p']),
                float(result['redchi']))
    except Exception:           # a fit that fails is a result
        return np.nan, np.nan, np.nan
