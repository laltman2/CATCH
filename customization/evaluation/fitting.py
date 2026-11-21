'''The fit that starts from an estimate, and the criterion for locking on'''
from collections.abc import Mapping
from dataclasses import dataclass


from CATCH.customization.evaluation.cases import Case, Crop
from CATCH.lib.fitter import (LOSSES, NPIXELS, fit,  # noqa: F401
                          fit_particle, start_redchi)

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
#: starts that vary the contrast of the index too, as triples of factors of
#: the radius, of ``z_p`` and of ``n_p - n_m``, the estimate first (the
#: estimates of the index are off by 10% of the contrast in the median)
GRID_N = tuple((a, z, c)
               for a, z, c in sorted(((a, z, c)
                                      for a in (1., 0.95, 1.05)
                                      for z in (1., 0.85, 1.15, 0.7, 1.3)
                                      for c in (1., 0.9, 1.1)),
                                     key=lambda f: (f != (1., 1., 1.),
                                                    sum(abs(x - 1)
                                                        for x in f))))
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
                 n_m: float, z_factor: float = 1.,
                 contrast: float = 1.) -> dict[str, float]:
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
    contrast : float
        Scale of the contrast of the index, ``n_p - n_m``, in addition to
        the move along the degeneracy.  Default: 1
    '''
    out = dict(start)
    out['z_p'] = start['z_p'] * z_factor
    out['a_p'] = start['a_p'] * factor
    out['n_p'] = n_m + (start['n_p'] - n_m) * contrast / factor
    return out


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
