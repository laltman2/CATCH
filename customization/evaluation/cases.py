'''Simulated particles and holograms for evaluating estimators'''
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pylorenzmie.theory import LorenzMie, Sphere

from CATCH.customization.evaluation.extent import feature_extent

#: radii of the particles, in micrometers
RADII = (0.5, 1., 2., 3., 4., 5.)
#: refractive indexes of polystyrene and of silica
INDEX = {'ps': 1.59, 'silica': 1.45}
#: range of the axial position, in pixels
Z_RANGE = (150., 600.)
#: offsets of the particle from the center of the crop, in pixels
OFFSETS = (0., 3.)
#: the instrument that CATCH is trained for (the default of pylorenzmie)
INSTRUMENT = {'wavelength': 0.447, 'magnification': 0.048, 'n_m': 1.34}

#: ``'fixed'``: 201 x 201 pixels with 2% noise.  ``'training'``: as CATCH
#: trains: the side is twice the extent of the feature, with a random error
#: of 5%, and the noise is 5%.
Condition = Literal['fixed', 'training']
CONDITIONS = ('fixed', 'training')


@dataclass(frozen=True)
class Case:
    '''A particle and the seed of the noise in its hologram

    Attributes
    ----------
    a_p : float
        Radius [um].
    n_p : float
        Refractive index.
    z_p : float
        Axial position [pixels].
    offset : float
        Offset of the particle from the center of the crop along x [pixels].
    seed : int
        Seed of the random numbers for the crop.
    '''
    a_p: float
    n_p: float
    z_p: float
    offset: float
    seed: int


@dataclass(frozen=True)
class Crop:
    '''A simulated hologram of one particle

    Attributes
    ----------
    data : numpy.ndarray
        The normalized hologram, with noise.
    center : float
        Coordinate of the center of the crop, in pixels, along x and y.
    noise : float
        Standard deviation of the noise.
    '''
    data: np.ndarray
    center: float
    noise: float


def make_cases(ncases: int = 180,
               seed: int = 2026,
               radii: tuple[float, ...] = RADII,
               indexes: dict[str, float] | None = None,
               z_range: tuple[float, float] = Z_RANGE,
               offsets: tuple[float, ...] = OFFSETS) -> list[Case]:
    '''Returns the same number of cases for each radius and index

    Parameters
    ----------
    ncases : int
        Number of cases, to be divided evenly among the radii and the
        indexes; a remainder is dropped.
        Default: 180
    seed : int
        Seed of the random numbers.  Default: 2026
    radii : tuple of float
        Radii [um].  Default: ``RADII``
    indexes : dict
        Refractive indexes by name.  Default: ``INDEX``
    z_range : tuple of float
        Range of the axial position [pixels].  Default: ``Z_RANGE``
    offsets : tuple of float
        The offset of a case from the center is one of these [pixels].
        Default: ``OFFSETS``
    '''
    indexes = INDEX if indexes is None else indexes
    rng = np.random.default_rng(seed)
    per = ncases // (len(radii) * len(indexes))
    cases: list[Case] = []
    for a_p in radii:
        for n_p in indexes.values():
            cases.extend(Case(a_p=a_p, n_p=n_p,
                              z_p=float(rng.uniform(*z_range)),
                              offset=float(rng.choice(offsets)),
                              seed=int(rng.integers(1 << 30)))
                         for _ in range(per))
    return cases


def extent(case: Case) -> float:
    '''Radius [pixels] of the feature of the particle of a case'''
    sphere = Sphere(a_p=case.a_p, n_p=case.n_p, z_p=case.z_p)
    return feature_extent(sphere, {'instrument': INSTRUMENT})


def simulate(case: Case, condition: Condition = 'training') -> Crop:
    '''Returns the hologram of a case

    Parameters
    ----------
    case : Case
        The particle.
    condition : ``'fixed'`` or ``'training'``
        Size and noise of the crop; see ``Condition``.
        Default: ``'training'``
    '''
    rng = np.random.default_rng(case.seed)
    if condition == 'fixed':
        size, noise = 201, 0.02
    elif condition == 'training':
        ext = extent(case)
        size, noise = int(2 * rng.normal(ext, 0.05 * ext)), 0.05
    else:
        raise ValueError(f'unknown condition {condition!r}')
    shape = (size, size)
    center = (size - 1) / 2.
    model = LorenzMie()
    model.instrument.noise = noise
    model.coordinates = model.meshgrid(shape)
    sphere = model.single(Sphere)
    sphere.a_p, sphere.n_p = case.a_p, case.n_p
    sphere.r_p = [center + case.offset, center, case.z_p]
    data = model.hologram().reshape(shape) + noise * rng.normal(size=shape)
    return Crop(data, center, noise)
