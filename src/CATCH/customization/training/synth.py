'''Synthetic frames for training, made on the fly

Frames of particles at random positions, including at the edges and the
corners of the image and with their holograms overlapping.  Nothing is
written to disk: the frames are drawn from a random number generator, so
that the data are never repeated and the seed names them.  The holograms
are computed by whichever backend ``PYLORENZMIE_BACKEND`` names (the
``torch`` one is the fast one on a GPU, in single precision).

The random numbers of the evaluation sets of
``CATCH.customization.evaluation.sets`` are made from their own seeds
(:data:`FORBIDDEN_SEEDS`); the training refuses those seeds, so the
training data cannot be the data that versions are compared on.
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pylorenzmie.theory import LorenzMie, Particle, Sphere

from CATCH.customization.evaluation.cases import INSTRUMENT
from CATCH.customization.evaluation.extent import Extent
from CATCH.customization.evaluation.scenes import describe

@dataclass(frozen=True)
class Distribution:
    '''The frames: how many particles, where, and what they are

    Attributes
    ----------
    shape : tuple of int
        ``(rows, columns)`` of a frame [pixels].  Default: (1024, 1280)
    nparticles : tuple of int
        The number of particles is uniform from the first number to the
        second, inclusive.  Default: (0, 10)
    a_p : tuple of float
        Range of the radius [um], uniform in the logarithm.
        Default: (0.2, 5)
    n_p : tuple of float
        Range of the refractive index, uniform in ``n_p - n_m``.
        Default: (1.38, 2.5)
    z_p : tuple of float
        Range of the axial position [pixels], always positive (the particles
        are above the focal plane); 104 pixels is 5 um at 48 nm per pixel,
        the nearest that the instrument reaches.  Default: (104, 600)
    clearance : float
        The axial position is at least ``clearance * a_p / magnification``,
        so that the plane of the image is outside the sphere: inside it the
        Lorenz-Mie series diverges (values of 1e20 and more, and NaN in
        single precision).  The evaluation sets have 1.44 at the least.
        Default: 1.2
    noise : tuple of float
        Range of the standard deviation of the noise, uniform for each
        frame.  Default: (0.03, 0.07)
    margin : float
        A center may be this far outside the image [pixels], so that the
        feature of a particle that is not labeled is in the frame.
        Default: 100
    edge_bias : float
        Probability that a particle is moved to within ``band`` of the
        nearest edge.  Default: 0.3
    corner_bias : float
        Probability that a particle is moved to within ``band`` of the
        nearest corner.  Default: 0.1
    band : float
        Width of the band at the edge [pixels].  Default: 150
    min_separation : float
        Smallest lateral distance between two centers [pixels].  Default: 20
    max_attempts : int
        Positions to try for each particle.  Default: 1000
    '''
    shape: tuple[int, int] = (1024, 1280)
    nparticles: tuple[int, int] = (0, 10)
    a_p: tuple[float, float] = (0.2, 5.)
    n_p: tuple[float, float] = (1.38, 2.5)
    z_p: tuple[float, float] = (104., 600.)
    clearance: float = 1.2
    noise: tuple[float, float] = (0.03, 0.07)
    margin: float = 100.
    edge_bias: float = 0.3
    corner_bias: float = 0.1
    band: float = 150.
    min_separation: float = 20.
    max_attempts: int = 1000
    n_m: float = INSTRUMENT['n_m']
    magnification: float = INSTRUMENT['magnification']


@dataclass(frozen=True)
class Frame:
    '''A frame and the particles in it

    Attributes
    ----------
    image : numpy.ndarray
        The normalized hologram with noise (float32, mean about 1).
    particles : pandas.DataFrame
        ``x_p``, ``y_p``, ``z_p``, ``a_p``, ``n_p``, ``extent`` (the radius
        of the feature [pixels]) and the columns of
        ``evaluation.scenes.describe`` (``edge``, ``neighbor``, ``inside``).
    noise : float
        Standard deviation of the noise.
    '''
    image: np.ndarray
    particles: pd.DataFrame
    noise: float


def sample_particles(rng: np.random.Generator,
                     dist: Distribution) -> pd.DataFrame:
    '''The particles of a frame: ``x_p``, ``y_p``, ``z_p``, ``a_p``, ``n_p``

    Raises
    ------
    RuntimeError
        If no position is far enough from the other particles.
    '''
    rows, columns = dist.shape
    count = int(rng.integers(dist.nparticles[0], dist.nparticles[1] + 1))
    positions: list[tuple[float, float]] = []
    for _ in range(count):
        for _ in range(dist.max_attempts):
            x = float(rng.uniform(-dist.margin, columns + dist.margin))
            y = float(rng.uniform(-dist.margin, rows + dist.margin))
            draw = rng.random()
            if draw < dist.corner_bias:
                x = _toward_edge(x, columns, dist.band, rng)
                y = _toward_edge(y, rows, dist.band, rng)
            elif draw < dist.corner_bias + dist.edge_bias:
                if rng.random() < 0.5:
                    x = _toward_edge(x, columns, dist.band, rng)
                else:
                    y = _toward_edge(y, rows, dist.band, rng)
            if all(np.hypot(x - u, y - v) >= dist.min_separation
                   for u, v in positions):
                positions.append((x, y))
                break
        else:
            raise RuntimeError(
                f'no position for particle {len(positions) + 1} of {count} '
                f'after {dist.max_attempts} attempts')
    low, high = np.log(dist.a_p[0]), np.log(dist.a_p[1])
    a_p = np.exp(rng.uniform(low, high, size=count))
    nearest = np.maximum(dist.z_p[0],
                         dist.clearance * a_p / dist.magnification)
    if (nearest >= dist.z_p[1]).any():
        raise ValueError('the range of z_p is too small for the radii: '
                         f'{dist.clearance} * a_p / magnification reaches '
                         f'{float(nearest.max()):.0f} pixels')
    return pd.DataFrame(dict(
        x_p=[p[0] for p in positions],
        y_p=[p[1] for p in positions],
        z_p=nearest + rng.random(size=count) * (dist.z_p[1] - nearest),
        a_p=a_p,
        n_p=dist.n_m + rng.uniform(dist.n_p[0] - dist.n_m,
                                   dist.n_p[1] - dist.n_m, size=count)))


def _toward_edge(value: float, size: int, band: float,
                 rng: np.random.Generator) -> float:
    '''A coordinate within ``band`` of the nearer end of ``[0, size]``

    The band is at most half of the size.
    '''
    depth = rng.uniform(0., min(band, size / 2.))
    return depth if value < size / 2. else size - depth


class Renderer:
    '''Computes the holograms of frames of one shape

    Parameters
    ----------
    shape : tuple of int
        ``(rows, columns)`` of the frames.
    instrument : dict
        ``wavelength``, ``magnification`` and ``n_m``.
        Default: ``CATCH.customization.evaluation.cases.INSTRUMENT``
    '''

    def __init__(self, shape: tuple[int, int],
                 instrument: dict[str, float] | None = None) -> None:
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.model = LorenzMie()
        self.model.instrument.properties = dict(instrument or INSTRUMENT)
        self.model.coordinates = self.model.meshgrid(self.shape)

    def __call__(self, particles: pd.DataFrame, noise: float,
                 rng: np.random.Generator) -> np.ndarray:
        '''The normalized hologram of the particles, with noise'''
        keys = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p']
        if len(particles):
            spheres: list[Particle] = [
                Sphere(**dict(zip(keys, map(float, row), strict=True)))
                for row in np.asarray(particles[keys], dtype=float)]
            self.model.particle = spheres
            field_ = np.asarray(self.model.hologram()).reshape(self.shape)
        else:
            field_ = np.ones(self.shape)
        image = field_ + noise * rng.standard_normal(self.shape,
                                                      dtype=np.float32)
        return image.astype(np.float32)


def make_frame(rng: np.random.Generator, dist: Distribution,
               renderer: Renderer, measure: Extent | None = None) -> Frame:
    '''A frame of random particles

    Raises
    ------
    FloatingPointError
        If the hologram is not finite.
    '''
    measure = measure or Extent()
    particles = sample_particles(rng, dist)
    noise = float(rng.uniform(*dist.noise))
    image = renderer(particles, noise, rng)
    if not np.isfinite(image).all():
        raise FloatingPointError(
            'the hologram is not finite for the particles\n'
            f'{particles.to_string()}')
    particles = describe(particles, dist.shape)
    particles['extent'] = [measure(p.z_p, p.a_p, p.n_p)
                           for _, p in particles.iterrows()]
    return Frame(image, particles, noise)


#: seeds of the evaluation sets, which training data never uses
FORBIDDEN_SEEDS = (2026,)
