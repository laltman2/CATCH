'''Simulated scenes with several particles, and the truth about them'''
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pylorenzmie.theory import LorenzMie, Particle, Sphere

from CATCH.evaluation.cases import INDEX, RADII, Z_RANGE

#: shape of the image of the camera, in pixels: (rows, columns)
SHAPE = (1024, 1280)


@dataclass(frozen=True, eq=False)
class Scene:
    '''A simulated image and the particles in it

    Attributes
    ----------
    image : numpy.ndarray
        The normalized hologram, with noise: the mean is about 1.
    truth : pandas.DataFrame
        One row for each particle, with the position ``x_p``, ``y_p`` and
        ``z_p`` [pixels], the radius ``a_p`` [um] and the index ``n_p``, and
        ``nparticles`` (in the scene), ``edge`` (distance [pixels] from the
        center to the nearest edge of the image, negative outside),
        ``neighbor`` (lateral distance [pixels] to the nearest other
        particle, infinite if there is none) and ``inside`` (whether the
        center is in the image).
    noise : float
        Standard deviation of the noise in the image.
    '''
    image: np.ndarray
    truth: pd.DataFrame
    noise: float


def describe(particles: pd.DataFrame,
             shape: tuple[int, int]) -> pd.DataFrame:
    '''Adds the columns that locate each particle in the scene

    Parameters
    ----------
    particles : pandas.DataFrame
        The columns ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``.
    shape : tuple of int
        ``(rows, columns)`` of the image.
    '''
    rows, columns = shape
    out = particles.reset_index(drop=True).copy()
    x, y = out.x_p.to_numpy(), out.y_p.to_numpy()
    out['nparticles'] = len(out)
    out['edge'] = np.minimum.reduce([x, columns - 1 - x, y, rows - 1 - y])
    separation = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(separation, np.inf)
    out['neighbor'] = separation.min(axis=1) if len(out) > 1 else np.inf
    out['inside'] = out.edge >= 0
    return out


def render(particles: pd.DataFrame, shape: tuple[int, int], noise: float,
           rng: np.random.Generator) -> np.ndarray:
    '''The normalized hologram of the particles, with noise

    The fields of the particles add before the intensity is taken, so that
    the fringes of one particle interfere with those of its neighbors.
    '''
    model = LorenzMie()
    model.instrument.noise = noise
    model.coordinates = model.meshgrid(shape)
    keys = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p']
    spheres: list[Particle] = [
        Sphere(**dict(zip(keys, map(float, row), strict=True)))
        for row in particles[keys].to_numpy(dtype=float)]
    model.particle = spheres
    return model.hologram().reshape(shape) + noise * rng.normal(size=shape)


def make_scenes(nscenes: int = 20,
                nparticles: tuple[int, int] = (1, 5),
                shape: tuple[int, int] = SHAPE,
                noise: float = 0.05,
                seed: int = 2026,
                radii: tuple[float, ...] = RADII,
                indexes: dict[str, float] | None = None,
                z_range: tuple[float, float] = Z_RANGE,
                margin: float = 0.,
                min_separation: float = 20.,
                max_attempts: int = 1000) -> list[Scene]:
    '''Returns scenes of particles at random positions

    Parameters
    ----------
    nscenes : int
        Number of scenes.  Default: 20
    nparticles : tuple of int
        The number of particles in a scene is random, from the first number
        to the second, inclusive.  Default: (1, 5)
    shape : tuple of int
        ``(rows, columns)`` of the image [pixels].  Default: ``SHAPE``
    noise : float
        Standard deviation of the noise.  Default: 0.05
    seed : int
        Seed of the random numbers.  Default: 2026
    radii : tuple of float
        The radius of a particle is one of these [um].
        Default: ``CATCH.evaluation.cases.RADII``
    indexes : dict
        The refractive index of a particle is one of the values.
        Default: ``CATCH.evaluation.cases.INDEX``
    z_range : tuple of float
        Range of the axial position [pixels].
        Default: ``CATCH.evaluation.cases.Z_RANGE``
    margin : float
        The center of a particle can be this far outside the image, so that
        its feature is truncated by the edge [pixels].  Default: 0
    min_separation : float
        Smallest lateral distance between the centers of two particles
        [pixels].  Default: 20
    max_attempts : int
        Number of positions to try for each particle.  Default: 1000

    Raises
    ------
    RuntimeError
        If no position is far enough from the other particles.
    '''
    indexes = INDEX if indexes is None else indexes
    index_values = tuple(indexes.values())
    rows, columns = shape
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(nscenes):
        count = int(rng.integers(nparticles[0], nparticles[1] + 1))
        positions: list[tuple[float, float]] = []
        for _ in range(count):
            for _ in range(max_attempts):
                x = float(rng.uniform(-margin, columns - 1 + margin))
                y = float(rng.uniform(-margin, rows - 1 + margin))
                if all(np.hypot(x - u, y - v) >= min_separation
                       for u, v in positions):
                    positions.append((x, y))
                    break
            else:
                raise RuntimeError(
                    f'no position for particle {len(positions) + 1} of '
                    f'{count} after {max_attempts} attempts: the image is '
                    f'too small for a separation of {min_separation} pixels')
        particles = pd.DataFrame(
            dict(x_p=[p[0] for p in positions],
                 y_p=[p[1] for p in positions],
                 z_p=[float(rng.uniform(*z_range)) for _ in positions],
                 a_p=[float(rng.choice(radii)) for _ in positions],
                 n_p=[float(rng.choice(index_values)) for _ in positions]))
        image = render(particles, shape, noise, rng)
        scenes.append(Scene(image, describe(particles, shape), noise))
    return scenes
