'''Simulated scenes with several particles, and the truth about them'''
import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pylorenzmie.theory import LorenzMie, Particle, Sphere

from CATCH.customization.evaluation.cases import (
    INDEX, INSTRUMENT, RADII, Z_RANGE)
from CATCH.lib.instruments import pixel_ratio
from CATCH.lib.profiles import CAMERA_LEVEL

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
    index : int
        Number of the scene in its set.  Default: 0
    instrument : dict
        ``wavelength`` and ``magnification`` [um] and ``n_m`` of the
        microscope that the scene simulates.  Default: the reference
        instrument of the networks.
    saturation : float | None
        The brightest pixel that the camera records, if the scene was
        recorded by ``record``.  Default: None, the image is not clipped.
    '''
    image: np.ndarray
    truth: pd.DataFrame
    noise: float
    index: int = 0
    instrument: Mapping[str, float] = field(
        default_factory=lambda: dict(INSTRUMENT))
    saturation: float | None = None


def record(scene: Scene, level: float = CAMERA_LEVEL) -> Scene:
    '''The scene as a camera of limited range records it

    The simulated hologram of a sphere that focuses the light has spots many
    times brighter than the background, which no camera records: the pixels
    saturate.

    Parameters
    ----------
    scene : Scene
        The scene, with an image that is not clipped.
    level : float
        The brightest pixel, in units of the background.  Default:
        ``CAMERA_LEVEL``, the 255 of an 8-bit image with a background of 100.

    Returns
    -------
    scene : Scene
        The same truth, with the image clipped to the range 0 to ``level``.
    '''
    return replace(scene, image=np.clip(scene.image, 0., level),
                   saturation=level)


def to_uint8(image: np.ndarray) -> np.ndarray:
    '''The image as a camera records it: 8 bits, with a mean of 100'''
    return np.clip(100. * image, 0, 255).astype(np.uint8)


def write_images(scenes: list[Scene], directory: str | Path) -> list[Path]:
    '''Writes the images of the scenes as 8-bit PNG files

    The files are ``scene_0000.png``, ``scene_0001.png`` and so on, by the
    ``index`` of the scene.  Any program can find the particles in them
    (see ``CATCH.customization.evaluation.detect_images``), and the detections
    can be scored with
    ``CATCH.customization.evaluation.pipelines.recorded_pipeline``.

    Returns
    -------
    paths : list of pathlib.Path
        The files that were written.
    '''
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for scene in scenes:
        path = directory / f'scene_{scene.index:04d}.png'
        Image.fromarray(to_uint8(scene.image)).save(path)
        paths.append(path)
    return paths


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
           rng: np.random.Generator,
           instrument: Mapping[str, float] | None = None) -> np.ndarray:
    '''The normalized hologram of the particles, with noise

    The fields of the particles add before the intensity is taken, so that
    the fringes of one particle interfere with those of its neighbors.

    Parameters
    ----------
    particles : pandas.DataFrame
        The columns ``x_p``, ``y_p``, ``z_p`` [pixels of the instrument],
        ``a_p`` [um] and ``n_p``.
    shape : tuple of int
        ``(rows, columns)`` of the image [pixels].
    noise : float
        Standard deviation of the noise.
    rng : numpy.random.Generator
        Source of the noise.
    instrument : mapping | None
        ``wavelength`` and ``magnification`` [um] and ``n_m``.  Default:
        None, the reference instrument.

    Returns
    -------
    image : numpy.ndarray
        The normalized hologram.
    '''
    model = LorenzMie()
    model.instrument.properties = dict(instrument or INSTRUMENT)
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
                z_range: tuple[float, float] | None = None,
                margin: float = 0.,
                min_separation: float = 20.,
                max_attempts: int = 1000,
                instrument: Mapping[str, float] | None = None
                ) -> list[Scene]:
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
        Default: ``CATCH.customization.evaluation.cases.RADII``
    indexes : dict
        The refractive index of a particle is one of the values.
        Default: ``CATCH.customization.evaluation.cases.INDEX``
    z_range : tuple of float | None
        Range of the axial position [pixels of the instrument].  Default:
        None, which is ``CATCH.customization.evaluation.cases.Z_RANGE`` of
        the reference instrument converted to the pixels of ``instrument``
        (the same heights in wavelengths of the medium).
    margin : float
        The center of a particle can be this far outside the image, so that
        its feature is truncated by the edge [pixels].  Default: 0
    min_separation : float
        Smallest lateral distance between the centers of two particles
        [pixels].  Default: 20
    max_attempts : int
        Number of positions to try for each particle.  Default: 1000
    instrument : mapping | None
        ``wavelength`` and ``magnification`` [um] and ``n_m`` of the
        microscope to simulate.  The positions, the radii and the indexes
        of the particles come from ``seed`` and do not depend on it, so a
        scene is the same particles as another instrument sees them.
        Default: None, the reference instrument.

    Raises
    ------
    RuntimeError
        If no position is far enough from the other particles.
    '''
    indexes = INDEX if indexes is None else indexes
    index_values = tuple(indexes.values())
    instrument = dict(instrument or INSTRUMENT)
    if z_range is None:
        ratio = pixel_ratio(instrument, INSTRUMENT)
        z_range = (Z_RANGE[0] / ratio, Z_RANGE[1] / ratio)
    rows, columns = shape
    rng = np.random.default_rng(seed)
    scenes = []
    for index in range(nscenes):
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
        image = render(particles, shape, noise, rng, instrument)
        scenes.append(Scene(image, describe(particles, shape), noise,
                            index, instrument))
    return scenes


def add_instrument_option(parser: argparse.ArgumentParser) -> None:
    '''Add ``--instrument`` to a command line

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.  The option is read with :func:`instrument_from_option`.
    '''
    parser.add_argument('--instrument', type=float, nargs=3, default=None,
                        metavar=('WAVELENGTH', 'MAGNIFICATION', 'N_M'),
                        help='the microscope to simulate: the wavelength '
                        'and the size of a pixel [um], and the index of the '
                        'medium.  Default: the reference instrument of the '
                        'networks (0.447 0.048 1.34)')


def instrument_from_option(values: Sequence[float] | None
                           ) -> dict[str, float] | None:
    '''The instrument that ``--instrument`` gave

    Parameters
    ----------
    values : sequence of float | None
        ``args.instrument``.

    Returns
    -------
    instrument : dict | None
        ``wavelength``, ``magnification`` and ``n_m``, or None for the
        reference instrument.
    '''
    if values is None:
        return None
    return dict(zip(('wavelength', 'magnification', 'n_m'),
                    (float(v) for v in values), strict=True))
