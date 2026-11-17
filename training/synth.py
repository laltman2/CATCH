'''Synthetic training data, made on the fly

Frames of particles at random positions, including at the edges and the
corners of the image and with their holograms overlapping, and the crops of
them that the Estimator is given, made the way that ``CATCHapp`` makes them
(twice the extent of the feature, a center that is off by what a Localizer
gets wrong, the part that is outside the image padded or shifted inside).
Nothing is written to disk: a ``Stream`` draws frames from a random number
generator, so that the data are never repeated and the seed names them.

The detector (a CenterNet) trains on the frames and their ``particles``; the
Estimator trains on the crops.  The holograms are computed by whichever
backend ``PYLORENZMIE_BACKEND`` names (the ``torch`` one is the fast one on a
GPU, and in single precision).

The random numbers of the evaluation sets of ``CATCH.evaluation.sets`` are
made from their own seeds; ``Stream`` refuses those seeds, so the training
data cannot be the data that versions are compared on.
'''
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from pylorenzmie.theory import LorenzMie, Particle, Sphere
from torchvision import transforms

from CATCH.evaluation.cases import INSTRUMENT
from CATCH.evaluation.pipelines import crop_around, pad_around
from CATCH.evaluation.scenes import describe

#: the shape of the images that the Estimator takes
SHAPE = (201, 201)


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
class CropPolicy:
    '''How the crops are made from a frame

    Attributes
    ----------
    mode : str
        ``'pad'``: centered on the center that was found, background where
        the crop leaves the image.  ``'shift'``: the crop is moved inside the
        image, as ``CATCHapp.crop`` does.  Default: ``'pad'``
    jitter : float
        Standard deviation of the error of the center [pixels] (a Localizer
        finds the center within 1-3 pixels).  Default: 3
    tail : float
        Probability of a larger error, of the size ``tail_scale`` (the
        Localizer's centers at corners).  Default: 0.1
    tail_scale : float
        Standard deviation of the larger error [pixels].  Default: 30
    side_noise : float
        Relative error of the side of the crop, which is twice the extent of
        the feature (the Localizer does not know it).  Default: 0.05
    minimum : int
        Smallest side of a crop [pixels].  Default: 31
    repeats : int
        Crops of each particle of a frame, with independent errors of the
        center and of the side.  The frame is the expensive part of a crop
        (it takes as long as 4 to 5 crops do), so several crops of it train
        at less cost; the batches are shuffled.  Default: 1
    '''
    mode: Literal['pad', 'shift'] = 'pad'
    jitter: float = 3.
    tail: float = 0.1
    tail_scale: float = 30.
    side_noise: float = 0.05
    minimum: int = 31
    repeats: int = 1


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


@dataclass(frozen=True)
class Samples:
    '''Crops for the Estimator

    Attributes
    ----------
    images : torch.Tensor
        Shape ``(N, 1, 201, 201)``, as ``Estimator`` gives them to the net.
    scale : torch.Tensor
        Shape ``(N, 1)``: the side of the crop over 201.
    targets : torch.Tensor
        Shape ``(N, 3)``: ``z_p`` [pixels], ``a_p`` [um] and ``n_p``.
    meta : pandas.DataFrame
        What is known about each crop: ``side``, ``offset`` (distance of
        the particle from the center of the crop [pixels]), ``edge``,
        ``neighbor``.
    '''
    images: torch.Tensor
    scale: torch.Tensor
    targets: torch.Tensor
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)


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
        Default: ``CATCH.evaluation.cases.INSTRUMENT``
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


class Extent:
    '''Radius of the feature of a particle [pixels], with one model

    The same number as ``evaluation.pipelines.extent`` (the 20th zero
    crossing of the hologram along a row of 300 pixels), computed with a
    calculator that is made once.
    '''

    def __init__(self, instrument: dict[str, float] | None = None,
                 nfringes: int = 20, maxrange: int = 300) -> None:
        try:
            from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie
            model = numbaLorenzMie()
        except ImportError:  # pragma: no cover
            model = LorenzMie()
        model.instrument.properties = dict(instrument or INSTRUMENT)
        model.coordinates = model.meshgrid((1, maxrange))
        self.model = model
        self.nfringes = nfringes
        self.maxrange = maxrange

    def __call__(self, z_p: float, a_p: float, n_p: float) -> float:
        self.model.particle = Sphere(a_p=a_p, n_p=n_p, z_p=z_p)
        crossings = np.flatnonzero(
            np.diff(np.sign(np.asarray(self.model.hologram()) - 1.))) + 1
        if len(crossings) <= self.nfringes:
            return float(self.maxrange)
        return float(crossings[self.nfringes])


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


def prepare(crop: np.ndarray, shape: tuple[int, int] = SHAPE
            ) -> torch.Tensor:
    '''A crop in the form that the network takes: ``(1, 201, 201)``

    The same steps as ``Estimator._load_image`` (a test checks that they
    agree): scaled to a mean of 100, clipped, 8 bits, and resized.
    '''
    image = np.asarray(crop, dtype=float)
    if np.mean(image) != 100:
        image = np.clip(image / np.mean(image) * 100., 0, 255)
    tensor = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize(shape)])(
        image.astype(np.uint8))
    return cast(torch.Tensor, tensor)


def make_crops(frame: Frame, rng: np.random.Generator,
               policy: CropPolicy | None = None,
               pixel_scale: float = 1.) -> Samples:
    '''The crops of the particles inside a frame, as the Estimator gets them

    Parameters
    ----------
    frame : Frame
    rng : numpy.random.Generator
    policy : CropPolicy, optional
        Default: ``CropPolicy()``
    pixel_scale : float
        ``pixel_ratio`` of the instrument against the one of the network
        (1 for the reference instrument).  Default: 1
    '''
    policy = policy or CropPolicy()
    images, scales, targets, meta = [], [], [], []
    inside = frame.particles[frame.particles.inside]
    for p in (row for _, row in inside.iterrows()
              for _ in range(policy.repeats)):
        side = int(2. * p.extent * (1. + policy.side_noise
                                    * rng.normal()))
        side = max(side, policy.minimum)
        sigma = policy.tail_scale if rng.random() < policy.tail \
            else policy.jitter
        cx = p.x_p + sigma * rng.normal()
        cy = p.y_p + sigma * rng.normal()
        if policy.mode == 'pad':
            crop = pad_around(frame.image, cx, cy, side)
            x0 = int(round(cx - side / 2))
            y0 = int(round(cy - side / 2))
        else:
            side = min(side, *frame.image.shape)
            crop, x0, y0 = crop_around(frame.image, cx, cy, side)
        images.append(prepare(100. * crop))
        scales.append(crop.shape[0] * pixel_scale / SHAPE[0])
        targets.append([p.z_p, p.a_p, p.n_p])
        meta.append(dict(side=crop.shape[0],
                         offset=float(np.hypot(p.x_p - (x0 + side / 2.),
                                               p.y_p - (y0 + side / 2.))),
                         edge=p.edge, neighbor=p.neighbor))
    if not images:
        return Samples(torch.empty(0, 1, *SHAPE), torch.empty(0, 1),
                       torch.empty(0, 3), pd.DataFrame(
                           columns=['side', 'offset', 'edge', 'neighbor']))
    return Samples(torch.stack(images),
                   torch.tensor(scales, dtype=torch.float32).unsqueeze(1),
                   torch.tensor(targets, dtype=torch.float32),
                   pd.DataFrame(meta))


#: seeds of the evaluation sets, which training data never uses
FORBIDDEN_SEEDS = (2026,)


class Stream:
    '''An endless supply of frames or of crops, from a seed

    Parameters
    ----------
    seed : int
        Seed of the random numbers; the same seed gives the same data.  The
        seeds of the evaluation sets are refused.
    dist : Distribution
        The frames.  Default: ``Distribution()``
    policy : CropPolicy
        How the crops are made.  Default: ``CropPolicy()``
    '''

    def __init__(self, seed: int, dist: Distribution | None = None,
                 policy: CropPolicy | None = None) -> None:
        from CATCH.evaluation.sets import SETS
        forbidden = {s.seed for s in SETS.values()} | set(FORBIDDEN_SEEDS)
        if seed in forbidden:
            raise ValueError(f'seed {seed} makes an evaluation set')
        self.seed = seed
        self.dist = dist or Distribution()
        self.policy = policy or CropPolicy()
        self.rng = np.random.default_rng(seed)
        self.renderer = Renderer(self.dist.shape)
        self.measure = Extent()

    def frames(self) -> Iterator[Frame]:
        '''Frames, forever'''
        while True:
            yield make_frame(self.rng, self.dist, self.renderer,
                             self.measure)

    def crops(self, batch_size: int) -> Iterator[Samples]:
        '''Batches of crops of ``batch_size``, forever

        The crops of a frame are of one scene, so the batch is a mixture of
        several frames and its order is shuffled.
        '''
        pending: list[Samples] = []
        count = 0
        for frame in self.frames():
            samples = make_crops(frame, self.rng, self.policy)
            if len(samples.targets) == 0:
                continue
            pending.append(samples)
            count += len(samples.targets)
            while count >= batch_size:
                batch = _join(pending)
                order = self.rng.permutation(len(batch.targets))
                yield _take(batch, order[:batch_size])
                rest = _take(batch, order[batch_size:])
                pending = [rest] if len(rest.targets) else []
                count = len(rest.targets)


def _join(parts: list[Samples]) -> Samples:
    return Samples(torch.cat([p.images for p in parts]),
                   torch.cat([p.scale for p in parts]),
                   torch.cat([p.targets for p in parts]),
                   pd.concat([p.meta for p in parts], ignore_index=True))


def _take(samples: Samples, index: np.ndarray) -> Samples:
    return Samples(samples.images[index], samples.scale[index],
                   samples.targets[index],
                   samples.meta.iloc[index].reset_index(drop=True))
