'''Synthetic frames for the networks of CATCH 3

A frame is a normalized hologram of 1 to ``nparticles`` spheres, rendered
with the pixels of an instrument whose pixel is ``scale`` reference pixels
(drawn log-uniformly from :attr:`Ranges.scales` for every frame), so that
one set of weights is meant to serve a range of instruments.  The heights are
log-uniform in micrometers, half of the index contrasts are log-uniform
(more faint particles), and the normalization is made imperfect the way
real backgrounds are: a black level that lowers the contrast, a smooth
gain across the field, and faint straight fringes.

Rendering needs pylorenzmie (``PYLORENZMIE_BACKEND=torch`` is the fast
backend on a GPU).
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from CATCH.lib.instruments import REFERENCE
from CATCH.customization.training.synth import (
    FORBIDDEN_SEEDS, Distribution, Renderer)


def log_uniform(rng: np.random.Generator, bounds: tuple[float, float],
                size: int | None = None) -> Any:
    '''Random numbers between ``bounds``, uniform in the logarithm

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the numbers.
    bounds : tuple of float
        Smallest and largest value.
    size : int | None
        Number of values.  Default: None, one value.

    Returns
    -------
    values : float or numpy.ndarray
    '''
    return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size))


@dataclass(frozen=True)
class Ranges:
    '''What the frames contain

    Attributes
    ----------
    shape : tuple of int
        ``(rows, columns)`` [pixels].  Default: (1024, 1280)
    scales : tuple of float
        Range of the pixel of the camera in reference pixels.
        Default: (1, 2.5)
    nparticles : tuple of int
        Fewest and most particles in a frame.  Default: (0, 10)
    a_p : tuple of float
        Radii [um], log-uniform.  Default: (0.2, 5.5)
    z_um : tuple of float
        Heights [um], log-uniform; at least 1.2 radii.  Default: (5, 330)
    contrast : tuple of float
        ``n_p - n_m``: half log-uniform, half uniform.
        Default: (0.04, 1.16)
    noise : tuple of float
        Standard deviation of the noise.  Default: (0.015, 0.07)
    imperfect : bool
        Imperfect normalization.  Default: True
    '''
    shape: tuple[int, int] = (1024, 1280)
    scales: tuple[float, float] = (1., 2.5)
    nparticles: tuple[int, int] = (0, 10)
    a_p: tuple[float, float] = (0.2, 5.5)
    z_um: tuple[float, float] = (5., 330.)
    contrast: tuple[float, float] = (0.04, 1.16)
    noise: tuple[float, float] = (0.015, 0.07)
    imperfect: bool = True


class Frames:
    '''Draws frames from a seed

    Parameters
    ----------
    seed : int
        Seed of the random numbers; the seeds of the evaluation sets are
        refused.
    ranges : Ranges
        What the frames contain.  Default: ``Ranges()``

    Raises
    ------
    ValueError
        If the seed is one of the evaluation sets.
    '''

    def __init__(self, seed: int, ranges: Ranges | None = None) -> None:
        if seed in FORBIDDEN_SEEDS:
            raise ValueError(f'seed {seed} makes an evaluation set')
        ranges = ranges or Ranges()
        self.ranges = ranges
        self.rng = np.random.default_rng(seed)
        self.dist = Distribution(shape=ranges.shape,
                                 nparticles=ranges.nparticles,
                                 a_p=ranges.a_p, z_p=(1., 1e9))
        self.render = Renderer(ranges.shape, dict(REFERENCE))
        rows, cols = ranges.shape
        self.yy, self.xx = np.mgrid[0:rows, 0:cols].astype(np.float32)

    def scale(self) -> float:
        '''A scale, log-uniform in the range'''
        return float(log_uniform(self.rng, self.ranges.scales))

    def particles(self, scale: float) -> pd.DataFrame:
        '''Random particles; ``z_p`` in pixels of the frame'''
        from CATCH.customization.training.synth import sample_particles
        rng, r = self.rng, self.ranges
        p = sample_particles(rng, self.dist)
        n = len(p)
        z = log_uniform(rng, r.z_um, n)
        p['z_p'] = np.maximum(z, 1.2 * p.a_p) / (
            REFERENCE['magnification'] * scale)
        c = np.where(rng.random(n) < .5,
                     log_uniform(rng, r.contrast, n),
                     rng.uniform(*r.contrast, n))
        p['n_p'] = REFERENCE['n_m'] + c
        return p

    def imperfect(self, image: np.ndarray) -> np.ndarray:
        '''A normalization that is not quite right'''
        rng = self.rng
        rows, cols = self.ranges.shape
        d = image - 1.
        if rng.random() < .5:                 # a black level
            d = d * rng.uniform(.7, 1.)
        gain = np.ones(self.ranges.shape, np.float32)
        if rng.random() < .7:                 # a smooth gain
            gain += rng.uniform(-.05, .05) * (self.xx / cols - .5)
            gain += rng.uniform(-.05, .05) * (self.yy / rows - .5)
            for _ in range(int(rng.integers(0, 4))):
                x0, y0 = rng.uniform(0, cols), rng.uniform(0, rows)
                w = rng.uniform(100, 600)
                gain += rng.uniform(-.06, .06) * np.exp(
                    -((self.xx - x0) ** 2 + (self.yy - y0) ** 2)
                    / (2 * w * w))
        out = gain * (1. + d)
        if rng.random() < .3:                 # faint straight fringes
            k = 2 * np.pi / rng.uniform(15, 200)
            t = rng.uniform(0, np.pi)
            out = out + rng.uniform(.003, .02) * np.cos(
                k * (np.cos(t) * self.xx + np.sin(t) * self.yy)
                + rng.uniform(0, 2 * np.pi))
        return out.astype(np.float32)

    def frame(self, scale: float | None = None
              ) -> tuple[np.ndarray, pd.DataFrame, float]:
        '''An image, its particles and its scale

        The particles are those whose centers are inside the image, with
        ``z_p`` in pixels of the frame and the column ``scale``.
        '''
        scale = float(scale) if scale else self.scale()
        self.render.model.instrument.magnification = (
            REFERENCE['magnification'] * scale)
        rows, cols = self.ranges.shape
        while True:
            p = self.particles(scale)
            noise = float(self.rng.uniform(*self.ranges.noise))
            image = self.render(p, noise, self.rng)
            if np.isfinite(image).all():
                break
        if self.ranges.imperfect:
            image = self.imperfect(image)
        inside = p.x_p.between(0, cols - 1) & p.y_p.between(0, rows - 1)
        p = pd.DataFrame(p[inside]).reset_index(drop=True)
        p['scale'] = scale
        return image, p, scale
