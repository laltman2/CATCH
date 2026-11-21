'''Radial profiles of the holograms of all the particles of a frame

The profile of a particle is the median of the hologram on rings around
its center, at the radii :data:`CATCH.lib.networks.RADII` of the reference
instrument.  Each ring is sampled at :data:`ANGLES` directions, taking the
nearest pixel of the camera (no interpolation, so the noise of the camera
is not smoothed); the median ignores the samples outside the frame and is
robust against the fringes of neighbors.  All the particles of a frame are
done at once, on a GPU if there is one.
'''
from __future__ import annotations

import warnings

import numpy as np
import torch
from numpy.typing import ArrayLike

from CATCH.lib.networks import RADII

#: directions sampled on each ring
ANGLES = 2048
#: fewest samples inside the frame for a ring to count
MINCOUNT = 8


def _sample(flat: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
            dx: torch.Tensor, dy: torch.Tensor, height: int,
            width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.round(x[:, None, None] + dx)
    iy = torch.round(y[:, None, None] + dy)
    ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    lin = (iy.clamp(0, height - 1) * width
           + ix.clamp(0, width - 1)).to(torch.int64)
    v = torch.where(ok, flat[lin], torch.full_like(dx, float('nan')))
    return v, ok.to(dx.dtype).mean(2)


class Profiler:
    '''Measures the radial profiles of particles in a frame

    Parameters
    ----------
    device : torch.device or str
        Where to compute.  Default: ``'cpu'``
    chunk : int
        Particles per pass (one shape for the compiled sampler).
        Default: 16
    compile : bool
        Fuse the sampling into one kernel with ``torch.compile`` (three
        times faster on a GPU); if compiling fails, the sampling runs
        uncompiled, with a warning.  Default: True
    '''

    def __init__(self, device: torch.device | str = 'cpu', chunk: int = 16,
                 compile: bool = True) -> None:
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.sample = _sample
        if compile:
            try:
                self.sample = torch.compile(_sample, dynamic=False)
            except Exception as error:  # pragma: no cover
                warnings.warn(f'torch.compile failed ({error}); the '
                              'profiles are measured uncompiled',
                              stacklevel=2)
        self._offsets: dict[float, tuple[torch.Tensor, ...]] = {}

    def offsets(self, scale: float) -> tuple[torch.Tensor, ...]:
        '''Polar offsets [camera pixels] of the rings, and circumferences'''
        if scale not in self._offsets:
            r = torch.as_tensor(RADII / scale, device=self.device,
                                dtype=torch.float32)
            theta = ((torch.arange(ANGLES, device=self.device,
                                   dtype=torch.float32) + .5)
                     * (2 * np.pi / ANGLES))
            circumference = (2 * np.pi * r).clamp(max=ANGLES)
            self._offsets[scale] = (r[:, None] * torch.cos(theta),
                                    r[:, None] * torch.sin(theta),
                                    circumference)
        return self._offsets[scale]

    def __call__(self, image: torch.Tensor, x: ArrayLike,
                 y: ArrayLike, scale: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Profiles of the particles centered at (x, y)

        Parameters
        ----------
        image : torch.Tensor, shape (H, W)
            Normalized hologram, on the device.
        x, y : sequence of float
            Centers [camera pixels].
        scale : float
            Size of a pixel of the camera in pixels of the reference
            (:func:`CATCH.lib.instruments.pixel_ratio`).

        Returns
        -------
        b, m, c : torch.Tensor, shape (P, NRADII)
            The profile (1 where it is not measured), the mask of the
            measured radii and the number of pixels on each ring.
        '''
        height, width = image.shape
        flat = image.reshape(-1).to(torch.float32)
        dx, dy, circumference = self.offsets(float(scale))
        xa = np.asarray(x, np.float32)
        ya = np.asarray(y, np.float32)
        B, M, C = [], [], []
        for s in range(0, len(xa), self.chunk):
            xs, ys = xa[s:s + self.chunk], ya[s:s + self.chunk]
            n = len(xs)
            if self.sample is not _sample and n < self.chunk:
                xs = np.r_[xs, np.repeat(xs[:1], self.chunk - n)]
                ys = np.r_[ys, np.repeat(ys[:1], self.chunk - n)]
            tx = torch.as_tensor(xs, device=self.device)
            ty = torch.as_tensor(ys, device=self.device)
            v, inside = self.sample(flat, tx, ty, dx, dy, height, width)
            b = torch.nanmedian(v, dim=2).values[:n]
            c = (inside * circumference)[:n]
            good = c >= MINCOUNT
            B.append(torch.where(good, b, torch.ones_like(b)))
            M.append(good.to(torch.float32))
            C.append(c * good)
        if not B:
            empty = torch.zeros((0, len(RADII)), device=self.device)
            return empty, empty, empty
        return torch.cat(B), torch.cat(M), torch.cat(C)
