'''Estimates the properties of particles from the radial profiles of their
holograms

For every particle of a frame, the Estimator measures the radial profile
of the hologram (:class:`CATCH.lib.profiles.Profiler`), and the
:class:`CATCH.lib.networks.ProfileNet` proposes 64 candidates for ``z_p``,
``a_p`` and ``n_p``.  Each candidate is checked against the profile with
the Lorenz-Mie theory (:class:`CATCH.lib.lorenzmie.RadialModel`), by a robust
misfit, and the best ones become the starts of a fit:

* the best :attr:`Estimator.top` candidates, and
* the best :attr:`Estimator.preferred` candidates with ``n_p`` in a
  preferred range: the standard range (``n_p`` up to 1.8) or a prior that
  the user gives for the sample.  If no candidate falls in it, a scan of
  ``a_p`` and ``n_p`` at the ``z_p`` of the best candidate provides them.

Large spheres far from the focal plane have holograms that are nearly the
same for different sizes and indexes (the size-index degeneracy), so the
best candidate is often not the right one, and the reported start is the
best by the misfit of the profile: a fit (``CATCHapp.analyze(fit=True)``)
decides among the starts.

The networks work in pixels of a reference instrument; the results are
converted to the instrument of the camera (:mod:`CATCH.lib.instruments`).
'''
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

from CATCH.lib.instruments import REFERENCE, from_reference, pixel_ratio
from CATCH.Localizer import default_device, model_directory, normalize
from CATCH.lib.lorenzmie import RadialModel
from CATCH.lib.networks import RADII, ProfileNet, profile_features
from CATCH.lib.profiles import Profiler

#: columns of the output
COLUMNS = ('z_p', 'a_p', 'n_p', 'misfit', 'snr', 'starts')

#: the largest candidates that are checked against the profile: the height
#: [reference pixels], the radius [um] and the index.  They are the bounds
#: of the fits (``CATCH.lib.fitter``), far beyond what the networks are
#: trained for.  A network that sees an input unlike its training, such as a
#: focal spot much brighter than a camera records, can propose absurd values,
#: and the Lorenz-Mie model of a sphere of millions of wavelengths never
#: finishes.
MAXIMUM = dict(z=20000., a=10., n=3.)

Instrument = Mapping[str, float]


def ring_radius(z_p: float, scale: float, rings: int = 10,
                reference: Instrument = REFERENCE) -> float:
    '''Radius [camera pixels] of the ``rings``-th bright ring of a hologram

    ``z_p`` in pixels of the reference instrument: the fringe of order
    ``m`` is where the path from the particle is ``m`` wavelengths longer
    than ``z_p``.

    Parameters
    ----------
    z_p : float
        Height of the particle [pixels of the reference instrument].
    scale : float
        Size of a camera pixel in reference pixels
        (:func:`CATCH.lib.instruments.pixel_ratio`).
    rings : int
        Order of the bright ring.  Default: 10
    reference : mapping
        The instrument of the networks.  Default: ``REFERENCE``

    Returns
    -------
    radius : float
        Radius of the ring [pixels of the camera].
    '''
    wavelength = (reference['wavelength'] / reference['n_m']
                  / reference['magnification'])        # reference pixels
    return float(np.sqrt((z_p + rings * wavelength) ** 2 - z_p ** 2)) / scale


class Estimator:
    '''Proposes the starts of fits for the particles of a frame

    Parameters
    ----------
    model : str | pathlib.Path | None
        Name of a trained model of the package (``models/estimator/<model>``)
        or the path of a directory with ``weights.pt`` and
        ``configuration.json``.  Default: None, which is ``'default'``.
    device : torch.device | str | None
        Where to compute.  Default: None, the fastest available.
    compile : bool
        Compile the inner loops with ``torch.compile``.  Default: True

    Attributes
    ----------
    top : int
        Best candidates kept as starts.
    preferred : int
        Best candidates in the preferred range of ``n_p`` added to them.
    standard : tuple of float
        The preferred range of ``n_p`` when no prior is given.
    scales : tuple of float
        Range of the scale (pixel of the camera in reference pixels) that
        the network was trained on.
    '''

    def __init__(self, model: str | Path | None = None,
                 device: torch.device | str | None = None,
                 compile: bool = True) -> None:
        self.model = model or 'default'
        self.device = torch.device(device) if device else default_device()
        path = model_directory('estimator', self.model)
        config = json.loads((path / 'configuration.json').read_text())
        self.reference: dict[str, float] = dict(config['reference'])
        self.top = int(config['top'])
        self.preferred = int(config['preferred'])
        self.standard = tuple(config['standard'])
        self.scales = tuple(config['scales'])
        self.stride = int(config['stride'])
        self.noise = float(config['noise'])
        self.floor = float(config['floor'])
        self.clearance = float(config['clearance'])
        self.net = ProfileNet().to(self.device).eval()
        state = torch.load(path / 'weights.pt', map_location=self.device,
                           weights_only=True)
        self.net.load_state_dict(state)
        self.profiler = Profiler(self.device, compile=compile)
        self.radial = RadialModel(RADII[::self.stride],
                                  self.reference['wavelength'],
                                  self.reference['magnification'],
                                  self.reference['n_m'], self.device,
                                  compile=compile)

    def __call__(self, image: np.ndarray | torch.Tensor,
                 x: ArrayLike, y: ArrayLike,
                 instrument: Instrument | None = None,
                 nrange: tuple[float, float] | None = None,
                 explore: int = 0,
                 saturation: float | None = None,
                 noise: float | None = None,
                 rings: int = 10) -> pd.DataFrame:
        '''Starts of fits for the particles centered at (x, y)

        Equivalent to :meth:`estimate`, to which the arguments go.
        '''
        return self.estimate(image, x, y, instrument, nrange, explore,
                             saturation, noise, rings)

    def misfits(self, owner: np.ndarray, z: np.ndarray, a: np.ndarray,
                n: np.ndarray, b: torch.Tensor, m: torch.Tensor,
                c: torch.Tensor) -> np.ndarray:
        '''Robust misfit of candidates of the particles ``owner``

        The mean over the measured radii of ``log(1 + (residual /
        scale)**2)``, with the scale set by the noise of the median of the
        pixels on each ring.

        Parameters
        ----------
        owner : numpy.ndarray
            For each candidate, the index of its particle.
        z, a, n : numpy.ndarray
            Candidates: ``z_p`` [reference pixels], ``a_p`` [um], ``n_p``.
        b, m, c : torch.Tensor
            Profiles, masks and counts of the particles, as measured by
            :class:`CATCH.lib.profiles.Profiler`.

        Returns
        -------
        misfits : numpy.ndarray
            One value for each candidate; lower is better.
        '''
        if len(z) == 0:
            return np.zeros(0)
        h = self.radial(z, a, n).to(torch.float32)
        s = self.stride
        bs, ms, cs = b[:, ::s], m[:, ::s], c[:, ::s]
        scale = torch.clamp(np.sqrt(np.pi / 2) * self.noise
                            / torch.sqrt(cs.clamp(min=1)), min=self.floor)
        o = torch.as_tensor(owner, device=self.device)
        e = torch.log1p(((bs[o] - h) / scale[o]) ** 2) * ms[o]
        return (e.sum(1) / ms[o].sum(1).clamp(min=1)).cpu().numpy()

    def estimate(self, image: np.ndarray | torch.Tensor,
                 x: ArrayLike, y: ArrayLike,
                 instrument: Instrument | None = None,
                 nrange: tuple[float, float] | None = None,
                 explore: int = 0,
                 saturation: float | None = None,
                 noise: float | None = None,
                 rings: int = 10) -> pd.DataFrame:
        '''Starts of fits for the particles centered at (x, y)

        Parameters
        ----------
        image : numpy.ndarray | torch.Tensor
            Normalized hologram (background 1, or 100 as in CATCH 2).
        x, y : sequence of float
            Centers of the particles [pixels].
        instrument : mapping | None
            ``wavelength`` and ``magnification`` [um] and ``n_m``.
            Default: None, the reference instrument.
        nrange : tuple of float | None
            Preferred range of ``n_p`` (a prior for the sample).
            Default: None, the standard range.
        explore : int
            With a prior: starts of distinct ``a_p`` (3% apart) added
            from a scan inside the range, to sample a valley of
            near-degenerate solutions.  Default: 0
        saturation : float | None
            Pixels at or above this value (in units of the background)
            are not used to measure the profiles: a camera that saturates
            records the limit of its range, not the hologram.  Default:
            None, every pixel counts.
        noise : float | None
            Standard deviation of the noise of the image, in units of the
            background, for ``snr``.  Default: None, the noise of the
            configuration of the model (0.05).
        rings : int
            Order of the outermost bright ring that ``snr`` counts: the
            box that is analyzed.  Default: 10

        Returns
        -------
        estimates : pandas.DataFrame
            One row per particle: the best start ``z_p`` [pixels of the
            camera], ``a_p`` [um] and ``n_p``; its ``misfit``; ``snr``, the
            signal-to-noise ratio of the hologram (see
            :meth:`signal_to_noise`); and ``starts``, the list of all the
            starts as ``(z_p, a_p, n_p)``, best first.
        '''
        instrument = dict(instrument or self.reference)
        scale = pixel_ratio(instrument, self.reference)
        xs = np.asarray(x, float)
        ys = np.asarray(y, float)
        if len(xs) == 0:
            return pd.DataFrame(columns=list(COLUMNS))
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.ascontiguousarray(normalize(image)))
        image = image.to(self.device, torch.float32)
        if saturation is not None:
            image = torch.where(image >= saturation - 1e-6,
                                torch.full_like(image, float('nan')), image)
        n_m = self.reference['n_m']
        low, high = (nrange if nrange is not None else self.standard)
        if nrange is not None:     # from the sample's medium to the reference
            low, high = (low / instrument['n_m'] * n_m,
                         high / instrument['n_m'] * n_m)
        with torch.no_grad():
            b, m, c = self.profiler(image, xs, ys, scale)
            cand = self.net.candidates(profile_features(
                b, m, self.reference['magnification'])).cpu().numpy()
            P, K = cand.shape[:2]
            z, a, q = cand.reshape(-1, 3).T
            nn = n_m + q
            ok = (z * self.reference['magnification'] >= self.clearance * a)
            ok &= ((z <= MAXIMUM['z']) & (a <= MAXIMUM['a'])
                   & (nn <= MAXIMUM['n']))
            owner = np.repeat(np.arange(P), K)
            mis = np.full(P * K, np.inf)
            mis[ok] = self.misfits(owner[ok], z[ok], a[ok], nn[ok], b, m, c)
            mis = mis.reshape(P, K)
            starts, need = [], []
            for i in range(P):
                order = [j for j in np.argsort(mis[i])
                         if np.isfinite(mis[i, j])]
                ranked = [(float(z.reshape(P, K)[i, j]),
                           float(a.reshape(P, K)[i, j]),
                           float(nn.reshape(P, K)[i, j]),
                           float(mis[i, j])) for j in order]
                best = ranked[:self.top]
                inside = [r for r in ranked[self.top:]
                          if low <= r[2] <= high][:self.preferred]
                starts.append(best + inside)
                if best and (explore or (not inside and not any(
                        low <= r[2] <= high for r in best))):
                    need.append(i)
            if need:
                self._scan(need, starts, (low, high), explore, b, m, c)
        sigma = self.noise if noise is None else float(noise)
        snr = self.signal_to_noise(
            b, m, c, [s[0][0] if s else np.nan for s in starts], scale,
            sigma, rings)
        rows = []
        for s, ratio in zip(starts, snr, strict=True):
            if not s:
                rows.append(dict(z_p=np.nan, a_p=np.nan, n_p=np.nan,
                                 misfit=np.nan, snr=np.nan, starts=[]))
                continue
            conv = [from_reference(zz, aa, nn_, instrument, self.reference)
                    for zz, aa, nn_, _ in s]
            conv = [tuple(float(v) for v in q) for q in conv]
            rows.append(dict(z_p=conv[0][0], a_p=conv[0][1],
                             n_p=conv[0][2], misfit=s[0][3],
                             snr=float(ratio), starts=conv))
        return pd.DataFrame(rows, columns=list(COLUMNS))

    def signal_to_noise(self, b: torch.Tensor, m: torch.Tensor,
                        c: torch.Tensor, z: ArrayLike, scale: float,
                        noise: float, rings: int = 10) -> np.ndarray:
        '''Signal-to-noise ratio of the holograms, from their profiles

        The energy of the hologram, the sum over the pixels of its
        deviation from the background squared, in units of the variance of
        the noise, is measured on the rings of the profile that lie within
        the ``rings``-th bright ring and were measured (not outside the
        frame, not saturated).  The median of a ring of ``c`` pixels has the
        noise of variance ``pi / 2 * noise**2 / c``, which is subtracted.  A
        hologram that is lost in the noise has a ratio near 1 or less.

        Parameters
        ----------
        b, m, c : torch.Tensor, shape (P, NRADII)
            Profiles, masks and counts of the particles, as measured by
            :class:`CATCH.lib.profiles.Profiler`.
        z : array_like
            Axial position of each particle [reference pixels]; the box is
            as large as its ``rings``-th bright ring.  ``nan`` for a
            particle without an estimate.
        scale : float
            Size of a pixel of the camera in reference pixels.
        noise : float
            Standard deviation of the noise, in units of the background.
        rings : int
            Order of the outermost bright ring.  Default: 10

        Returns
        -------
        snr : numpy.ndarray
            The square root of the energy, for each particle; ``nan`` where
            ``z`` is.
        '''
        z = np.asarray(z, float)
        wavelength = (self.reference['wavelength'] / self.reference['n_m']
                      / self.reference['magnification'])    # pixels
        with np.errstate(invalid='ignore'):
            radius = np.sqrt((z + rings * wavelength) ** 2 - z ** 2)
        radii = np.asarray(RADII, float)
        step = float(radii[1] - radii[0]) / scale   # camera pixels per ring
        inside = torch.as_tensor(radii[None, :] <= radius[:, None],
                                 device=b.device) & (m > 0)
        pixels = c * step                           # in the annulus
        energy = pixels * (b - 1.) ** 2 / noise ** 2 - np.pi / 2 * step
        total = (energy * inside).sum(1).clamp(min=0).cpu().numpy()
        return np.where(np.isnan(z), np.nan, np.sqrt(total))

    def _scan(self, need: list[int], starts: list[list[tuple]],
              nrange: tuple[float, float], explore: int,
              b: torch.Tensor, m: torch.Tensor, c: torch.Tensor) -> None:
        '''Starts in the preferred range from a scan of a_p and n_p'''
        low, high = nrange
        first = max(low, self.reference['n_m'] + .04)
        if high <= first:
            return
        ns = np.linspace(first, high,
                         max(int(round((high - first) / .05)) + 1, 3))
        separation = min(.1, (high - first) / 3)
        rows = []
        for i in need:
            z0, a0 = starts[i][0][:2]
            top = z0 * self.reference['magnification'] / self.clearance
            rows.extend((i, z0, aa, nn) for nn in ns
                        for aa in np.geomspace(.5 * a0, 1.3 * a0, 25)
                        if aa <= top)
        if not rows:
            return
        grid = np.array(rows)
        v = self.misfits(grid[:, 0].astype(int), grid[:, 1], grid[:, 2],
                         grid[:, 3], b, m, c)
        for i in need:
            sel = grid[:, 0] == i
            best: dict[float, tuple[float, float, float, float]] = {}
            for (_, zz, aa, nn), vv in zip(grid[sel], v[sel], strict=True):
                if nn not in best or vv < best[nn][3]:
                    best[nn] = (float(zz), float(aa), float(nn), float(vv))
            ranked = sorted(best.values(), key=lambda r: r[3])
            picked: list[tuple[float, float, float, float]] = []
            if explore:      # every point of the grid, by distinct a_p
                have = [r[1] for r in starts[i]]
                every = sorted(((float(zz), float(aa), float(nn), float(vv))
                                for (_, zz, aa, nn), vv in
                                zip(grid[sel], v[sel], strict=True)),
                               key=lambda r: r[3])
                for r in every:
                    if all(abs(r[1] / h - 1) > .03
                           for h in have + [p[1] for p in picked]):
                        picked.append(r)
                    if len(picked) == explore:
                        break
            else:
                for r in ranked:
                    if all(abs(r[2] - p[2]) > separation for p in picked):
                        picked.append(r)
                    if len(picked) == self.preferred:
                        break
            starts[i].extend(picked)
