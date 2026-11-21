'''Radial profiles of Lorenz-Mie holograms of many spheres at once

The Estimator checks dozens of candidates for every particle by comparing
the radial profile of the hologram with the profile of the model.  This
module computes those model profiles for a batch of spheres in one pass,
on a GPU if there is one.

The profile of a sphere at height ``z_p`` above the focal plane, lit by a
plane wave polarized along ``x``, is the mean of the hologram along the
``x`` and ``y`` axes through its center: on the ``x`` axis only the
scattered field's ``E_r`` and ``E_theta`` survive, on the ``y`` axis only
``E_phi``, and the mean cancels the ``cos 2 phi`` dependence of the
polarization.

Credits
-------
The partial-wave sum is a port of the Numba kernel of pylorenzmie
(Bohren and Huffman conventions, the upward recurrences of Wiscombe), and
:func:`coefficients` follows pylorenzmie's ``mie_ab`` (the recurrences of
Yang, *Appl. Opt.* **42**, 1710 (2003)).  pylorenzmie is distributed under
the GPL; its author, David G. Grier, permits the use of these ports in
CATCH under the MIT license.

Differences from pylorenzmie: the coefficients are vectorized over spheres
and cut at the Wiscombe criterion (the Yang term only adds orders that do
not change the field outside the sphere; the profiles agree to 4e-8), and
the propagation phase ``exp(i k (r - z_p))`` is factored out of the
Riccati-Bessel recurrence and computed in double precision, so that single
precision is enough on the GPU (the profiles agree to 1e-4).
'''
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch

#: the downward recurrence starts ``MARGIN[0] + MARGIN[1] * L**(1/3)``
#: orders above the last order ``L`` (see pylorenzmie ``_mie.start_margin``)
MARGIN = (15., 6.)


def orders(x: np.ndarray) -> np.ndarray:
    '''Number of partial waves of spheres of size parameter ``x``

    The criterion of Wiscombe (1980).
    '''
    x = np.asarray(x, float)
    return np.where(x <= 8., np.floor(x + 4. * np.cbrt(x) + 1.),
                    np.floor(x + 4.05 * np.cbrt(x) + 2.)).astype(int)


def coefficients(x: np.ndarray, m: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    '''Mie coefficients ``a_n`` and ``b_n`` of homogeneous spheres

    Parameters
    ----------
    x : numpy.ndarray, shape (K,)
        Size parameters in the medium, ``k a_p``.
    m : numpy.ndarray, shape (K,)
        Refractive indexes relative to the medium (real).

    Returns
    -------
    a, b : numpy.ndarray, shape (K, L), complex
        The coefficients, ``L = 1 + max(orders(x))``, with ``a[:, 0] = 0``
        and zeros above the number of orders of each sphere.
    '''
    x = np.asarray(x, float)
    m = np.asarray(m, float).astype(complex)
    K = len(x)
    nstop = orders(x)
    L = int(nstop.max()) + 1 if K else 1
    # the recurrence inside the sphere must start above |m x| (Yang)
    top = max(L, int(np.abs(x * m).max()) + 1 if K else 1)
    start = top + int(MARGIN[0] + MARGIN[1] * np.cbrt(top))
    z = x * m
    # logarithmic derivatives, downward from beyond the last order
    D_in = np.zeros((K, L), complex)
    D_out = np.zeros((K, L), complex)
    d_in = np.zeros(K, complex)
    d_out = np.zeros(K, complex)
    for n in range(start, 0, -1):
        d_in = n / z - 1. / (d_in + n / z)
        d_out = n / x - 1. / (d_out + n / x)
        if n - 1 < L:
            D_in[:, n - 1] = d_in
            D_out[:, n - 1] = d_out
    # upward recurrences of psi, zeta and D3, stopped at each sphere's order
    psi = np.zeros((K, L), complex)
    zeta = np.zeros((K, L), complex)
    psi[:, 0] = np.sin(x)
    zeta[:, 0] = -1j * np.exp(1j * x)
    psizeta = .5 * (1. - np.exp(2j * x))
    D3 = 1j * np.ones(K, complex)
    for n in range(1, L):
        live = n <= nstop
        f1 = n / x - D_out[:, n - 1]
        f3 = n / x - D3
        psi[:, n] = np.where(live, psi[:, n - 1] * f1, 0.)
        zeta[:, n] = np.where(live, zeta[:, n - 1] * f3, 1.)
        psizeta = np.where(live, psizeta * f1 * f3, 1.)
        D3 = D_out[:, n] + 1j / psizeta
    n = np.arange(L)[None, :]
    live = (n >= 1) & (n <= nstop[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        fa = D_in / m[:, None] + n / x[:, None]
        fb = D_in * m[:, None] + n / x[:, None]
        a = (fa * psi - np.roll(psi, 1, 1)) / (fa * zeta - np.roll(zeta, 1, 1))
        b = (fb * psi - np.roll(psi, 1, 1)) / (fb * zeta - np.roll(zeta, 1, 1))
    return np.where(live, a, 0.), np.where(live, b, 0.)


def _step(state: torch.Tensor, cf: torch.Tensor, inv: torch.Tensor,
          cos_t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    '''One partial wave of the sum, for all spheres and radii

    ``state`` holds (pi_{n-1}, pi_n, xi_{n-2}, xi_{n-1} as re and im, and
    the running sums E_r, E_theta, E_phi as re and im); complex numbers are
    pairs of reals, which every backend supports.
    '''
    (pi1, pi, xi2r, xi2i, xi1r, xi1i,
     err, eri, etr, eti, epr, epi) = state.unbind(0)
    a0r, a0i, ar, ai, br, bi = cf.unbind(0)
    sw = pi * cos_t
    tw = sw - pi1
    tau = pi1 - k * tw                        # -tau_n
    xir = (2 * k - 1) * xi1r * inv - xi2r
    xii = (2 * k - 1) * xi1i * inv - xi2i
    dr = k * xir * inv - xi1r
    di = k * xii * inv - xi1i
    mtr, mti, mpr, mpi = pi * xir, pi * xii, tau * xir, tau * xii
    ntr, nti, npr, npi = tau * dr, tau * di, pi * dr, pi * di
    return torch.stack([
        pi, sw + (1 + 1 / k) * tw, xi1r, xi1i, xir, xii,
        err + a0r * mtr - a0i * mti, eri + a0r * mti + a0i * mtr,
        etr + ar * ntr - ai * nti - (br * mtr - bi * mti),
        eti + ar * nti + ai * ntr - (br * mti + bi * mtr),
        epr + ar * npr - ai * npi - (br * mpr - bi * mpi),
        epi + ar * npi + ai * npr - (br * mpi + bi * mpr)])


class RadialModel:
    '''Model profiles of many spheres on a fixed set of radii

    Parameters
    ----------
    radii : numpy.ndarray
        Distances from the axis of the sphere [pixels].
    wavelength : float
        Vacuum wavelength [um].
    magnification : float
        Size of a pixel [um].
    n_m : float
        Refractive index of the medium.
    device : torch.device or str
        Where to compute.  Default: ``'cpu'``
    chunk : int
        Spheres per pass; every pass has this shape, so that a compiled
        step is made once.  Default: 512
    dtype : torch.dtype or None
        Precision of the sum.  Default: None, which is ``torch.float64``
        except on Apple's GPU (``mps``), which has no double precision.
        In single precision the profiles of large spheres close to the
        focal plane (``a_p`` above 4 um and ``z_p`` below about six radii)
        are off by up to 0.02, elsewhere by less than 1e-3.
    compile : bool
        Compile the step of the partial-wave sum with ``torch.compile``
        (one fused kernel per partial wave instead of dozens of small ones,
        about five times faster on a GPU).  If compiling fails, the step
        runs uncompiled, with a warning.  Default: True
    '''

    def __init__(self, radii: np.ndarray, wavelength: float,
                 magnification: float, n_m: float,
                 device: torch.device | str = 'cpu', chunk: int = 512,
                 dtype: torch.dtype | None = None,
                 compile: bool = True) -> None:
        self.radii = np.asarray(radii, float)
        self.n_m = float(n_m)
        self.k_um = 2. * np.pi * self.n_m / float(wavelength)
        self.k_px = self.k_um * float(magnification)
        self.device = torch.device(device)
        self.chunk = int(chunk)
        if dtype is None:
            dtype = (torch.float32 if self.device.type == 'mps'
                     else torch.float64)
        self.dtype = dtype
        self._ks: list[torch.Tensor] = []
        self.step: Callable[..., torch.Tensor] = _step
        if compile:
            try:
                self.step = torch.compile(_step, dynamic=False)
            except Exception as error:  # pragma: no cover
                warnings.warn(f'torch.compile failed ({error}); the model '
                              'profiles are computed uncompiled',
                              stacklevel=2)

    def _k(self, n: int) -> torch.Tensor:
        while len(self._ks) <= n:
            self._ks.append(torch.tensor(float(len(self._ks)),
                                         device=self.device,
                                         dtype=self.dtype))
        return self._ks[n]

    def __call__(self, z_p: np.ndarray, a_p: np.ndarray,
                 n_p: np.ndarray) -> torch.Tensor:
        '''Profiles, shape (K, len(radii)), on the device

        Parameters
        ----------
        z_p : numpy.ndarray
            Heights above the focal plane [pixels], positive.
        a_p : numpy.ndarray
            Radii [um].
        n_p : numpy.ndarray
            Refractive indexes.
        '''
        z_p, a_p, n_p = (np.atleast_1d(np.asarray(q, float))
                         for q in (z_p, a_p, n_p))
        parts = [self._chunk(z_p[i:i + self.chunk], a_p[i:i + self.chunk],
                             n_p[i:i + self.chunk])
                 for i in range(0, len(z_p), self.chunk)]
        if not parts:
            return torch.zeros((0, len(self.radii)), device=self.device,
                               dtype=self.dtype)
        return torch.cat(parts)

    def _chunk(self, z: np.ndarray, a: np.ndarray,
               n: np.ndarray) -> torch.Tensor:
        count = len(z)
        if self.step is not _step and count < self.chunk:
            pad = self.chunk - count       # one shape for the compiled step
            z, a, n = (np.r_[q, np.repeat(q[:1], pad)] for q in (z, a, n))
        A, B = coefficients(self.k_um * a, n / self.n_m)
        L = int(np.ceil(A.shape[1] / 16) * 16)
        A = np.pad(A, ((0, 0), (0, L - A.shape[1])))
        B = np.pad(B, ((0, 0), (0, L - B.shape[1])))
        order = np.arange(L, dtype=float)
        nn = np.maximum(order, 1.)
        En = (1j ** order) * (2 * order + 1) / (nn * nn + nn)
        an = 1j * En * A
        an0 = an * (order * order + order)
        bn = En * B
        R = self.radii
        krho = self.k_px * R[None, :] * np.ones_like(z)[:, None]
        kz = self.k_px * z[:, None] * np.ones_like(R)[None, :]
        kr = np.hypot(krho, kz)
        phase = krho ** 2 / (kr + kz)            # k (r - z_p), no cancellation
        T = lambda q: torch.as_tensor(q, device=self.device,  # noqa: E731
                                      dtype=self.dtype)
        cf = T(np.stack([an0.real, an0.imag, an.real, an.imag,
                         bn.real, bn.imag]))
        inv, cos_t, sin_t = T(1 / kr), T(kz / kr), T(krho / kr)
        rr, ri = T(np.cos(phase)), T(np.sin(phase))
        one, zero = torch.ones_like(inv), torch.zeros_like(inv)
        # xi_{-1} = 1 and xi_0 = -i once exp(i k r) is factored out
        state = torch.stack([zero, one, one, zero, zero, -one,
                             zero, zero, zero, zero, zero, zero])
        for k in range(1, L):
            state = self.step(state, cf[:, :, k:k + 1], inv, cos_t,
                              self._k(k))
        err, eri, etr, eti, epr, epi = state[6:].unbind(0)

        def turn(re, im, g):
            return (re * rr - im * ri) * g, (re * ri + im * rr) * g

        err, eri = turn(err, eri, sin_t * inv * inv)
        etr, eti = turn(etr, eti, inv)
        epr, epi = turn(epr, epi, inv)
        exr, exi = err * sin_t + etr * cos_t, eri * sin_t + eti * cos_t
        ezr, ezi = err * cos_t - etr * sin_t, eri * cos_t - eti * sin_t
        h = ((1 + exr) ** 2 + exi ** 2 + ezr ** 2 + ezi ** 2
             + (1 - epr) ** 2 + epi ** 2) / 2
        return h[:count]
