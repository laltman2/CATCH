'''Fits of the Lorenz-Mie theory from the starts of the Estimator

Needs pylorenzmie (``pip install CATCH[fit]``), which is imported only
when a fit is made.  Each particle is fitted from every start that the
Estimator proposed, in a square crop around it, and
:func:`CATCH.lib.selection.select` chooses among the fits and reports how well
the data constrain the result.
'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np

#: number of pixels in a fit (a random subset of the crop)
NPIXELS = 4000
#: the losses of a fit: plain least squares, the robust Cauchy loss, and
#: both in turn (least squares to reach the basin, then Cauchy to refine it
#: without the pull of outliers such as the fringes of a neighbor)
LOSSES = ('linear', 'cauchy', 'both')
#: bounds of a fit that are not relative to the start
LIMITS: dict[str, Any] = dict(z_p=(20., 3000.), a_p=(0.1, 10.),
                              n_p_max=2.6)


def crop_square(image: np.ndarray, x: float, y: float,
                side: int) -> tuple[np.ndarray, int, int]:
    '''A square crop around (x, y), shifted inward at the edges

    Returns the crop and the column and row of its corner in the image.
    A side larger than the image is reduced to fit.

    Parameters
    ----------
    image : numpy.ndarray
        The image.
    x, y : float
        Center of the crop [pixels].
    side : int
        Side of the crop [pixels].

    Returns
    -------
    crop : numpy.ndarray
        The crop.
    x0, y0 : int
        Column and row of its corner in the image.
    '''
    rows, columns = image.shape[:2]
    side = int(min(side, rows, columns))
    x0 = int(round(x - side / 2))
    y0 = int(round(y - side / 2))
    x0 = min(max(x0, 0), columns - side)
    y0 = min(max(y0, 0), rows - side)
    return image[y0:y0 + side, x0:x0 + side], x0, y0


def fit(model: Any, hologram: Any, radius: float = np.inf, seed: int = 0,
        loss: str = 'linear', limits: Mapping[str, object] | None = None,
        npixels: int | None = None,
        saturation: float | np.ndarray | None = None) -> Any:
    '''Least-squares fit of a model to a hologram, with bounds

    Fits the pixels within ``radius`` of the particle, at most
    :data:`NPIXELS` of them.  The bounds keep a fit from running off to
    unphysical values: the radius, the index and the height are positive,
    and the center stays within 15 pixels of where it started.

    Parameters
    ----------
    model : pylorenzmie.theory.LorenzMie
        Model with one particle, set to the starting values.
    hologram : pylorenzmie.analysis.Hologram
        The data.
    radius : float
        Distance from the particle [pixels].  Default: all the pixels.
    seed : int
        Seed for the choice of the pixels.  Default: 0
    loss : str
        ``'linear'`` (least squares), ``'cauchy'`` (robust) or ``'both'``:
        a least-squares fit, then a robust fit from its result.  Default:
        ``'linear'``
    limits : mapping | None
        Bounds of ``z_p`` [pixels] and ``a_p`` [um] as pairs, and the
        largest ``n_p`` (``n_p_max``).  Default: :data:`LIMITS`
    npixels : int | None
        Pixels in the fit (a random subset).  Default: None, which is
        :data:`NPIXELS`
    saturation : float | numpy.ndarray | None
        Pixels at or above this value, or the ``True`` pixels of this
        boolean mask of the shape of the hologram, are left out of the fit:
        a camera that saturates records the limit of its range, not the
        hologram.  Default: None, every pixel counts.

    Returns
    -------
    result : pandas.Series
        The fitted properties and the reduced chi-squared (``redchi``).

    Raises
    ------
    ValueError
        If ``loss`` is not ``'linear'``, ``'cauchy'`` or ``'both'``.
    '''
    if loss not in LOSSES:
        raise ValueError(f'loss must be one of {LOSSES}, not {loss!r}')
    from pylorenzmie.analysis import Mask, Optimizer
    bounds: dict[str, Any] = dict(LIMITS, **(limits or {}))
    particle = model.single()
    x0, y0 = particle.x_p, particle.y_p
    x, y = hologram.coordinates
    exclude = np.hypot(x - x0, y - y0) > radius
    if isinstance(saturation, np.ndarray):
        exclude = exclude | np.asarray(saturation, bool).reshape(
            exclude.shape)
    elif saturation is not None:
        data = np.asarray(hologram.data).reshape(exclude.shape)
        exclude = exclude | (data >= saturation - 1e-9)
    npix = int((~exclude).sum())
    mask = Mask(shape=hologram.shape,
                fraction=min(1., (npixels or NPIXELS) / max(npix, 1)),
                exclude=exclude, seed=seed)
    optimizer = Optimizer(model=model, mask=mask)
    table = dict(x_p=(x0 - 15, x0 + 15), y_p=(y0 - 15, y0 + 15),
                 z_p=bounds['z_p'], a_p=bounds['a_p'],
                 n_p=(model.instrument.n_m + 0.02, bounds['n_p_max']))
    variables = list(optimizer.variables)
    start = [model.properties[p] for p in variables]
    lower = [table[p][0] for p in variables]
    upper = [table[p][1] for p in variables]
    settings = optimizer.settings
    settings['method'] = 'trf'
    settings['loss'] = 'cauchy' if loss == 'cauchy' else 'linear'
    settings['bounds'] = (lower, upper)
    for p, value, low, high in zip(variables, start, lower, upper,
                                   strict=True):
        if not low < value < high:              # start inside the bounds
            model.properties = {p: min(max(value, low + 1e-6),
                                       high - 1e-6)}
    result = optimizer.optimize(hologram)
    if loss == 'both':
        fitted = result.to_dict()
        model.properties = {p: float(fitted[p]) for p in variables}
        settings['loss'] = 'cauchy'
        result = optimizer.optimize(hologram)
    return result


def start_redchi(model: Any, hologram: Any, noise: float,
                 npixels: int = NPIXELS) -> float:
    '''Reduced chi-squared of the model as it is, before any fit

    A measure of the quality of a start that does not depend on the fit: it
    is large for a start far from the data and about 1 for the truth.  It
    uses at most ``npixels`` pixels, evenly spaced.

    Parameters
    ----------
    model : pylorenzmie.theory.LorenzMie
        Model with one particle, set to the start.  Its coordinates are
        restored afterward.
    hologram : pylorenzmie.analysis.Hologram
        The data.
    noise : float
        Standard deviation of the noise of the data.
    npixels : int
        Largest number of pixels.  Default: :data:`NPIXELS`

    Returns
    -------
    redchi : float
        Mean of the squared residuals in units of ``noise``.
    '''
    data = np.asarray(hologram.flat_data).ravel()
    step = max(1, data.size // npixels)
    model.coordinates = hologram.flat_coordinates[:, ::step]
    residual = data[::step] - np.asarray(model.hologram()).ravel()
    model.coordinates = hologram.flat_coordinates
    return float(np.mean((residual / noise) ** 2))


def fit_particle(data: np.ndarray, noise: float,
                 start: Mapping[str, float],
                 loss: str = 'linear',
                 stages: Sequence[float] = (),
                 instrument: Mapping[str, float] | None = None,
                 limits: Mapping[str, object] | None = None,
                 npixels: int | None = None,
                 saturation: float | np.ndarray | None = None
                 ) -> dict[str, float]:
    '''Fits a model of one particle to a hologram, from a starting point

    Parameters
    ----------
    data : numpy.ndarray
        The normalized hologram (a crop of the image).
    noise : float
        Standard deviation of the noise in the hologram.
    start : mapping
        Starting values of ``x_p`` and ``y_p`` (pixels, in the crop),
        ``z_p`` (pixels), ``a_p`` (micrometers) and ``n_p``.
    loss : str
        ``'linear'``, ``'cauchy'`` or ``'both'`` (see :func:`fit`).
        Default: ``'linear'``
    stages : sequence of float
        Radii of fits that come before the fit of all the pixels, each from
        the result of the one before, as fractions of half the side of the
        crop.  Default: none
    instrument : mapping | None
        ``wavelength``, ``magnification`` and ``n_m``.  Default: None,
        pylorenzmie's default instrument.
    limits : mapping | None
        Bounds (see :func:`fit`).  Default: :data:`LIMITS`
    npixels : int | None
        Pixels in the fit (see :func:`fit`).  Default: None
    saturation : float | numpy.ndarray | None
        Pixels at or above this value, or in this mask, are left out of the
        fit (see :func:`fit`).  Default: None

    Returns
    -------
    result : dict
        The fitted ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``, the
        reduced chi-squared ``redchi`` and that of the start, ``redchi0``.
        All are ``nan`` for a fit that fails.
    '''
    from pylorenzmie.analysis import Hologram
    from pylorenzmie.theory import LorenzMie, Sphere
    hologram = Hologram(data)
    model = LorenzMie()
    model.parallel = False          # the jobs run in parallel instead
    if instrument is not None:
        model.instrument.wavelength = instrument['wavelength']
        model.instrument.magnification = instrument['magnification']
        model.instrument.n_m = instrument['n_m']
    model.instrument.noise = noise
    model.coordinates = hologram.flat_coordinates
    particle = model.single(Sphere)
    particle.x_p, particle.y_p = start['x_p'], start['y_p']
    particle.z_p, particle.a_p, particle.n_p = (start['z_p'], start['a_p'],
                                                start['n_p'])
    keys = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'redchi')
    try:
        redchi0 = start_redchi(model, hologram, noise, npixels or NPIXELS)
        for fraction in stages:
            staged = fit(model, hologram, loss=loss, limits=limits,
                         npixels=npixels, saturation=saturation,
                         radius=fraction * hologram.shape[0] / 2.).to_dict()
            model.properties = {key: float(staged[key]) for key in
                                ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')}
        result = fit(model, hologram, loss=loss, limits=limits,
                     npixels=npixels, saturation=saturation).to_dict()
        out = {key: float(result[key]) for key in keys}
        out['redchi0'] = redchi0
        return out
    except Exception:           # a fit that fails is a result
        return dict.fromkeys((*keys, 'redchi0'), np.nan)


def _fit_all(task: tuple) -> list[dict[str, float]]:
    crop, noise, starts, loss, instrument, limits, *rest = task
    npixels = rest[0] if rest else None
    saturation = rest[1] if len(rest) > 1 else None
    return [fit_particle(crop, noise, s, loss, instrument=instrument,
                         limits=limits, npixels=npixels,
                         saturation=saturation) for s in starts]


class Fitter:
    '''Fits the particles of a frame from the starts of the Estimator

    Parameters
    ----------
    instrument : mapping
        ``wavelength`` and ``magnification`` [um] and ``n_m``.
    loss : str
        See :func:`fit`.  Default: ``'both'``
    nprocs : int
        Processes for the fits; 1 fits in this process.  Default: 1
    side : int | None
        Largest side of a crop [pixels].  Default: None, which is the
        size of the image
    npixels : int
        Pixels in each fit, a random subset of its crop.  More pixels fit
        better and take longer.  Default: :data:`NPIXELS`

    Raises
    ------
    ValueError
        If ``loss`` is not ``'linear'``, ``'cauchy'`` or ``'both'``.
    '''

    def __init__(self, instrument: Mapping[str, float], loss: str = 'both',
                 nprocs: int = 1, side: int | None = None,
                 npixels: int = NPIXELS) -> None:
        if loss not in LOSSES:
            raise ValueError(f'loss must be one of {LOSSES}, not {loss!r}')
        self.instrument = dict(instrument)
        self.loss = loss
        self.nprocs = int(nprocs)
        self.side = int(side) if side else None
        self.npixels = int(npixels)
        # z_p of the fits: from just above the sphere to far beyond the
        # range of the training, in pixels of this instrument
        mag = self.instrument['magnification']
        self.limits = dict(z_p=(1. / mag, 1000. / mag))
        self._pool: ProcessPoolExecutor | None = None

    def close(self) -> None:
        '''Shut down the processes of the fits'''
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __del__(self) -> None:
        self.close()

    def __call__(self, image: np.ndarray, noise: float,
                 particles: Sequence[tuple[float, float, float,
                                           Sequence[tuple[float, float,
                                                          float]]]],
                 saturation: float | np.ndarray | None = None
                 ) -> list[list[dict[str, float]]]:
        '''Fits of each particle, from each of its starts

        Parameters
        ----------
        image : numpy.ndarray
            Normalized hologram.
        noise : float
            Standard deviation of the noise.
        particles : sequence
            ``(x_p, y_p, side, starts)`` per particle: the center and the
            side of its crop [pixels], and its starts ``(z_p, a_p, n_p)``.
        saturation : float | numpy.ndarray | None
            Pixels at or above this value, or in this boolean mask of the
            shape of the image, are left out of the fits (see :func:`fit`).
            Default: None, every pixel counts.

        Returns
        -------
        fits : list of list of dict
            For each particle, the result of :func:`fit_particle` (in
            coordinates of the image) for each start.
        '''
        tasks, corners = [], []
        for x, y, side, starts in particles:
            crop, x0, y0 = crop_square(
                image, x, y, min(int(side), self.side or int(side)))
            corners.append((x0, y0))
            left_out = saturation
            if isinstance(saturation, np.ndarray):
                left_out = saturation[y0:y0 + crop.shape[0],
                                      x0:x0 + crop.shape[1]]
            tasks.append((crop, noise,
                          [dict(x_p=x - x0, y_p=y - y0, z_p=z, a_p=a, n_p=n)
                           for z, a, n in starts],
                          self.loss, self.instrument, self.limits,
                          self.npixels, left_out))
        if self.nprocs > 1 and len(tasks) > 1:
            if self._pool is None:     # kept for the next frames
                self._pool = ProcessPoolExecutor(self.nprocs)
            results = list(self._pool.map(_fit_all, tasks))
        else:
            results = [_fit_all(t) for t in tasks]
        for fits, (x0, y0) in zip(results, corners, strict=True):
            for f in fits:
                f['x_p'] += x0
                f['y_p'] += y0
        return results
