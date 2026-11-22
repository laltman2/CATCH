'''The extent of the hologram of a particle (for the evaluation harness)

The model of a hologram and the radius of its 20th zero crossing (about 10
bright rings), which the evaluation harness uses to size its crops.
'''
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from pylorenzmie.theory import LMHologram, LorenzMie, Sphere

from CATCH.lib.instruments import REFERENCE


def make_model(shape: Sequence[int],
               instrument: Mapping[str, float]) -> Any:
    '''Returns a hologram calculator for an image of the given shape

    Parameters
    ----------
    shape : sequence of int
        ``(rows, columns)`` of the image, in pixels.
    instrument : mapping
        Properties of the instrument: ``wavelength`` and ``magnification``
        (micrometers per pixel), and ``n_m``, the refractive index of the
        medium.

    Returns
    -------
    model : pylorenzmie.theory.LMHologram
        Set ``model.particle`` and call ``model.hologram()``, which returns
        the flattened image.
    '''
    model = LMHologram()
    model.instrument.properties = dict(instrument)
    rows, columns = shape
    model.coordinates = model.meshgrid((rows, columns))
    return model


class Extent:
    '''Radius of the feature of a particle [pixels], with one model

    The distance from the center of the hologram of a sphere to its
    ``nfringes``-th zero crossing (the 20th, about 10 bright rings), found
    along a row of ``maxrange`` pixels, or ``maxrange`` if there are fewer.
    The calculator is made once, which makes a call ten times faster than
    one that makes it for each (:func:`feature_extent` shares them).

    Parameters
    ----------
    instrument : mapping | None
        ``wavelength`` and ``magnification`` [um] and ``n_m``.  Default:
        None, the reference instrument.
    nfringes : int
        Order of the zero crossing.  Default: 20
    maxrange : int
        Length of the row, and the value for a hologram with fewer zero
        crossings [pixels].  Default: 300
    '''

    def __init__(self, instrument: Mapping[str, float] | None = None,
                 nfringes: int = 20, maxrange: int = 300) -> None:
        try:
            from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie
            model = numbaLorenzMie()
        except ImportError:  # pragma: no cover
            model = LorenzMie()
        model.instrument.properties = dict(instrument or REFERENCE)
        model.coordinates = model.meshgrid((1, maxrange))
        self.model = model
        self.nfringes = nfringes
        self.maxrange = maxrange

    def __call__(self, z_p: float, a_p: float, n_p: float) -> float:
        '''Extent of the hologram of a sphere [pixels of the instrument]

        Parameters
        ----------
        z_p, a_p, n_p : float
            Axial position [pixels], radius [um] and refractive index.

        Returns
        -------
        extent : float
            The distance of the zero crossing of order ``nfringes`` from the
            center, or ``maxrange`` if the hologram has fewer.
        '''
        self.model.particle = Sphere(a_p=a_p, n_p=n_p, z_p=z_p)
        crossings = np.flatnonzero(
            np.diff(np.sign(np.asarray(self.model.hologram()) - 1.))) + 1
        if len(crossings) <= self.nfringes:
            return float(self.maxrange)
        return float(crossings[self.nfringes])


@lru_cache(maxsize=16)
def _extent_of(instrument: tuple[tuple[str, float], ...], nfringes: int,
               maxrange: int) -> Extent:
    return Extent(dict(instrument), nfringes, maxrange)


def feature_extent(sphere: Sphere, config: Mapping[str, Any],
                   nfringes: int = 20, maxrange: int = 300) -> float:
    '''Radius [pixels] of the feature of a sphere

    Parameters
    ----------
    sphere : pylorenzmie.theory.Sphere
        The particle.
    config : mapping
        Configuration with the properties of the ``instrument``.
    nfringes : int
        Order of the zero crossing.  Default: 20
    maxrange : int
        Longest radius [pixels].  Default: 300

    Returns
    -------
    extent : float
        See :class:`Extent`, whose calculator is kept for the next call.
    '''
    instrument = tuple(sorted(dict(config['instrument']).items()))
    return _extent_of(instrument, nfringes, maxrange)(
        sphere.z_p, sphere.a_p, sphere.n_p)
