'''The extent of the hologram of a particle (for the evaluation harness)

The model of a hologram and the radius of its 20th zero crossing (about 10
bright rings), which the evaluation harness uses to size its crops.
'''
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pylorenzmie.theory import LMHologram, Sphere


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


def feature_extent(sphere: Sphere, config: Mapping[str, Any],
                   nfringes: int = 20, maxrange: int = 300) -> float:
    '''Radius of holographic feature in pixels

    Parameters
    ----------
    sphere : pylorenzmie.theory.Sphere
        The particle: only ``a_p``, ``n_p`` and ``z_p`` are used.
    config : dict
        Configuration with the properties of the ``instrument``.
    nfringes : int
        The feature ends at the zero crossing of this number.
        Default: 20
    maxrange : int
        The largest radius to look at, in pixels, which is returned if the
        hologram has no more than ``nfringes`` zero crossings in that range.
        Default: 300

    Returns
    -------
    extent : float
        Distance in pixels from the center of the particle to the
        ``nfringes``-th zero crossing of the hologram minus its background.
    '''

    # one row of pixels outward from the particle; a grid would run the
    # crossings of the next row on when the first has too few
    h = make_model((1, maxrange), config['instrument'])
    h.particle = Sphere(a_p=sphere.a_p, n_p=sphere.n_p, z_p=sphere.z_p)
    # roughly estimate radii of zero crossings
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    if len(ndx) <= nfringes:
        return float(maxrange)
    return float(ndx[nfringes])
