'''The extent of the hologram of a particle (for the evaluation harness)

What remains of the data generation of CATCH 2: the model of a hologram and
the radius of its 20th fringe, which the harness uses to size its crops.
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
    model.coordinates = model.meshgrid(tuple(shape))
    return model


def particle_properties(sphere: Sphere) -> dict[str, float]:
    '''Returns the properties of a sphere, as plain floats'''
    return {key: float(value) for key, value in sphere.properties.items()}


def feature_extent(sphere, config, nfringes=20, maxrange=300):
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
        return maxrange
    else:
        return float(ndx[nfringes])
