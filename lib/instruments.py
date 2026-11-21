'''Use a network trained for one instrument with another

A hologram in pixels depends on the instrument only through the pixel size
in wavelengths of the medium, ``magnification / (wavelength / n_m)``, and
on the particle through ``a_p / lambda``, ``z_p * magnification / lambda``
and the relative index ``n_p / n_m``, where ``lambda = wavelength / n_m``.
A network trained for one (reference) instrument can therefore serve another
if the crop is described in units of the reference pixel and the outputs are
converted back.  The conversion is exact for the Lorenz-Mie theory of an
ideal sphere; it does not describe aberrations, the noise of a camera or
dispersion of the refractive indexes, and a particle may fall outside the
ranges of size and depth that the network was trained for.
'''
from collections.abc import Mapping
from typing import Any

#: the instrument of the shipped networks
REFERENCE = {'wavelength': 0.447, 'magnification': 0.048, 'n_m': 1.34}


def wavelength_in_medium(instrument: Mapping[str, float]) -> float:
    '''Wavelength [um] in the medium of an instrument

    Parameters
    ----------
    instrument : mapping
        ``wavelength`` [um] and ``n_m``.

    Returns
    -------
    wavelength : float
        ``wavelength / n_m``.
    '''
    return instrument['wavelength'] / instrument['n_m']


def pixel_ratio(instrument: Mapping[str, float],
                reference: Mapping[str, float] = REFERENCE) -> float:
    '''Size of a pixel of an instrument in pixels of the reference

    Both are measured in wavelengths of the medium.  A crop of ``N`` pixels
    of the instrument is a crop of ``N * pixel_ratio`` pixels of the
    reference.  A ratio above 1 means that the instrument samples the
    hologram more coarsely than the reference, so that the outer fringes may
    be undersampled.

    Parameters
    ----------
    instrument : mapping
        ``wavelength`` and ``magnification`` [um], and ``n_m``.
    reference : mapping
        The instrument of the network.  Default: ``REFERENCE``

    Returns
    -------
    ratio : float
        Pixel of ``instrument`` divided by the pixel of ``reference``.
    '''
    pixel = instrument['magnification'] / wavelength_in_medium(instrument)
    pixel0 = reference['magnification'] / wavelength_in_medium(reference)
    return pixel / pixel0


def from_reference(z_p: Any, a_p: Any, n_p: Any,
                   instrument: Mapping[str, float],
                   reference: Mapping[str, float] = REFERENCE
                   ) -> tuple[Any, Any, Any]:
    '''Converts properties from the reference to an instrument

    Parameters
    ----------
    z_p, a_p, n_p : float or array_like
        Axial position [reference pixels], radius [um] and refractive index
        of a particle as the reference instrument sees it.
    instrument, reference : mapping
        See :func:`pixel_ratio`.  Default of ``reference``: ``REFERENCE``

    Returns
    -------
    z_p, a_p, n_p : float or array_like
        The same particle as the instrument sees it: ``z_p`` in pixels of
        the instrument.
    '''
    ratio = pixel_ratio(instrument, reference)
    scale = wavelength_in_medium(instrument) / wavelength_in_medium(reference)
    return (z_p / ratio,
            a_p * scale,
            n_p / reference['n_m'] * instrument['n_m'])


def to_reference(z_p: Any, a_p: Any, n_p: Any,
                 instrument: Mapping[str, float],
                 reference: Mapping[str, float] = REFERENCE
                 ) -> tuple[Any, Any, Any]:
    '''Converts properties from an instrument to the reference

    The inverse of :func:`from_reference`.

    Parameters
    ----------
    z_p, a_p, n_p : float or array_like
        Axial position [pixels of the instrument], radius [um] and
        refractive index of a particle as the instrument sees it.
    instrument, reference : mapping
        See :func:`pixel_ratio`.  Default of ``reference``: ``REFERENCE``

    Returns
    -------
    z_p, a_p, n_p : float or array_like
        The same particle as the reference instrument sees it: ``z_p`` in
        reference pixels.
    '''
    ratio = pixel_ratio(instrument, reference)
    scale = wavelength_in_medium(reference) / wavelength_in_medium(instrument)
    return (z_p * ratio,
            a_p * scale,
            n_p / instrument['n_m'] * reference['n_m'])
