'''Does a network trained for one instrument serve another?

The same particles are simulated as the reference instrument sees them and
as another instrument sees them (see ``CATCH.instruments``), and the
``Estimator`` is run on both.  If the conversion is right, the errors
agree, in units of the instrument's own length scales.

Usage::

    python -m CATCH.evaluation.equivalence [--ncases 180]
        [--wavelength 0.532] [--magnification 0.03] [--n_m 1.33]
'''
import argparse
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from pylorenzmie.theory import LorenzMie, Sphere

from CATCH.evaluation.cases import INSTRUMENT, Case, extent, make_cases
from CATCH.instruments import from_reference, pixel_ratio

Estimate = Callable[[np.ndarray], pd.DataFrame]


def simulate(instrument: Mapping[str, float],
             a_p: float, n_p: float, z_p: float,
             size: int, seed: int, noise: float = 0.05) -> np.ndarray:
    '''8-bit crop of one particle at its center, as an instrument sees it

    Parameters
    ----------
    instrument : mapping
        ``wavelength`` and ``magnification`` [um], and ``n_m``.
    a_p, n_p, z_p : float
        Radius [um], refractive index and axial position [pixels of the
        instrument].
    size : int
        Side of the crop [pixels].
    seed : int
        Seed of the noise.
    noise : float
        Standard deviation of the noise.  Default: 0.05
    '''
    rng = np.random.default_rng(seed)
    model = LorenzMie()
    model.instrument.properties = dict(instrument)
    model.coordinates = model.meshgrid((size, size))
    sphere = model.single(Sphere)
    sphere.a_p, sphere.n_p = a_p, n_p
    center = (size - 1) / 2.
    sphere.r_p = [center, center, z_p]
    data = model.hologram().reshape(size, size)
    data += noise * rng.normal(size=(size, size))
    return np.clip(100. * data, 0, 255).astype(np.uint8)


def compare(estimate: Estimate,
            instrument: Mapping[str, float],
            cases: Sequence[Case]) -> pd.DataFrame:
    '''Errors of estimates for the particles of cases, on an instrument

    The crop is twice the extent of the feature, as in training, in pixels
    of the instrument.  The errors are relative to the scales of the
    instrument: ``da`` to the radius, ``dz`` to the axial position and
    ``dn`` to the contrast of the index, ``n_p - n_m``.

    Parameters
    ----------
    estimate : callable
        Takes a crop and returns a one-row DataFrame of ``z_p``, ``a_p`` and
        ``n_p`` in the units of ``instrument``, such as
        ``Estimator(instrument=instrument)``.
    instrument : mapping
        The instrument, for example ``INSTRUMENT`` (the reference).
    cases : sequence of Case
        The particles, as the reference instrument sees them.

    Returns
    -------
    errors : pandas.DataFrame
        One row for each case, with ``a_p``, ``n_p``, ``z_p`` (true values
        for ``instrument``), ``size`` of the crop, and ``da``, ``dn``, ``dz``.
    '''
    ratio = pixel_ratio(instrument, INSTRUMENT)
    rows = []
    for case in cases:
        size = max(int(round(2 * extent(case) / ratio)), 8)
        z_p, a_p, n_p = from_reference(case.z_p, case.a_p, case.n_p,
                                       instrument, INSTRUMENT)
        found = estimate(simulate(instrument, a_p, n_p, z_p, size,
                                  case.seed)).iloc[0]
        rows.append({'a_p': a_p, 'n_p': n_p, 'z_p': z_p, 'size': size,
                     'da': (found.a_p - a_p) / a_p,
                     'dn': (found.n_p - n_p) / (n_p - instrument['n_m']),
                     'dz': (found.z_p - z_p) / z_p})
    return pd.DataFrame(rows)


def summarize(errors: pd.DataFrame) -> pd.Series:
    '''Median absolute errors and the fraction of radii within 5%'''
    return pd.Series({
        'size': errors['size'].median(),
        'median |da|': errors['da'].abs().median(),
        'median |dn|': errors['dn'].abs().median(),
        'median |dz|': errors['dz'].abs().median(),
        'a within 5%': (errors['da'].abs() < 0.05).mean()})


def main() -> None:  # pragma: no cover
    from CATCH.Estimator import Estimator

    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--ncases', type=int, default=180)
    parser.add_argument('--wavelength', type=float, default=0.532)
    parser.add_argument('--magnification', type=float, default=0.03)
    parser.add_argument('--n_m', type=float, default=1.33)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    instrument = {'wavelength': args.wavelength,
                  'magnification': args.magnification, 'n_m': args.n_m}
    cases = make_cases(args.ncases)
    table = {}
    for name, inst in (('reference', INSTRUMENT), ('other', instrument)):
        estimator = Estimator(args.model, instrument=inst)
        table[name] = summarize(compare(estimator, inst, cases))
    print(f'pixel ratio {pixel_ratio(instrument, INSTRUMENT):.3f}')
    print(pd.DataFrame(table).to_string(float_format='%.3f'))


if __name__ == '__main__':  # pragma: no cover
    main()
