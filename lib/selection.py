'''The reported fit of a particle, and how well the data constrain it

A particle is fitted from several starts, and large spheres far from the
focal plane often give several fits of nearly the same quality: different
sizes and indexes whose holograms are nearly the same.  :func:`select`
chooses among them and reports the ambiguity.

The choice:

* with no prior: the best fit with ``n_p`` up to :data:`NMAX`, unless a fit
  with a higher index has a reduced chi-squared lower by more than
  :data:`DELTA` (real data rarely tell the two apart better than that,
  and most samples are in the standard range);
* with a prior ``(low, high)`` for the sample: the best fit with ``n_p``
  within :data:`PAD` of the range, or, if there is none, the best fit,
  flagged ``'outside prior'``.

The ambiguity: the distinct solutions (by ``a_p``, :data:`SAME_A` apart)
whose reduced chi-squared is within :data:`TOL` of the reported fit's
(inside the range of the prior, if there is one), their number and the
spread of ``a_p``, ``n_p`` and ``z_p``.  It counts only the solutions that
the starts found, so it is a lower bound: a sample with many
near-degenerate minima (such as polystyrene spheres of 3 um far from the
focal plane) can look unambiguous when it is not.
'''
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: the largest index of the standard range
NMAX = 1.8
#: a fit above the standard range must be better by this fraction
DELTA = 0.05
#: tolerance of the range of a prior
PAD = 0.02
#: fits within this fraction of the best are equally good
TOL = 0.005
#: fits whose radii differ by less than this [um] are the same solution
SAME_A = 0.005

#: the columns that :func:`select` adds
COLUMNS = ('x_fit', 'y_fit', 'z_fit', 'a_fit', 'n_fit', 'redchi', 'flag',
           'nsolutions', 'a_spread', 'n_spread', 'z_spread', 'alt_a',
           'alt_n', 'alt_redchi')


def distinct(fits: Sequence[dict]) -> list[dict]:
    '''One fit per solution (by ``a_p``), the best of each, best first'''
    out: list[dict] = []
    for f in sorted(fits, key=lambda g: g['redchi']):
        if all(abs(f['a_p'] - g['a_p']) >= SAME_A for g in out):
            out.append(f)
    return out


def select(fits: Sequence[dict],
           nrange: tuple[float, float] | None = None) -> dict[str, Any]:
    '''The reported fit and the ambiguity of a particle

    Parameters
    ----------
    fits : sequence of dict
        Fits from the starts: ``x_p``, ``y_p``, ``z_p``, ``a_p``, ``n_p``
        and ``redchi`` (``nan`` for a fit that failed).
    nrange : tuple of float | None
        Prior range of ``n_p``.  Default: None

    Returns
    -------
    result : dict
        The :data:`COLUMNS`: the reported fit (``x_fit`` ... ``n_fit``,
        ``redchi``), a ``flag`` (``''``, ``'high index'``, ``'outside
        prior'`` or ``'no fit'``), the number of equally good solutions
        and their spreads, and the best other solution (``alt_*``).
    '''
    ok = [f for f in fits if np.isfinite(f['redchi'])]
    out: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    if not ok:
        out.update(flag='no fit', nsolutions=0)
        return out
    flag = ''
    if nrange is None:
        low = [f for f in ok if f['n_p'] <= NMAX]
        high = [f for f in ok if f['n_p'] > NMAX]
        b = min(low, key=lambda f: f['redchi']) if low else None
        h = min(high, key=lambda f: f['redchi']) if high else None
        if b is None or (h is not None
                         and h['redchi'] < (1 - DELTA) * b['redchi']):
            best, flag = h, 'high index'
        else:
            best = b
        pool = ok
    else:
        lo, hi = nrange
        pool = [f for f in ok if lo - PAD <= f['n_p'] <= hi + PAD]
        if pool:
            best = min(pool, key=lambda f: f['redchi'])
        else:
            best, flag, pool = min(ok, key=lambda f: f['redchi']), \
                'outside prior', ok
    assert best is not None
    near = [f for f in distinct(pool)
            if f['redchi'] <= (1 + TOL) * best['redchi']]
    if not any(f is best for f in near):
        near.append(best)

    def spread(key: str) -> float:
        return float(max(f[key] for f in near) - min(f[key] for f in near))

    others = [f for f in distinct(ok)
              if abs(f['a_p'] - best['a_p']) >= SAME_A]
    out.update(x_fit=best['x_p'], y_fit=best['y_p'], z_fit=best['z_p'],
               a_fit=best['a_p'], n_fit=best['n_p'], redchi=best['redchi'],
               flag=flag, nsolutions=len(near), a_spread=spread('a_p'),
               n_spread=spread('n_p'), z_spread=spread('z_p'))
    if others:
        alt = min(others, key=lambda f: f['redchi'])
        out.update(alt_a=alt['a_p'], alt_n=alt['n_p'],
                   alt_redchi=alt['redchi'])
    return out
