'''Compare two versions on the same particles

Two versions evaluated on the same scenes (a set from ``sets``) succeed or
fail on the same particles, so the comparison should look at the particles
that differ, not at the two rates: most of the noise of a rate is the choice
of particles, and it cancels.  The test is McNemar's: among the particles
on which exactly one version succeeded, is the split different from even?

Usage::

    python -m CATCH.evaluation.paired A_particles.csv B_particles.csv
        [--column ok] [--all]
'''
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEYS = ['scene', 'index']


@dataclass(frozen=True)
class Paired:
    '''The comparison of two versions on the same particles

    Attributes
    ----------
    n : int
        Number of particles.
    both, neither, a_only, b_only : int
        Number of particles on which both versions, neither, only A or only
        B succeeded.
    rate_a, rate_b : float
        The success rates.
    difference : float
        ``rate_b - rate_a``.
    low, high : float
        The 95% confidence interval of ``difference``.
    pvalue : float
        Two-sided exact probability of a split of the particles that differ
        at least as uneven, if the versions were equally good.
    '''
    n: int
    both: int
    neither: int
    a_only: int
    b_only: int
    rate_a: float
    rate_b: float
    difference: float
    low: float
    high: float
    pvalue: float

    def __str__(self) -> str:
        return (f'{self.n} particles: A {self.rate_a:.1%}, B {self.rate_b:.1%}'
                f'; B - A = {self.difference:+.1%} '
                f'[{self.low:+.1%}, {self.high:+.1%}] (95%), '
                f'p = {self.pvalue:.3g}; only A {self.a_only}, '
                f'only B {self.b_only}, both {self.both}, '
                f'neither {self.neither}')


def paired(a: pd.DataFrame, b: pd.DataFrame, column: str = 'ok',
           inside_only: bool = True) -> Paired:
    '''Compares two versions on the particles that both evaluated

    Parameters
    ----------
    a, b : pandas.DataFrame
        The ``particles`` of two ``SceneResults`` from the same scenes.
    column : str
        The Boolean column to compare: ``'ok'`` (the fit locked on),
        ``'close'``, ``'found'`` or another.  Default: ``'ok'``
    inside_only : bool
        Count only the particles whose centers are in the image.
        Default: True

    Returns
    -------
    comparison : Paired

    Raises
    ------
    ValueError
        If the two do not share particles (they are not of the same scenes).
    '''
    merged = a[[*KEYS, column, 'inside']].merge(
        b[[*KEYS, column]], on=KEYS, suffixes=('_a', '_b'))
    if inside_only:
        merged = merged[merged['inside'].astype(bool)]
    if len(merged) == 0 or len(merged) != len(a[a['inside']] if inside_only
                                               else a):
        raise ValueError('the two sets of results are not of the same '
                         'particles')
    ok_a = np.asarray(pd.Series(merged[column + '_a']), dtype=bool)
    ok_b = np.asarray(pd.Series(merged[column + '_b']), dtype=bool)
    n = len(merged)
    a_only = int(np.sum(ok_a & ~ok_b))
    b_only = int(np.sum(~ok_a & ok_b))
    both = int(np.sum(ok_a & ok_b))
    discordant = a_only + b_only
    pvalue = (binomtest(b_only, discordant, 0.5).pvalue
              if discordant else 1.)
    difference = (b_only - a_only) / n
    # the standard error of a difference of paired proportions
    error = np.sqrt(max(discordant - (b_only - a_only) ** 2 / n, 0.)) / n
    return Paired(n=n, both=both, neither=n - both - discordant,
                  a_only=a_only, b_only=b_only,
                  rate_a=float(ok_a.mean()), rate_b=float(ok_b.mean()),
                  difference=float(difference),
                  low=float(difference - 1.96 * error),
                  high=float(difference + 1.96 * error), pvalue=float(pvalue))


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    '''Compare two runs of the scene harness, particle by particle

    Parameters
    ----------
    argv : list[str] | None
        Command line.  Default: None, ``sys.argv``
    '''
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('a', help='particles CSV file of version A')
    parser.add_argument('b', help='particles CSV file of version B')
    parser.add_argument('--column', default='ok')
    parser.add_argument('--all', action='store_true',
                        help='include particles outside the image')
    args = parser.parse_args(argv)
    print(paired(pd.read_csv(args.a), pd.read_csv(args.b), args.column,
                 inside_only=not args.all))


if __name__ == '__main__':  # pragma: no cover
    main()
