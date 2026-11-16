'''Evaluate an estimator: how often does a fit that starts from it lock on?

Usage::

    python -m CATCH.evaluation.run [--ncases 180] [--nprocs 4]
        [--model default] [--output results.csv]
'''
import argparse
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from CATCH.evaluation.cases import (CONDITIONS, Case, Condition,
                                    make_cases, simulate)
from CATCH.evaluation.fitting import Job, locks_on, refine

#: takes the crops and returns a DataFrame of ``z_p``, ``a_p`` and ``n_p``
Estimate = Callable[[list[np.ndarray]], pd.DataFrame]


def evaluate(estimate: Estimate,
             cases: Sequence[Case],
             condition: Condition = 'training',
             nprocs: int = 1) -> pd.DataFrame:
    '''Estimates the parameters of each case and fits from the estimates

    Parameters
    ----------
    estimate : callable
        Takes a list of the crops (numpy arrays) and returns a DataFrame
        with one row for each, and columns ``z_p``, ``a_p`` and ``n_p``.
    cases : sequence of Case
        The particles.
    condition : ``'fixed'`` or ``'training'``
        Size and noise of the crops; see ``CATCH.evaluation.cases``.
        Default: ``'training'``
    nprocs : int
        Number of processes for the fits.  Default: 1, which fits in this
        process.

    Returns
    -------
    results : pandas.DataFrame
        A row for each case, with the case, the ``condition``, the ``size``
        of the crop, the estimates (``z_est``, ``a_est``, ``n_est``), the
        time of the estimate (``est_seconds``, per crop), the fit (``a_fit``,
        ``n_fit``, ``redchi``), the relative errors of the estimates
        (``z_rel``, ``a_rel``, ``n_rel``), whether all of them are within 2%
        and within 5% (``all2``, ``all5``), and whether the fit locks on
        (``ok``).
    '''
    crops = [simulate(case, condition) for case in cases]
    start = time.perf_counter()
    estimates = estimate([crop.data for crop in crops])
    seconds = (time.perf_counter() - start) / max(len(cases), 1)
    values = estimates[['z_p', 'a_p', 'n_p']].to_numpy(dtype=float)
    rows = [dict(vars(case), condition=condition, size=crop.data.shape[0],
                 z_est=z, a_est=a, n_est=n, est_seconds=seconds)
            for case, crop, (z, a, n) in zip(cases, crops, values,
                                             strict=True)]
    jobs = [Job(case, crop, (row['z_est'], row['a_est'], row['n_est']))
            for case, crop, row in zip(cases, crops, rows, strict=True)]
    if nprocs > 1:
        with ProcessPoolExecutor(nprocs) as pool:
            fits = list(pool.map(refine, jobs, chunksize=2))
    else:
        fits = [refine(job) for job in jobs]
    for row, (a_fit, n_fit, redchi) in zip(rows, fits, strict=True):
        row.update(a_fit=a_fit, n_fit=n_fit, redchi=redchi)
    return score(pd.DataFrame(rows))


def score(df: pd.DataFrame) -> pd.DataFrame:
    '''Adds the relative errors of the estimates and whether the fit locked on
    '''
    df = df.copy()
    for key, truth in (('z', 'z_p'), ('a', 'a_p'), ('n', 'n_p')):
        df[key + '_rel'] = (df[key + '_est'] - df[truth]).abs() / df[truth]
    rel = df[['z_rel', 'a_rel', 'n_rel']]
    df['all2'] = (rel < 0.02).all(axis=1)
    df['all5'] = (rel < 0.05).all(axis=1)
    df['ok'] = [locks_on(fit, truth)
                for fit, truth in zip(df.a_fit, df.a_p, strict=True)]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    '''One column for each condition: the errors of the estimates and the
    fraction of the fits that lock on'''
    return df.groupby('condition').agg(
        crop=('size', 'median'),
        z_med=('z_rel', 'median'), a_med=('a_rel', 'median'),
        n_med=('n_rel', 'median'), within2=('all2', 'mean'),
        within5=('all5', 'mean'), locks_on=('ok', 'mean'),
        ms=('est_seconds', lambda s: 1000 * s.mean())).round(3).T


def by_radius(df: pd.DataFrame) -> pd.DataFrame:
    '''The fraction of the fits that lock on, for each radius [um]'''
    return df.pivot_table(index='a_p', columns='condition', values='ok',
                          aggfunc='mean').round(2)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ncases', type=int, default=180)
    parser.add_argument('--nprocs', type=int, default=4)
    parser.add_argument('--model', default=None,
                        help='name of the Estimator model (default: the '
                        'default model)')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                        default=list(CONDITIONS))
    parser.add_argument('--output', help='write the results to this CSV')
    args = parser.parse_args(argv)

    from CATCH import Estimator
    estimator = Estimator(model=args.model)
    cases = make_cases(args.ncases)
    results = pd.concat([evaluate(estimator, cases, condition, args.nprocs)
                         for condition in args.conditions])
    if args.output:
        results.to_csv(args.output, index=False)
    print(summarize(results).to_string())
    print(by_radius(results).to_string())


if __name__ == '__main__':  # pragma: no cover
    main()
