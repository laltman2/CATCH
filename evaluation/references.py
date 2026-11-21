'''Starting points that need no network, as references

pylorenzmie estimates the properties of a particle from its hologram without
a trained network: an analytic estimator (the azimuthal profile; it knows
nothing of the refractive index, which it leaves at a fixed value) and a
global search by differential evolution.  A network has to do better than
these, in the fraction of fits that lock on after it and in the time, to be
worth its training.  They have the interface of ``CATCH.Estimator``: they
take a list of crops and return a DataFrame of ``z_p``, ``a_p`` and ``n_p``,
so ``pipelines.crops_on_truth`` and the harness use them as they are.
'''
from collections.abc import Callable

import numpy as np
import pandas as pd

from CATCH.evaluation.cases import INSTRUMENT

Estimate = Callable[[list[np.ndarray]], pd.DataFrame]

NAMES = ('analytic', 'de')


def _model(hologram):
    from pylorenzmie.theory import LorenzMie
    model = LorenzMie()
    model.parallel = False
    model.instrument.properties = dict(INSTRUMENT)
    model.coordinates = hologram.flat_coordinates
    return model


def pylorenzmie_estimator(name: str, n_p: float = 1.5,
                          seed: int = 0) -> Estimate:
    '''An estimator of pylorenzmie, for the crops of the harness

    Parameters
    ----------
    name : ``'analytic'`` or ``'de'``
        ``'analytic'``: ``pylorenzmie.analysis.Estimator``, which finds the
        height and the radius from the azimuthal profile and takes the
        index to be ``n_p``.  ``'de'``: ``DEEstimator``, a global search
        of the three by differential evolution on a few percent of the
        pixels (seconds per particle).
    n_p : float
        The index that the analytic estimator assumes.  Default: 1.5
    seed : int
        Seed of the search of ``'de'``.  Default: 0

    Returns
    -------
    estimate : callable
        Takes a list of crops and returns a DataFrame of ``z_p``, ``a_p`` and
        ``n_p`` with a row for each.

    Raises
    ------
    ValueError
        For a name that is not one of ``NAMES``.
    '''
    if name not in NAMES:
        raise ValueError(f'unknown estimator {name!r}: choose from '
                         + ', '.join(NAMES))

    def estimate(crops: list[np.ndarray]) -> pd.DataFrame:
        from pylorenzmie.analysis import Hologram
        from pylorenzmie.analysis.DEEstimator import DEEstimator
        from pylorenzmie.analysis.Estimator import Estimator
        rows = []
        for crop in crops:
            hologram = Hologram(crop)
            model = _model(hologram)
            if name == 'analytic':
                estimator = Estimator(instrument=model.instrument, n_p=n_p)
            else:
                estimator = DEEstimator(model=model, seed=seed)
            try:
                found = estimator.estimate(hologram).to_dict()
                row = {key: float(found[key])
                       for key in ('z_p', 'a_p', 'n_p')}
            except Exception:       # an estimate that fails is a result
                row = dict.fromkeys(('z_p', 'a_p', 'n_p'), np.nan)
            rows.append(row)
        return pd.DataFrame(rows, columns=['z_p', 'a_p', 'n_p'])
    return estimate


def catch_estimator(model: str | None = None) -> Callable:
    '''The Estimator of a model of the package

    Parameters
    ----------
    model : str, optional
        Name of a model of ``cfg_estimator``.  Default: the shipped one.

    Returns
    -------
    estimator : callable
        Takes the image and the centers ``x``, ``y`` [pixels] and returns
        a DataFrame of ``z_p``, ``a_p`` and ``n_p`` (the best starts), for
        ``pipelines.estimator_on_truth``.
    '''
    from CATCH import Estimator
    return Estimator(model=model)
