'''Training cost: what it takes to reach a given quality

A better network that takes as long to train is not an improvement in what
this project is after.  The cost is measured as the **time and the number of
examples that a training run needs to reach a target**: of the lock-on rate
of the fit (``validate``), or of the loss.  The time is in seconds of the
clock on a stated machine (``describe_hardware``), without the time spent
on validation, and the examples are the numbers of images the network has
been trained on, which does not depend on the machine.

``TrainingLog`` records a run, ``cost_to_reach`` reads the cost from it, and
``evaluate_checkpoints`` measures the lock-on rate of the checkpoints that a
run saved, so that a run that did not validate can still be compared.
'''
import json
import os
import platform
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def describe_hardware() -> dict[str, Any]:
    '''The machine and the libraries, to say what a time was measured on'''
    import torch
    info: dict[str, Any] = dict(
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), cpus=os.cpu_count(),
        python=platform.python_version(), torch=torch.__version__,
        cuda=torch.cuda.is_available(), devices=[])
    if torch.cuda.is_available():
        info['devices'] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        info['devices'] = ['mps']
    return info


class TrainingLog:
    '''A record of a training run: time, examples and metrics

    Parameters
    ----------
    **settings
        What the run was: the configuration, a name, anything that is needed
        to repeat it.  They are saved with the log.

    Notes
    -----
    The clock starts when the log is made.  Time spent in ``pause()`` is
    not counted, so that validating does not cost the run.
    '''

    def __init__(self, **settings: Any) -> None:
        self.settings = dict(settings)
        self.hardware = describe_hardware()
        self._rows: list[dict[str, Any]] = []
        self._start = time.perf_counter()
        self._paused = 0.

    @property
    def seconds(self) -> float:
        '''Time of the run so far, without the pauses [seconds]'''
        return time.perf_counter() - self._start - self._paused

    @contextmanager
    def pause(self) -> Iterator[None]:
        '''Leaves the time of a block, such as a validation, out of the run'''
        start = time.perf_counter()
        try:
            yield
        finally:
            self._paused += time.perf_counter() - start

    def log(self, epoch: int, examples: int, **metrics: float) -> None:
        '''Records the state of the run

        Parameters
        ----------
        epoch : int
            Number of epochs completed.
        examples : int
            Number of images the network has been trained on, counting each
            time that it sees an image.
        **metrics
            The losses and the quality of the network: ``lock_on`` and so on.
        '''
        self._rows.append(dict(seconds=self.seconds, epoch=epoch,
                               examples=examples, **metrics))

    def frame(self) -> pd.DataFrame:
        '''The record, a row for each time that it was logged'''
        return pd.DataFrame(self._rows)

    def save(self, path: str | Path) -> None:
        '''Writes the record to a CSV file, and to a JSON file next to it
        the settings and the machine'''
        path = Path(path)
        self.frame().to_csv(path, index=False)
        path.with_suffix('.json').write_text(json.dumps(
            dict(settings=self.settings, hardware=self.hardware), indent=2,
            default=str))

    @classmethod
    def load(cls, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        '''Reads what ``save`` wrote: the record, and the settings and the
        hardware'''
        path = Path(path)
        info = json.loads(path.with_suffix('.json').read_text())
        return pd.read_csv(path), info


def cost_to_reach(record: pd.DataFrame, metric: str, target: float,
                  lower_is_better: bool = False) -> dict[str, float] | None:
    '''The cost of the first time that a run reached a target

    Parameters
    ----------
    record : pandas.DataFrame
        The record of a run (``TrainingLog.frame``), with the columns
        ``seconds``, ``epoch`` and ``examples`` and the metric.  Rows
        where the metric was not measured (``nan``) are ignored.
    metric : str
        The column to look at, such as ``'lock_on'`` or ``'test_loss'``.
    target : float
        The value to reach.
    lower_is_better : bool
        True for a loss.  Default: False

    Returns
    -------
    cost : dict or None
        ``seconds``, ``epoch`` and ``examples`` at that time; None if the
        run never got there.
    '''
    values = record[metric]
    reached = values <= target if lower_is_better else values >= target
    hit = record[reached.fillna(False)]
    if hit.empty:
        return None
    first = hit.iloc[0]
    return dict(seconds=float(first['seconds']), epoch=int(first['epoch']),
                examples=int(first['examples']))


def throughput(record: pd.DataFrame) -> float:
    '''Examples per second over the whole run'''
    last = record.iloc[-1]
    return float(last['examples'] / last['seconds'])


def validate(estimate: Callable[[list], Any],
             ncases: int = 60, nprocs: int = 1,
             condition: str = 'training') -> dict[str, float]:
    '''The quality of an estimator by the fits that start from it

    Uses the single-particle cases of ``CATCH.evaluation.cases``, which take
    about a second for each fit on a core.

    Parameters
    ----------
    estimate : callable
        Takes the crops and returns a DataFrame of ``z_p``, ``a_p`` and
        ``n_p``, such as ``CATCH.Estimator``.
    ncases : int
        Number of particles.  Default: 60
    nprocs : int
        Number of processes for the fits.  Default: 1
    condition : str
        ``'training'`` or ``'fixed'``; see ``CATCH.evaluation.cases``.

    Returns
    -------
    metrics : dict
        ``lock_on``, the fraction of the fits that lock on, ``within5``,
        the fraction of the particles for which all the estimates are within
        5%, and the median relative errors ``z_med``, ``a_med`` and
        ``n_med``.
    '''
    from CATCH.evaluation.cases import Condition, make_cases
    from CATCH.evaluation.run import evaluate
    from typing import cast
    results = evaluate(estimate, make_cases(ncases),
                       cast(Condition, condition), nprocs)
    def column(name: str) -> pd.Series:
        return pd.Series(results[name])

    return dict(lock_on=float(column('ok').mean()),
                within5=float(column('all5').mean()),
                z_med=float(column('z_rel').median()),
                a_med=float(column('a_rel').median()),
                n_med=float(column('n_rel').median()))


def epoch_of(path: str | Path) -> int:
    '''The number of epochs in the name of a checkpoint, ``epoch300.pt``'''
    match = re.search(r'epoch(\d+)', Path(path).name)
    if match is None:
        raise ValueError(f'no epoch in the name {Path(path).name!r}')
    return int(match.group(1))


def evaluate_checkpoints(paths: Sequence[str | Path],
                         make_estimate: Callable[[Path], Callable],
                         examples_per_epoch: int,
                         ncases: int = 60, nprocs: int = 1,
                         extra: Mapping[str, Any] | None = None
                         ) -> pd.DataFrame:
    '''The lock-on rate of the checkpoints that a training run saved

    Parameters
    ----------
    paths : sequence of path
        Checkpoint files named ``epoch<N>.pt``.
    make_estimate : callable
        Takes the path of a checkpoint and returns the estimator that it
        defines: ``lambda path: Estimator(weights=path, config=config)``.
    examples_per_epoch : int
        Number of images in an epoch.
    ncases, nprocs
        As for ``validate``.
    extra : mapping
        Constant columns for the result, such as the name of the run.

    Returns
    -------
    record : pandas.DataFrame
        A row for each checkpoint, in order of the epoch: ``epoch``,
        ``examples`` and the metrics of ``validate``.  (``seconds`` is not
        known from a checkpoint; take it from the log of the run.)
    '''
    rows = []
    for path in sorted(paths, key=epoch_of):
        epoch = epoch_of(path)
        rows.append(dict(epoch=epoch, examples=epoch * examples_per_epoch,
                         **validate(make_estimate(Path(path)), ncases,
                                    nprocs), **(extra or {})))
    return pd.DataFrame(rows)
