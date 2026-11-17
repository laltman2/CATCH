'''Training loop of the Estimator, on data made on the fly

The crops come from ``synth.Stream`` (frames with edges, corners and
neighbors, crops made as ``CATCHapp`` makes them), made by worker processes
while the network trains.  The run is measured for what it costs: the time
and the number of examples at checkpoints that are spaced evenly in the
logarithm (a network that is half as good after a tenth of the training is
the comparison that matters, so the early part is sampled densely), with
``evaluation.cost.TrainingLog``.  At each checkpoint the weights are saved
and the network is scored on a fixed set of validation crops (the seed of
which no training stream uses), and optionally by the fits that start from
its estimates (``cost.validate``).

A run writes to its directory ``configuration.json`` (what ``Estimator``
needs, with ``config=``), ``examples<N>.pt`` for each checkpoint, ``log.csv``
and ``log.json``.

Usage::

    PYLORENZMIE_BACKEND=torch python -m CATCH.training.loop
        --output runs/first --examples 2000000 [--workers 4]
        [--mode pad|shift] [--repeats 3] [--lock-cases 60]
        [--init runs/first/examples2000000.pt --seed 2]

The backend of the frames is that of the environment: the ``torch`` one is
10 times faster than ``numba`` on a GPU (the validation set alone takes
three minutes without it).  ``Settings.backend`` sets it for the workers.
'''
import argparse
import json
import math
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from CATCH.evaluation.cases import INSTRUMENT
from CATCH.evaluation.cost import TrainingLog, validate
from CATCH.ParamScale import ParamScale
from CATCH.TorchEstimator import TorchEstimator
from CATCH.training.synth import CropPolicy, Distribution, Samples, Stream

#: seed of the validation crops; training streams use ``Settings.seed`` and
#: its multiples
VALIDATION_SEED = 4242
#: the relative errors that make an estimate a good start for a fit
Z_GOOD, A_GOOD = 0.1, 0.05
#: a particle is near an edge if its center is within this of it [pixels]
EDGE = 100.


@dataclass(frozen=True)
class Settings:
    '''What a run is

    Attributes
    ----------
    examples : int
        Total number of crops to train on.  Default: 1,000,000
    batch_size : int
        Default: 64
    scaling : str
        ``'linear'``, or ``'log'``: the network predicts ``ln a_p`` and
        ``ln(n_p - n_m)``, so that relative errors weigh the same at every
        size.  Default: ``'linear'``
    lr : float
        Learning rate (Adam), decayed to zero by a cosine over the run if
        ``decay``.  Default: 1e-3
    decay : bool
        Default: True
    seed : int
        The training data are streams of seeds ``1000 * seed + worker``.
        Default: 1
    workers : int
        Processes that make the crops; 0 makes them in the training process.
        Default: 4
    device : str
        ``'auto'`` (cuda, mps or cpu), or a name.  Default: ``'auto'``
    backend : str or None
        ``PYLORENZMIE_BACKEND`` of the workers (the default is the
        environment's).  Default: None
    first : int
        Examples at the first checkpoint.  Default: 1000
    per_decade : int
        Checkpoints per factor of ten of examples.  Default: 4
    nvalidation : int
        Number of validation crops.  Default: 2000
    lock_cases : int
        Number of single-particle cases that the fits of a checkpoint start
        from the estimates of (``cost.validate``); 0 for none.  Default: 0
    nprocs : int
        Processes for those fits.  Default: 1
    init : str or None
        A checkpoint (``examples<N>.pt``) to start from, in place of new
        weights: the run resumes the training of that network with a new
        schedule of the learning rate (and a new optimizer).  Its
        ``configuration.json`` next to it must have the same scaling and
        ranges.  Use another ``seed`` than the run it continues, so that the
        data are new.  The examples of the run count from zero.
        Default: None
    '''
    examples: int = 1_000_000
    batch_size: int = 64
    scaling: str = 'linear'
    lr: float = 1e-3
    decay: bool = True
    seed: int = 1
    workers: int = 4
    device: str = 'auto'
    backend: str | None = None
    first: int = 1000
    per_decade: int = 4
    nvalidation: int = 2000
    lock_cases: int = 0
    nprocs: int = 1
    init: str | None = None


def pick_device(name: str = 'auto') -> torch.device:
    '''The device of a run: ``'auto'`` is cuda, then mps, then the cpu'''
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def schedule(total: int, first: int = 1000, per_decade: int = 4
             ) -> list[int]:
    '''Numbers of examples at which to checkpoint, evenly spaced in the log

    Parameters
    ----------
    total : int
        The last one.
    first : int
        The first one.  Default: 1000
    per_decade : int
        How many for each factor of ten.  Default: 4

    Returns
    -------
    marks : list of int
        Increasing, ending at ``total``.
    '''
    if first >= total:
        return [total]
    n = math.ceil(per_decade * math.log10(total / first))
    marks = {int(round(first * (total / first) ** (k / n)))
             for k in range(n + 1)}
    return sorted(marks | {total})


def network_config(dist: Distribution,
                   scaling: str = 'linear') -> dict[str, Any]:
    '''What ``Estimator(config=...)`` needs for a network trained on ``dist``

    The ranges of ``z_p``, ``a_p`` and ``n_p`` are those of the
    distribution; the network outputs are scaled into them (by the
    logarithm of ``a_p`` and of ``n_p - n_m`` if ``scaling`` is ``'log'``).
    '''
    if scaling not in ('linear', 'log'):
        raise ValueError(f"scaling must be 'linear' or 'log', not "
                         f'{scaling!r}')
    return dict(scaling={} if scaling == 'linear' else
                dict(a_p='log', n_p='log_contrast'),
                instrument=dict(wavelength=INSTRUMENT['wavelength'],
                                magnification=dist.magnification,
                                n_m=dist.n_m),
                particle=dict(z_p=list(dist.z_p), a_p=list(dist.a_p),
                              n_p=list(dist.n_p)),
                shape=[201, 201])


class Scaler:
    '''Scales ``(z_p, a_p, n_p)`` into [0, 1] as the config says

    The scaling is that of ``ParamScale``, which ``Estimator`` undoes.
    '''

    def __init__(self, config: dict[str, Any],
                 device: torch.device | str = 'cpu') -> None:
        self.scale = ParamScale(config)
        self.n_m = config.get('instrument', {}).get('n_m', 0.)
        self.device = torch.device(device)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.scale.normalize(list(values.unbind(1))), 1)

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.scale.unnormalize(list(values.unbind(1))), 1)


class CropStream(IterableDataset):
    '''Batches of crops, forever, different in each worker

    Yields ``(images, scale, targets)``; use ``DataLoader(batch_size=None)``.
    '''

    def __init__(self, seed: int, batch_size: int, dist: Distribution,
                 policy: CropPolicy, backend: str | None = None) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.dist = dist
        self.policy = policy
        self.backend = backend

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        worker = 0 if info is None else info.id
        stream = Stream(1000 * self.seed + worker, self.dist, self.policy)
        for batch in stream.crops(self.batch_size):
            yield batch.images, batch.scale, batch.targets


def _init_worker(_: int) -> None:  # pragma: no cover
    '''One thread for each worker: the workers are the parallelism'''
    torch.set_num_threads(1)
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


def validation_set(dist: Distribution, policy: CropPolicy,
                   n: int) -> Samples:
    '''The fixed crops that every checkpoint is scored on'''
    return next(Stream(VALIDATION_SEED, dist, policy).crops(n))


def score(net: nn.Module, samples: Samples, scaler: Scaler,
          device: torch.device, batch_size: int = 256) -> dict[str, float]:
    '''Quality of a network on crops

    Returns
    -------
    metrics : dict
        ``val_loss`` (smooth L1 of the scaled outputs, as the training
        loss), the median relative errors ``z_med``, ``a_med`` and
        ``n_med`` (of ``n_p``), ``c_med`` (of ``n_p - n_m``), ``good``,
        the fraction of the crops with ``z_p`` within 10% and ``a_p`` within
        5% (the estimates from which a fit tends to lock on), and
        ``good_edge`` and ``good_interior`` of it for the crops of particles
        within ``EDGE`` pixels of an edge and elsewhere.
    '''
    was_training = net.training
    net.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(samples.targets), batch_size):
            part = slice(i, i + batch_size)
            outputs.append(net(samples.images[part].to(device).float(),
                               samples.scale[part].to(device)).cpu())
    net.train(was_training)
    predicted = scaler.unnormalize(torch.cat(outputs).to(scaler.device))
    truth = samples.targets.to(scaler.device)
    loss = nn.functional.smooth_l1_loss(scaler.normalize(predicted),
                                        scaler.normalize(truth))
    relative = ((predicted - truth) / truth).abs().cpu().numpy()
    contrast = ((predicted[:, 2] - truth[:, 2]).abs()
                / (truth[:, 2] - scaler.n_m)).cpu().numpy()
    good = (relative[:, 0] < Z_GOOD) & (relative[:, 1] < A_GOOD)
    near = np.asarray(samples.meta['edge']) < EDGE
    return dict(val_loss=float(loss),
                z_med=float(np.median(relative[:, 0])),
                a_med=float(np.median(relative[:, 1])),
                n_med=float(np.median(relative[:, 2])),
                c_med=float(np.median(contrast)),
                good=float(good.mean()),
                good_edge=float(good[near].mean()) if near.any()
                else float('nan'),
                good_interior=float(good[~near].mean()) if (~near).any()
                else float('nan'))


def load_weights(net: nn.Module, path: str | Path,
                 config: dict[str, Any]) -> None:
    '''Loads a checkpoint into a network, if it was trained for ``config``

    Raises
    ------
    ValueError
        If the ``configuration.json`` next to the checkpoint differs from
        ``config`` in the scaling or the ranges of the parameters.
    '''
    path = Path(path)
    saved = path.parent / 'configuration.json'
    if saved.exists():
        before = json.loads(saved.read_text())
        for key in ('scaling', 'particle'):
            if before.get(key, {}) != config.get(key, {}):
                raise ValueError(f'{path} was trained with a different '
                                 f'{key}: {before.get(key)} against '
                                 f'{config.get(key)}')
    net.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])


def train(settings: Settings, output: str | Path,
          dist: Distribution | None = None,
          policy: CropPolicy | None = None,
          net: nn.Module | None = None) -> TrainingLog:
    '''Trains an Estimator network on synthetic crops

    Parameters
    ----------
    settings : Settings
    output : path
        Directory of the run (made if it is not there).
    dist : Distribution, optional
        The frames.  Default: ``Distribution()``
    policy : CropPolicy, optional
        The crops.  Default: ``CropPolicy()``
    net : torch.nn.Module, optional
        The network, which takes ``(images, scale)`` and returns three
        numbers scaled into [0, 1].  Default: a new ``TorchEstimator``

    Returns
    -------
    log : TrainingLog
        Time, examples, training loss and the scores at each checkpoint.
    '''
    dist = dist or Distribution()
    policy = policy or CropPolicy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = network_config(dist, settings.scaling)
    (output / 'configuration.json').write_text(json.dumps(config))
    if settings.backend:
        os.environ['PYLORENZMIE_BACKEND'] = settings.backend
    device = pick_device(settings.device)
    net = net or TorchEstimator()
    if settings.init:
        load_weights(net, settings.init, config)
    net = net.to(device)
    scaler = Scaler(config, device)
    validation = validation_set(dist, policy, settings.nvalidation)
    marks = schedule(settings.examples, settings.first, settings.per_decade)
    steps = math.ceil(settings.examples / settings.batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr)
    decay = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
             if settings.decay else None)
    dataset = CropStream(settings.seed, settings.batch_size, dist, policy,
                         settings.backend)
    loader = DataLoader(
        dataset, batch_size=None, num_workers=settings.workers,
        worker_init_fn=_init_worker if settings.workers else None,
        prefetch_factor=4 if settings.workers else None,
        persistent_workers=bool(settings.workers))
    log = TrainingLog(settings=asdict(settings), policy=asdict(policy),
                      distribution=asdict(dist), device=str(device))
    net.train()
    examples = 0
    running, count = 0., 0
    pending = list(marks)
    for batch in loader:
        images, scale, targets = (t.to(device) for t in batch)
        optimizer.zero_grad()
        loss = nn.functional.smooth_l1_loss(
            net(images.float(), scale), scaler.normalize(targets))
        loss.backward()
        optimizer.step()
        if decay is not None:
            decay.step()
        examples += len(targets)
        running += loss.item()
        count += 1
        if examples >= pending[0]:
            while pending and examples >= pending[0]:
                pending.pop(0)
            metrics = dict(train_loss=running / count)
            running, count = 0., 0
            with log.pause():
                metrics.update(checkpoint(net, output, examples, config,
                                          validation, scaler, device,
                                          settings))
            log.log(epoch=examples // settings.batch_size,
                    examples=examples, **metrics)
            log.save(output / 'log.csv')
        if not pending:
            break
    return log


def checkpoint(net: nn.Module, output: Path, examples: int,
               config: dict[str, Any], validation: Samples, scaler: Scaler,
               device: torch.device, settings: Settings) -> dict[str, float]:
    '''Saves the weights and scores the network'''
    path = output / f'examples{examples}.pt'
    torch.save({'state_dict': {k: v.cpu()
                               for k, v in net.state_dict().items()}}, path)
    metrics = score(net, validation, scaler, device)
    if settings.lock_cases:
        from CATCH.Estimator import Estimator
        metrics.update(validate(Estimator(weights=path, config=config),
                                settings.lock_cases, settings.nprocs))
    return metrics


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--examples', type=int, default=Settings.examples)
    parser.add_argument('--batch-size', type=int,
                        default=Settings.batch_size)
    parser.add_argument('--scaling', choices=('linear', 'log'),
                        default=Settings.scaling)
    parser.add_argument('--lr', type=float, default=Settings.lr)
    parser.add_argument('--seed', type=int, default=Settings.seed)
    parser.add_argument('--workers', type=int, default=Settings.workers)
    parser.add_argument('--device', default=Settings.device)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--first', type=int, default=Settings.first)
    parser.add_argument('--per-decade', type=int,
                        default=Settings.per_decade)
    parser.add_argument('--lock-cases', type=int,
                        default=Settings.lock_cases)
    parser.add_argument('--nprocs', type=int, default=Settings.nprocs)
    parser.add_argument('--init', default=None,
                        help='checkpoint to resume from')
    parser.add_argument('--mode', choices=('pad', 'shift'), default='pad')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    settings = replace(
        Settings(), examples=args.examples, batch_size=args.batch_size,
        scaling=args.scaling, lr=args.lr, seed=args.seed, workers=args.workers,
        device=args.device, backend=args.backend, first=args.first,
        per_decade=args.per_decade, lock_cases=args.lock_cases,
        nprocs=args.nprocs, init=args.init)
    log = train(settings, args.output, policy=CropPolicy(mode=args.mode,
                                            repeats=args.repeats))
    print(log.frame().to_string())


if __name__ == '__main__':  # pragma: no cover
    main()
