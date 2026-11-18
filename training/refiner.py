'''Training of the Refiner, on data made on the fly

An example is a crop of a synthetic frame (with its edges, corners and
neighbors, made as ``synth.make_crops`` makes them), an estimate that is
off from the truth by a random amount, and the model of that estimate
rendered on the grid of the crop.  The network learns the correction that
takes the estimate to the truth.  Several estimates (``Settings.guesses``)
are drawn for each crop, because the frame is the expensive part.

The errors of the estimates are drawn from a wide range (0.2% to 12% by
default, log-uniform) so that one network serves for the first step, which
has to cross a few percent, and the last, which has to find a few tenths of
a percent.  Half of the errors are along the direction in which ``z_p`` and
``a_p`` change together, which is where the errors of the Estimator lie.

A run writes to its directory ``examples<N>.pt`` for each checkpoint (the
averaged weights as ``state_dict``, the raw ones as ``raw``),
``configuration.json``, ``log.csv`` and ``log.json``.  ``CATCH.Refiner``
loads the checkpoints.

Usage::

    PYLORENZMIE_BACKEND=torch python -m CATCH.training.refiner
        --output runs/refiner --examples 2000000 [--workers 4]
'''
import argparse
import json
import math
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from CATCH.evaluation.cost import TrainingLog
from CATCH.Refiner import (CLEARANCE, N, N_M, PIXEL, UNITS, RefinerNet,
                           Renderer, crop, refine_states, resize,
                           resize_mask)
from CATCH.training.loop import (Average, _init_worker,
                                 learning_rate_factor, pick_device,
                                 schedule)
from CATCH.training.synth import (CropPolicy, Distribution, Extent, Frame,
                                  Renderer as FrameRenderer, make_frame)

#: seed of the validation examples (training streams use other seeds)
VALIDATION_SEED = 4243
#: the frames of the Refiner: only the particles that it is for
DISTRIBUTION = Distribution(a_p=(1.5, 5.), n_p=(1.42, 1.75),
                            nparticles=(1, 10))
#: the crops: padded and shifted, as the pipelines make them
POLICY = CropPolicy(mode='mix')
#: weights of the outputs in the loss
OUTPUT_WEIGHT = torch.tensor([1., 1., 0.4, 0.3, 0.3])


@dataclass(frozen=True)
class Settings:
    '''What a run is

    Attributes
    ----------
    examples : int
        Total number of examples.  Default: 1,000,000
    batch_size : int
        Default: 128
    guesses : int
        Estimates drawn for each crop.  Default: 4
    sigma : tuple of float
        Range of the error of the estimates (a fraction, log-uniform); the
        error of the contrast is 2.5 times as large.  Default: (0.002, 0.12)
    lr : float
        Learning rate (Adam), decayed to zero by a cosine.  Default: 1e-3
    warmup : int
        Steps of the rise of the learning rate.  Default: 200
    clip : float
        Largest norm of the gradient.  Default: 1
    seed : int
        The training data are streams of seeds ``1000 * seed + worker``.
        Default: 1
    workers : int
        Processes that make the examples.  Default: 4
    device : str
        ``'auto'`` or a name.  Default: ``'auto'``
    backend : str or None
        ``PYLORENZMIE_BACKEND`` of the workers.  Default: None
    first, per_decade : int
        The checkpoints: the first, and the number for a factor of ten.
        Default: 5000 and 3
    nvalidation : int
        Examples of the validation set (a fixed error of 4%).  Default: 400
    steps : int
        Applications of the network in the scores.  Default: 3
    ema : float
        Decay of the running average of the weights.  Default: 0.999
    init : str or None
        A checkpoint to start from.  Default: None
    resume : bool
        Continue a run that stopped (the GPU of a laptop kills long jobs
        with ``Impacting Interactivity``): the state is saved at every
        checkpoint and every ``save_every`` steps, and a run that finds
        ``state.pt`` in its directory continues from it, on new data.
        Default: False
    save_every : int
        Steps between saves of the state.  Default: 100
    '''
    examples: int = 1_000_000
    batch_size: int = 128
    guesses: int = 4
    sigma: tuple[float, float] = (0.002, 0.12)
    lr: float = 1e-3
    warmup: int = 200
    clip: float = 1.
    seed: int = 1
    workers: int = 4
    device: str = 'auto'
    backend: str | None = None
    first: int = 5000
    per_decade: int = 3
    nvalidation: int = 400
    steps: int = 3
    ema: float = 0.999
    init: str | None = None
    resume: bool = False
    save_every: int = 100
    #: the distribution of the frames and the crops, for the log
    distribution: Distribution = field(default=DISTRIBUTION)
    policy: CropPolicy = field(default=POLICY)


@dataclass(frozen=True)
class Examples:
    '''A batch of examples

    Attributes
    ----------
    data, guess, mask : torch.Tensor
        Shape ``(B, 201, 201)``: the hologram, the model at the estimate,
        and the pixels that are in the image.
    scale : torch.Tensor
        Shape ``(B,)``: the side of the crop over 201.
    target : torch.Tensor
        Shape ``(B, 5)``: the correction in the units of the network.
    sigma : torch.Tensor
        Shape ``(B,)``: the size of the error of the estimate.
    states, truth : numpy.ndarray
        Shape ``(B, 5)``: ``(z_p, a_p, n_p - n_m, x, y)`` of the estimate
        and of the particle (offsets from the center of the crop).
    sides : numpy.ndarray
        Sides of the crops [pixels].
    meta : pandas.DataFrame
        ``edge`` and ``neighbor`` of the particle.
    '''
    data: torch.Tensor
    guess: torch.Tensor
    mask: torch.Tensor
    scale: torch.Tensor
    target: torch.Tensor
    sigma: torch.Tensor
    states: np.ndarray
    truth: np.ndarray
    sides: np.ndarray
    meta: pd.DataFrame


def perturbation(rng: np.random.Generator, sigma: float) -> np.ndarray:
    '''Relative errors of ``(z_p, a_p, contrast)`` of an estimate

    Half of them are independent, and half are mostly along the direction in
    which ``z_p`` and ``a_p`` change together.  The error of the contrast is
    2.5 times as large.
    '''
    scale = np.array([1., 1., 2.5]) * sigma
    if rng.random() < 0.5:
        return rng.normal(size=3) * scale
    t = rng.normal() * sigma
    return np.array([t, t, 0.]) + rng.normal(size=3) * scale \
        * np.array([.25, .25, 1.])


def make_examples(frame: Frame, rng: np.random.Generator,
                  renderer: Renderer,
                  policy: CropPolicy = POLICY, guesses: int = 4,
                  sigma: tuple[float, float] = (0.002, 0.12),
                  fixed: float | None = None) -> Examples | None:
    '''Examples from the particles of a frame that are in it

    Parameters
    ----------
    frame : Frame
    rng : numpy.random.Generator
    renderer : Renderer
        Renders the estimates.
    policy : CropPolicy
        How the crops are made (the side, the error of the center, padded
        or shifted).
    guesses : int
        Estimates of each crop.  Default: 4
    sigma : tuple of float
        The error of an estimate is log-uniform in this range.
    fixed : float, optional
        A fixed error, in place of the range.

    Returns
    -------
    examples : Examples or None
        None if no particle is inside the frame.
    '''
    data, guess, mask, scale, target, sig = [], [], [], [], [], []
    states, truths, sides, meta = [], [], [], []
    inside = frame.particles[frame.particles.inside]
    for _, p in inside.iterrows():
        for _ in range(policy.repeats):
            side = int(2. * p.extent * (1. + policy.side_noise
                                        * rng.normal()))
            side = max(side, policy.minimum)
            spread = policy.tail_scale if rng.random() < policy.tail \
                else policy.jitter
            cx = p.x_p + spread * rng.normal()
            cy = p.y_p + spread * rng.normal()
            pad = (rng.random() < 0.5 if policy.mode == 'mix'
                   else policy.mode == 'pad')
            patch, valid, ox, oy = crop(frame.image, cx, cy, side, pad)
            side = patch.shape[0]
            small, valid_small = resize(patch), resize_mask(valid)
            truth = np.array([p.z_p, p.a_p, p.n_p - N_M,
                              ox + p.x_p - cx, oy + p.y_p - cy])
            for _ in range(guesses):
                s = fixed if fixed is not None else float(np.exp(
                    rng.uniform(np.log(sigma[0]), np.log(sigma[1]))))
                d = perturbation(rng, s)
                state = truth.copy()
                state[:3] *= np.exp(-d)
                state[0] = max(state[0], CLEARANCE * state[1] / PIXEL)
                state[3:] = (ox, oy)
                image = renderer(side, state)
                if not np.isfinite(image).all():
                    continue
                data.append(small)
                mask.append(valid_small)
                guess.append(torch.as_tensor(image, dtype=torch.float32))
                scale.append(side / N)
                correction = np.empty(5)
                correction[:3] = np.log(truth[:3] / state[:3]) * 100.
                correction[3:] = (truth[3:] - state[3:]) * UNITS[3:]
                target.append(correction)
                sig.append(s)
                states.append(state)
                truths.append(truth)
                sides.append(side)
                meta.append(dict(edge=p.edge, neighbor=p.neighbor))
    if not data:
        return None
    return Examples(
        torch.stack(data), torch.stack(guess), torch.stack(mask),
        torch.tensor(scale, dtype=torch.float32),
        torch.tensor(np.array(target), dtype=torch.float32),
        torch.tensor(sig, dtype=torch.float32),
        np.array(states), np.array(truths), np.array(sides, dtype=float),
        pd.DataFrame(meta))


def join(parts: list[Examples]) -> Examples:
    '''One batch from several'''
    return Examples(
        torch.cat([p.data for p in parts]),
        torch.cat([p.guess for p in parts]),
        torch.cat([p.mask for p in parts]),
        torch.cat([p.scale for p in parts]),
        torch.cat([p.target for p in parts]),
        torch.cat([p.sigma for p in parts]),
        np.concatenate([p.states for p in parts]),
        np.concatenate([p.truth for p in parts]),
        np.concatenate([p.sides for p in parts]),
        pd.concat([p.meta for p in parts], ignore_index=True))


def take(e: Examples, index: np.ndarray) -> Examples:
    '''Some of the examples'''
    t = torch.as_tensor(index)
    return Examples(e.data[t], e.guess[t], e.mask[t], e.scale[t],
                    e.target[t], e.sigma[t], e.states[index],
                    e.truth[index], e.sides[index],
                    e.meta.iloc[index].reset_index(drop=True))


class ExampleStream:
    '''Batches of examples, forever, from a seed

    Parameters
    ----------
    seed : int
        Seed of the frames and the errors.
    batch_size : int
    dist : Distribution
        The frames.  Default: ``DISTRIBUTION``
    policy : CropPolicy
        Default: ``POLICY``
    guesses : int
        Default: 4
    sigma : tuple of float
        Default: (0.002, 0.12)
    fixed : float, optional
        A fixed error.
    '''

    def __init__(self, seed: int, batch_size: int,
                 dist: Distribution = DISTRIBUTION,
                 policy: CropPolicy = POLICY, guesses: int = 4,
                 sigma: tuple[float, float] = (0.002, 0.12),
                 fixed: float | None = None) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.dist = dist
        self.policy = policy
        self.guesses = guesses
        self.sigma = sigma
        self.fixed = fixed
        self.rng = np.random.default_rng(seed)
        self._frames = FrameRenderer(dist.shape)
        self._extent = Extent()
        self._renderer: Renderer | None = None

    def batches(self) -> Iterator[Examples]:
        '''Batches of ``batch_size``, shuffled across frames'''
        self._renderer = self._renderer or Renderer()
        pending: list[Examples] = []
        count = 0
        while True:
            frame = make_frame(self.rng, self.dist, self._frames,
                               self._extent)
            examples = make_examples(frame, self.rng, self._renderer,
                                     self.policy, self.guesses, self.sigma,
                                     self.fixed)
            if examples is None:
                continue
            pending.append(examples)
            count += len(examples.scale)
            while count >= 2 * self.batch_size:
                batch = join(pending)
                order = self.rng.permutation(len(batch.scale))
                yield take(batch, order[:self.batch_size])
                rest = take(batch, order[self.batch_size:])
                pending = [rest]
                count = len(rest.scale)


class _Dataset(IterableDataset):
    '''The stream in worker processes, each with its own seed'''

    def __init__(self, settings: Settings, offset: int = 0) -> None:
        self.settings = settings
        self.offset = offset

    def __iter__(self) -> Iterator[Examples]:
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        worker = 0 if info is None else info.id
        s = self.settings
        seed = 1000 * (s.seed + 100 * self.offset) + worker
        yield from ExampleStream(
            seed, s.batch_size, s.distribution, s.policy, s.guesses,
            s.sigma).batches()


def validation_set(settings: Settings) -> Examples:
    '''The fixed examples that every checkpoint is scored on (a 4% error)'''
    stream = ExampleStream(VALIDATION_SEED, settings.nvalidation,
                           settings.distribution, settings.policy, 1,
                           settings.sigma, fixed=0.04)
    return next(stream.batches())


def state_errors(states: np.ndarray, truth: np.ndarray) -> np.ndarray:
    '''Absolute relative errors of ``z_p``, ``a_p`` and the contrast, and
    the distance of the center [pixels]; shape ``(B, 4)``'''
    relative = np.abs(states[:, :3] / truth[:, :3] - 1.)
    distance = np.hypot(states[:, 3] - truth[:, 3],
                        states[:, 4] - truth[:, 4])
    return np.column_stack([relative, distance])


def score(net: nn.Module, examples: Examples, renderer: Renderer,
          device: torch.device, steps: int = 3) -> dict[str, float]:
    '''Quality of a network by what the refinement does to the errors

    Returns
    -------
    metrics : dict
        Median relative errors ``z_med``, ``a_med`` and ``c_med`` after
        ``steps`` steps (the errors before are ``z0_med``...);
        ``good``, the fraction within 0.3% in ``z_p`` and ``a_p`` after
        one step (``good1``) and after ``steps`` (``good``); ``good_edge``,
        the same for the particles within 100 pixels of an edge.
    '''
    was = net.training
    net.eval()
    try:
        first = refine_states(net, examples.data, examples.mask,
                              examples.sides, examples.states, renderer, 1,
                              device)
        last = refine_states(net, examples.data, examples.mask,
                             examples.sides, examples.states, renderer,
                             steps, device)
    finally:
        net.train(was)
    before = state_errors(examples.states, examples.truth)
    one = state_errors(first, examples.truth)
    after = state_errors(last, examples.truth)

    def good(e: np.ndarray) -> np.ndarray:
        return (e[:, 0] < 0.003) & (e[:, 1] < 0.003)
    near = np.asarray(examples.meta['edge']) < 100.
    return dict(z0_med=float(np.median(before[:, 0])),
                a0_med=float(np.median(before[:, 1])),
                z_med=float(np.median(after[:, 0])),
                a_med=float(np.median(after[:, 1])),
                c_med=float(np.median(after[:, 2])),
                xy_med=float(np.median(after[:, 3])),
                good1=float(good(one).mean()),
                good=float(good(after).mean()),
                good_edge=float(good(after)[near].mean()) if near.any()
                else float('nan'))


def loss_of(out: torch.Tensor, batch: Examples,
            device: torch.device) -> torch.Tensor:
    '''L1 of the corrections, weighted by output and by the inverse of the
    size of the error (so that the small errors count: they are the ones
    the last steps have to find)'''
    weight = 1. / (batch.sigma.to(device) * 100. + 0.3)
    error = (out - batch.target.to(device)).abs() \
        * OUTPUT_WEIGHT.to(device)
    return (error * weight[:, None]).mean()


def train(settings: Settings, output: str | Path) -> TrainingLog:
    '''Trains a Refiner network on synthetic examples

    Parameters
    ----------
    settings : Settings
    output : path
        Directory of the run (made if it is not there).

    Returns
    -------
    log : TrainingLog
    '''
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configuration.json').write_text(json.dumps(
        dict(settings=asdict(settings))))
    if settings.backend:
        os.environ['PYLORENZMIE_BACKEND'] = settings.backend
    device = pick_device(settings.device)
    net = RefinerNet()
    if settings.init:
        net.load_state_dict(torch.load(settings.init,
                                       map_location='cpu')['state_dict'])
    net = net.to(device)
    renderer = Renderer()
    validation = validation_set(settings)
    marks = schedule(settings.examples, settings.first, settings.per_decade)
    steps = math.ceil(settings.examples / settings.batch_size)
    average = Average(net, settings.ema)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr)
    decay = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda k: learning_rate_factor(k, steps, settings.warmup))
    state_path = output / 'state.pt'
    examples, resumes, rows, seconds = 0, 0, [], 0.
    if settings.resume and state_path.exists():
        saved = torch.load(state_path, map_location='cpu',
                           weights_only=False)
        net.load_state_dict(saved['net'])
        average.module.load_state_dict(saved['average'])
        average.steps = saved['average_steps']
        optimizer.load_state_dict(saved['optimizer'])
        decay.load_state_dict(saved['decay'])
        examples, rows, seconds = (saved['examples'], saved['rows'],
                                   saved['seconds'])
        resumes = saved['resumes'] + 1
    loader = DataLoader(
        _Dataset(settings, resumes), batch_size=None,
        num_workers=settings.workers,
        worker_init_fn=_init_worker if settings.workers else None,
        prefetch_factor=2 if settings.workers else None,
        persistent_workers=bool(settings.workers))
    log = TrainingLog(settings=asdict(settings), device=str(device))
    log._rows = list(rows)
    log._start -= seconds
    net.train()
    running, count = 0., 0
    pending = [m for m in marks if m > examples]

    def save_state() -> None:
        torch.save(dict(net=net.state_dict(),
                        average=average.module.state_dict(),
                        average_steps=average.steps,
                        optimizer=optimizer.state_dict(),
                        decay=decay.state_dict(), examples=examples,
                        rows=log._rows, seconds=log.seconds,
                        resumes=resumes), state_path)

    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        out = net(batch.data.to(device), batch.guess.to(device),
                  batch.mask.to(device), batch.scale.to(device))
        loss = loss_of(out, batch, device)
        loss.backward()
        if settings.clip:
            nn.utils.clip_grad_norm_(net.parameters(), settings.clip)
        optimizer.step()
        average.update(net)
        decay.step()
        examples += len(batch.scale)
        running += loss.item()
        count += 1
        if settings.resume and step % settings.save_every == 0:
            save_state()
        if pending and examples >= pending[0]:
            while pending and examples >= pending[0]:
                pending.pop(0)
            metrics = dict(train_loss=running / count)
            running, count = 0., 0
            with log.pause():
                torch.save({'state_dict': {k: v.cpu() for k, v in
                                           average.module.state_dict()
                                           .items()},
                            'raw': {k: v.cpu() for k, v in
                                    net.state_dict().items()}},
                           output / f'examples{examples}.pt')
                metrics.update(score(average.module, validation, renderer,
                                     device, settings.steps))
            log.log(epoch=examples // settings.batch_size,
                    examples=examples, **metrics)
            log.save(output / 'log.csv')
            if settings.resume:
                save_state()
        if not pending:
            break
    return log


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--examples', type=int, default=Settings.examples)
    parser.add_argument('--batch-size', type=int,
                        default=Settings.batch_size)
    parser.add_argument('--guesses', type=int, default=Settings.guesses)
    parser.add_argument('--sigma', type=float, nargs=2,
                        default=Settings.sigma)
    parser.add_argument('--lr', type=float, default=Settings.lr)
    parser.add_argument('--seed', type=int, default=Settings.seed)
    parser.add_argument('--workers', type=int, default=Settings.workers)
    parser.add_argument('--device', default=Settings.device)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--first', type=int, default=Settings.first)
    parser.add_argument('--per-decade', type=int,
                        default=Settings.per_decade)
    parser.add_argument('--nvalidation', type=int,
                        default=Settings.nvalidation)
    parser.add_argument('--init', default=None)
    parser.add_argument('--resume', action='store_true',
                        help='continue from state.pt of the output '
                        'directory, if it is there (and save it)')
    parser.add_argument('--radius', type=float, nargs=2, default=(1.5, 5.),
                        help='range of the radius [um]')
    parser.add_argument('--index', type=float, nargs=2, default=(1.42, 1.75),
                        help='range of the refractive index')
    args = parser.parse_args()
    dist = replace(DISTRIBUTION, a_p=tuple(args.radius),
                   n_p=tuple(args.index))
    settings = replace(
        Settings(), examples=args.examples, batch_size=args.batch_size,
        guesses=args.guesses, sigma=tuple(args.sigma), lr=args.lr,
        seed=args.seed, workers=args.workers, device=args.device,
        backend=args.backend, first=args.first, per_decade=args.per_decade,
        nvalidation=args.nvalidation, init=args.init, resume=args.resume,
        distribution=dist)
    log = train(settings, args.output)
    print(log.frame().to_string())


if __name__ == '__main__':  # pragma: no cover
    main()
