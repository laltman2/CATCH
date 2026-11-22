'''Train the network of the Estimator on synthetic profiles

Usage::

    python -m CATCH.customization.training.estimator OUT --steps 40000
    python -m CATCH.customization.training.estimator OUT --steps 10000 \
        --init default

The :class:`CATCH.lib.networks.ProfileNet` learns to propose candidates for
``z_p``, ``a_p`` and ``n_p`` from a radial profile.  Its training examples
are made in two parts:

* a **library** of what measurement adds to a profile: synthetic frames
  (:mod:`CATCH.customization.training.frames`, several particles, noise,
  pixels of 1 to 2.5 reference pixels) are measured the way the Estimator
  measures them (:class:`CATCH.lib.profiles.Profiler`, with centers that
  are off by 0.1 pixel), and the model profile of each particle is
  subtracted: what remains is noise, neighbors, edges and the sampling of
  the camera;
* **clean profiles** of random particles, computed for every batch by the
  batched Lorenz-Mie model (:class:`CATCH.lib.lorenzmie.RadialModel`), to
  which random residuals of the library are added.

Some of the examples (``saturated``, half by default) are made as a camera
that saturates would record them: the rings of the profile that reach the
limit of the camera are marked as not measured, as they are when the
Estimator is given the ``saturation`` of the camera.  The limit is that of
an 8-bit camera with a background of 100 (2.55 times the background), as for
the shipped model; ``--levels LOW HIGH`` draws it for each example instead.

The loss lets each example train only its best candidate (winner takes
all), so the 64 candidates spread over the near-degenerate solutions, and
a cross-entropy teaches the logits which candidate is best.

``OUT`` receives ``weights.pt`` and ``configuration.json`` (a model that
``Estimator(OUT)`` loads), ``library.npz``, ``state.pt`` (to resume: run
the same command again) and ``eval.jsonl``: on validation profiles at
both ends of the range of scales, for particles away from edges and
neighbors and by height, the fraction with a candidate within 3% (and 1%)
of the truth.  The shipped model took 40,000 steps of 512 profiles, about
5 hours on an Apple M-series GPU, and then 10,000 steps (``--init
default --lr 3e-4``, about 75 minutes) with half of the examples as a
camera that saturates records them (``--saturated``).
'''
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from CATCH.lib.instruments import REFERENCE
from CATCH.customization.training.common import (
    checkpoint, configure, log_evaluation, restore, seal)
from CATCH.Localizer import default_device
from CATCH.lib.lorenzmie import RadialModel
from CATCH.lib.networks import CENTER, RADII, ProfileNet, profile_features
from CATCH.lib.profiles import CAMERA_LEVEL, Profiler
from CATCH.customization.training.frames import Frames, Ranges, log_uniform

PIXEL, N_M = REFERENCE['magnification'], REFERENCE['n_m']
#: ranges of the training profiles: a_p [um], n_p - n_m, z_p [um]
A_P, CONTRAST, Z_UM = (0.2, 5.5), (0.04, 1.16), (5., 320.)
#: weights of the errors of log z_p, log a_p and log(n_p - n_m)
WEIGHT = (1., 1., .4)
#: the frames of the library and of the validation
LIBRARY = Ranges(nparticles=(1, 6), a_p=(0.25, 5.), z_um=(5., 320.),
                 noise=(.015, .06), imperfect=False)
VALIDATION_SEEDS = (7500, 7501)
#: heights of the validation bands [um]
BANDS = ((10, 31), (31, 80), (80, 160), (160, 320))


def measure(frames: Frames, count: int, profiler: Profiler,
            model: RadialModel, jitter: float = .1,
            scale: float | None = None,
            rng: np.random.Generator | None = None) -> dict:
    '''Profiles of the particles of ``count`` frames

    Returns the profiles ``b``, masks ``m`` and counts ``c``, the residuals
    ``res`` (profile minus model, where measured), and for each particle
    ``z`` [reference pixels], ``a``, ``n``, ``scale``, ``edge`` (distance to
    the nearest edge) and ``neighbor`` (to the nearest other particle).
    '''
    rng = rng or frames.rng
    keys = ('b', 'm', 'c', 'res', 'z', 'a', 'n', 'scale', 'edge',
            'neighbor')
    out: dict[str, list] = {k: [] for k in keys}
    rows, cols = frames.ranges.shape
    for _ in range(count):
        image, p, s = frames.frame(scale)
        if len(p) == 0:
            continue
        x = p.x_p.to_numpy() + jitter * rng.normal(size=len(p))
        y = p.y_p.to_numpy() + jitter * rng.normal(size=len(p))
        t = torch.from_numpy(image).to(profiler.device)
        with torch.no_grad():
            b, m, c = (q.cpu().numpy() for q in profiler(t, x, y, s))
        z = p.z_p.to_numpy() * s                 # reference pixels
        clean = model(z, p.a_p.to_numpy(), p.n_p.to_numpy()).cpu().numpy()
        P = p[['x_p', 'y_p']].to_numpy()
        d = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
        d[np.arange(len(P)), np.arange(len(P))] = np.inf
        out['b'].append(b)
        out['m'].append(m)
        out['c'].append(c)
        out['res'].append((b - clean) * m)
        out['z'].append(z)
        out['a'].append(p.a_p.to_numpy())
        out['n'].append(p.n_p.to_numpy())
        out['scale'].append(np.full(len(p), s))
        out['edge'].append(np.minimum.reduce([P[:, 0], P[:, 1],
                                              cols - 1 - P[:, 0],
                                              rows - 1 - P[:, 1]]))
        out['neighbor'].append(d.min(1))
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def saturate(profile: torch.Tensor, mask: torch.Tensor,
             level: float | torch.Tensor = CAMERA_LEVEL) -> torch.Tensor:
    '''The mask of the rings that a camera of limited range measures

    Parameters
    ----------
    profile : torch.Tensor
        Profiles, in units of the background.
    mask : torch.Tensor
        1 where a radius is measured, 0 elsewhere.
    level : float | torch.Tensor
        The brightest pixel that the camera records, in units of the
        background: a number, or one for each profile (shape ``(P, 1)``).
        Default: ``CAMERA_LEVEL``

    Returns
    -------
    mask : torch.Tensor
        ``mask`` without the rings whose profile reaches ``level``.
    '''
    return mask * (profile < level).to(mask.dtype)


class Batches:
    '''Training examples: clean profiles of random particles plus residuals

    Parameters
    ----------
    library : dict
        ``res`` and ``m`` from :func:`measure`.
    model : RadialModel
        On all the radii, on the device of the training.
    seed : int
    saturated : float
        Fraction of the examples that are made as a camera that saturates
        records them (see :func:`saturate`).  Default: 0.5
    level : float | tuple of float
        The brightest pixel of that camera, in units of the background: a
        number, or the range (low, high) from which each example draws its
        own, with the background uniformly between 1 / high and 1 / low of
        the full scale of the camera (an exposure is set, in practice, so
        that the background is about a third to half of the range).
        Default: ``CAMERA_LEVEL``
    '''

    def __init__(self, library: dict, model: RadialModel, seed: int,
                 saturated: float = 0.5,
                 level: float | tuple[float, float] = CAMERA_LEVEL) -> None:
        self.res = torch.as_tensor(library['res'], device=model.device)
        self.m = torch.as_tensor(library['m'], device=model.device)
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.saturated = float(saturated)
        self.level: float | tuple[float, float] = (
            (float(level[0]), float(level[1]))
            if isinstance(level, (tuple, list)) else float(level))

    def __call__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Inputs (n, 4, NRADII) and targets (n, 3)'''
        rng = self.rng
        a = log_uniform(rng, A_P, n)
        c = log_uniform(rng, CONTRAST, n)
        z = np.maximum(log_uniform(rng, Z_UM, n) / PIXEL, 1.2 * a / PIXEL)
        clean = self.model(z, a, N_M + c).to(torch.float32)
        k = torch.as_tensor(rng.integers(len(self.res), size=n),
                            device=self.model.device)
        profile, m = clean + self.res[k], self.m[k]
        if self.saturated > 0:
            hit = torch.as_tensor(rng.random(n) < self.saturated,
                                  device=self.model.device)
            level: float | torch.Tensor
            if isinstance(self.level, tuple):
                low, high = self.level
                level = torch.as_tensor(
                    1. / rng.uniform(1. / high, 1. / low, n),
                    dtype=profile.dtype, device=profile.device)[:, None]
            else:
                level = self.level
            m = torch.where(hit[:, None], saturate(profile, m, level), m)
        x = profile_features(profile, m, PIXEL)
        t = np.log(np.stack([z, a, c], 1)) - CENTER
        return x, torch.as_tensor(100 * t, dtype=torch.float32,
                                  device=self.model.device)


def loss_of(p: torch.Tensor, logit: torch.Tensor,
            t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Winner-takes-all regression, and cross-entropy of the winner'''
    w = torch.tensor(WEIGHT, device=p.device)
    e = ((p - t[:, None]).abs() * w).sum(-1)
    return (e.min(1).values.mean() + .02 * e.mean(),
            nn.functional.cross_entropy(logit, e.argmin(1)))


def evaluate(net: ProfileNet, val: dict, device: torch.device) -> dict:
    '''Candidates near the truth, for particles away from edges and others'''
    net.eval()
    b = torch.as_tensor(val['b'], device=device)
    m = torch.as_tensor(val['m'], device=device)
    levels = (CAMERA_LEVEL, 1.6)            # cameras that saturate
    cuts = [saturate(b, m, level) for level in levels]
    with torch.no_grad():
        p = net.candidates(profile_features(b, m, PIXEL)).cpu().numpy()
        qs = [net.candidates(profile_features(b, cut, PIXEL)).cpu().numpy()
              for cut in cuts]
    net.train()

    def closest(c: np.ndarray) -> np.ndarray:
        return np.maximum.reduce([
            np.abs(c[..., 0] / val['z'][:, None] - 1),
            np.abs(c[..., 1] / val['a'][:, None] - 1),
            np.abs(c[..., 2] / (val['n'][:, None] - N_M) - 1) / 2]).min(1)

    near = closest(p)
    standard = (val['edge'] >= 100) & (val['neighbor'] >= 150)
    z_um = val['z'] * PIXEL
    out = {}
    for level, cut, q in zip(levels, cuts, qs, strict=True):
        clipped = standard & (cut.cpu().numpy() < m.cpu().numpy()).any(1)
        if clipped.any():
            after = closest(q)
            key = 'saturated' if level == CAMERA_LEVEL \
                else f'saturated at {level:g}'
            out[key] = dict(
                n=int(clipped.sum()),
                within3=float((after[clipped] <= .03).mean()),
                within1=float((after[clipped] <= .01).mean()),
                within3_unmasked=float((near[clipped] <= .03).mean()))
    for s in np.unique(val['scale']):
        for lo, hi in BANDS:
            g = standard & (val['scale'] == s) & (z_um >= lo) & (z_um < hi)
            if g.sum():
                key = f'scale {s:g}, {lo}-{hi} um'
                out[key] = dict(n=int(g.sum()),
                                within3=float((near[g] <= .03).mean()),
                                within1=float((near[g] <= .01).mean()))
    return out


def _library(path: Path, ranges: Ranges, scales: tuple[float, float],
             seed: int, library_frames: int, validation_frames: int,
             device: torch.device) -> dict:
    '''The library of the residuals and the validation profiles of a run

    Measured once, from synthetic frames, and kept in ``path``.

    Parameters
    ----------
    path : pathlib.Path
        ``library.npz`` of the run, which is read if it is there.
    ranges : Ranges
        The frames of the library.
    scales : tuple of float
        The two ends of the range of scales, which the validation is at.
    seed : int
        Seed of the frames of the library.
    library_frames, validation_frames : int
        Number of frames of the library, and of each end of the validation.
    device : torch.device
        Where the profiles are measured.

    Returns
    -------
    library : dict
        The arrays of :func:`measure`, and those of the validation as
        ``val_...``.
    '''
    if path.exists():
        return dict(np.load(path))
    profiler = Profiler(device, compile=False)
    exact = RadialModel(RADII, REFERENCE['wavelength'], PIXEL, N_M, 'cpu',
                        compile=False)
    print('measuring the library', flush=True)
    library = measure(Frames(seed + 7000, ranges), library_frames, profiler,
                      exact)
    val: dict[str, Any] = {}
    for vs, s in zip(VALIDATION_SEEDS, scales, strict=True):
        part = measure(Frames(vs, Ranges(**{
            **ranges.__dict__, 'scales': (s, s)})), validation_frames,
            profiler, exact, jitter=0., scale=s)
        for k, v in part.items():
            val[k] = np.concatenate([val[k], v]) if k in val else v
    arrays: dict[str, Any] = {**library,
                              **{f'val_{k}': v for k, v in val.items()}}
    np.savez(path, **arrays)
    library = dict(np.load(path))
    print(f"library {len(library['res'])} profiles, validation "
          f"{len(library['val_z'])}", flush=True)
    return library


def train(out: str | Path, steps: int, init: str | None = None,
          lr: float = 1e-3, batch: int = 512, every: int = 3000,
          scales: tuple[float, float] = (1., 2.5), library_frames: int = 1500,
          validation_frames: int = 120, seed: int = 21,
          shape: tuple[int, int] = (1024, 1280),
          device: str | None = None, saturated: float = 0.5,
          levels: tuple[float, float] | None = None) -> None:
    '''Train (or fine-tune) the ProfileNet; see the module's documentation'''
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device) if device else default_device()
    ranges = Ranges(**{**LIBRARY.__dict__, 'scales': tuple(scales),
                       'shape': tuple(shape)})
    library = _library(out / 'library.npz', ranges, scales, seed,
                       library_frames, validation_frames, dev)
    val = {k[4:]: v for k, v in library.items() if k.startswith('val_')}
    net = ProfileNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    state = out / 'state.pt'
    step = restore(net, opt, state, 'estimator', init, dev)
    configure(out, 'estimator', steps, init, scales=list(scales))
    clean = RadialModel(RADII, REFERENCE['wavelength'], PIXEL, N_M, dev,
                        chunk=batch, compile=dev.type != 'cpu')
    batches = Batches(library, clean, seed + step, saturated,
                      (float(levels[0]), float(levels[1])) if levels
                      else CAMERA_LEVEL)
    warm = min(500, max(steps // 10, 1))
    start, last, total = time.time(), step, 0.
    while step < steps:
        rate = min(1., (step + 1) / warm) * (
            .01 + .99 * .5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g['lr'] = lr * rate
        x, t = batches(batch)
        p, logit = net(x)
        reg, ce = loss_of(p, logit, t)
        loss = reg + .1 * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        total += float(reg)
        step += 1
        if step % 100 == 0 or step == steps:
            n = step - last
            print(f'{step} loss {total / n:.3f} ce {float(ce):.2f} '
                  f'{n * batch / (time.time() - start):.0f} profiles/s',
                  flush=True)
            start, last, total = time.time(), step, 0.
        if step % 500 == 0 or step == steps:
            checkpoint(state, net=net.state_dict(), opt=opt.state_dict(),
                       step=step)
        if step % every == 0 or step == steps:
            log_evaluation(out, dict(step=step, **evaluate(net, val, dev)))
            start, last = time.time(), step
    torch.save(net.state_dict(), out / 'weights.pt')
    seal(out)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    '''Train the Estimator (see the module docstring)

    Parameters
    ----------
    argv : list[str] | None
        Command line.  Default: None, ``sys.argv``
    '''
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('out', help='directory of the model and the logs')
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--init', default=None,
                        help='fine-tune these weights (a model or a '
                        'directory)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--every', type=int, default=3000)
    parser.add_argument('--scales', type=float, nargs=2, default=(1., 2.5))
    parser.add_argument('--library-frames', type=int, default=1500)
    parser.add_argument('--validation-frames', type=int, default=120)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--saturated', type=float, default=0.5,
                        help='fraction of the examples made as a camera '
                        'that saturates records them')
    parser.add_argument('--levels', type=float, nargs=2, default=None,
                        metavar=('LOW', 'HIGH'),
                        help='the brightest pixel that the camera records, '
                        'in units of the background, is drawn for each '
                        'saturated example between LOW and HIGH (the '
                        'background at 1/HIGH to 1/LOW of the full scale). '
                        'Default: 2.55 for all, an 8-bit camera with a '
                        'background of 100, which is how the shipped model '
                        'was trained')
    parser.add_argument('--device', default=None)
    a = parser.parse_args(argv)
    train(a.out, a.steps, a.init, a.lr, a.batch, a.every, tuple(a.scales),
          a.library_frames, a.validation_frames, a.seed, device=a.device,
          saturated=a.saturated, levels=tuple(a.levels))


if __name__ == '__main__':  # pragma: no cover
    main()
