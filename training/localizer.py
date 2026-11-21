'''Train the detector of the Localizer on synthetic frames

Usage::

    python -m CATCH.training.localizer OUT --steps 20000
    python -m CATCH.training.localizer OUT --steps 8000 --init default

The network (:class:`CATCH.lib.networks.DetectorNet`) learns a heatmap of the
centers (a focal loss on Gaussian peaks, as in CenterNet) and the position
of each center inside its cell of 4 pixels (an L1 loss at the cells of the
centers).  The frames are made on the fly (:mod:`CATCH.training.frames`),
with pixels of 1 to 2.5 reference pixels unless ``--scales`` says
otherwise.  ``--init`` starts from trained weights (a model name such as
``default``, or a directory) to fine-tune them, which takes a fraction of
the steps.

``OUT`` receives ``weights.pt`` and ``configuration.json`` (a model that
``Localizer(OUT)`` loads), ``state.pt`` (to resume an interrupted run:
run the same command again) and ``eval.jsonl``: recall, false detections
and the median error of the centers on fixed validation frames at both
ends of the range of scales.  The shipped model took 20,000 steps of 4
frames from scratch and 8,000 more to fine-tune the offsets, about 4 hours
on an Apple M-series GPU with ``PYLORENZMIE_BACKEND=torch``.
'''
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from CATCH.evaluation.matching import match
from CATCH.Localizer import default_device, model_directory
from CATCH.lib.networks import STRIDE, DetectorNet, grid_shape, peaks
from CATCH.training.frames import Frames, Ranges

#: width of the Gaussian peak of a center [cells]
SIGMA = 1.5
#: seeds of the validation frames, one per end of the range of scales
VALIDATION_SEEDS = (7600, 7601)
#: a detection belongs to a particle within this distance [pixels]
GATE = 15.


def targets(centers: list[np.ndarray], shape: tuple[int, int],
            device: torch.device) -> tuple[torch.Tensor, ...]:
    '''Heatmaps (B, h, w), offsets (B, 2, h, w) and their mask (B, h, w)

    The heatmap is a Gaussian around each center, exactly 1 in the cell
    of the center; the offset is the position of the center in that cell.
    '''
    h, w = grid_shape(shape)
    heat = torch.zeros(len(centers), h, w, device=device)
    offset = torch.zeros(len(centers), 2, h, w, device=device)
    mask = torch.zeros(len(centers), h, w, device=device)
    yy = (torch.arange(h, device=device) * STRIDE + 1.5) / STRIDE
    xx = (torch.arange(w, device=device) * STRIDE + 1.5) / STRIDE
    for i, c in enumerate(centers):
        for x, y in c:
            g = torch.exp(-((xx - x / STRIDE) ** 2)[None] / (2 * SIGMA ** 2)
                          - ((yy - y / STRIDE) ** 2)[:, None]
                          / (2 * SIGMA ** 2))
            j = min(int(y // STRIDE), h - 1)
            k = min(int(x // STRIDE), w - 1)
            g[j, k] = 1.
            heat[i] = torch.maximum(heat[i], g)
            offset[i, 0, j, k] = min(max(x / STRIDE - k, 0.), 1.)
            offset[i, 1, j, k] = min(max(y / STRIDE - j, 0.), 1.)
            mask[i, j, k] = 1.
    return heat, offset, mask


def focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''The focal loss of CenterNet, per center'''
    p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    hit = target >= 1
    loss = torch.where(hit, -(1 - p) ** 2 * torch.log(p),
                       -(1 - target) ** 4 * p ** 2 * torch.log(1 - p))
    return loss.sum() / hit.sum().clamp(min=1)


def validation(ranges: Ranges, count: int) -> dict[float, tuple]:
    '''Fixed frames at both ends of the range of scales'''
    out = {}
    for seed, scale in zip(VALIDATION_SEEDS, ranges.scales, strict=True):
        frames = Frames(seed, ranges)
        made = [frames.frame(scale) for _ in range(count)]
        out[scale] = ([m[0] for m in made], [m[1] for m in made])
    return out


def score(net: DetectorNet, images: list[np.ndarray], truths: list,
          device: torch.device, threshold: float
          ) -> dict[str, float | None]:
    '''Recall, false detections and the errors of the centers'''
    errors, found, total, false = [], 0, 0, 0
    for k in range(0, len(images), 4):
        batch = torch.from_numpy(np.stack(images[k:k + 4])).to(device)
        with torch.no_grad():
            xs, ys, vs = peaks(net, batch)
        for x, y, v, t in zip(xs, ys, vs, truths[k:k + 4], strict=True):
            keep = v >= threshold
            detections = np.column_stack([x[keep], y[keep]])
            m = match(t[['x_p', 'y_p']].to_numpy(), detections, GATE)
            found += len(m.truth)
            total += len(t)
            false += int(keep.sum()) - len(m.truth)
            errors.extend(np.asarray(m.distance, float).tolist())
    e = np.array(errors)
    return dict(recall=found / max(total, 1), false=false,
                center_median=float(np.median(e)) if len(e) else None,
                center_over2=int((e > 2).sum()))


def train(out: str | Path, steps: int, init: str | None = None,
          lr: float | None = None, batch: int = 4, every: int = 1000,
          ranges: Ranges | None = None, seed: int = 7900, nval: int = 40,
          offset_weight: float = 1., threshold: float = .35,
          device: str | None = None) -> None:
    '''Train (or fine-tune) the detector; see the module's documentation'''
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ranges = ranges or Ranges()
    dev = torch.device(device) if device else default_device()
    net = DetectorNet().to(dev)
    lr = lr if lr is not None else (3e-4 if init else 1e-3)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    warm = min(300, max(steps // 10, 1))

    def rate(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        u = (s - warm) / max(steps - warm, 1)
        return .01 + .99 * .5 * (1 + math.cos(math.pi * min(u, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    step, state = 0, out / 'state.pt'
    if state.exists():
        s = torch.load(state, map_location=dev, weights_only=False)
        net.load_state_dict(s['net'])
        opt.load_state_dict(s['opt'])
        sched.load_state_dict(s['sched'])
        step = s['step']
        print('resumed at step', step, flush=True)
    elif init:
        weights = model_directory('localizer', init) / 'weights.pt'
        print('fine-tuning', weights, flush=True)
        net.load_state_dict(torch.load(weights, map_location=dev,
                                       weights_only=True))
    config = json.loads((model_directory('localizer', None)
                         / 'configuration.json').read_text())
    config.update(description=f'trained by CATCH.training.localizer, '
                  f'{steps} steps' + (f' from {init}' if init else ''),
                  scales=list(ranges.scales), threshold=threshold)
    (out / 'configuration.json').write_text(json.dumps(config, indent=2))
    val = validation(ranges, nval)
    frames = Frames(seed + step, ranges)
    t, seen, totals = time.time(), 0, np.zeros(2)
    while step < steps:
        net.train()
        made = [frames.frame() for _ in range(batch)]
        images = torch.from_numpy(np.stack([m[0] for m in made])).to(dev)
        centers = [m[1][['x_p', 'y_p']].to_numpy() for m in made]
        heat, offset = net(images)
        th, to, mask = targets(centers, ranges.shape, dev)
        lh = focal(heat, th)
        lo = ((offset - to).abs().sum(1) * mask).sum() / mask.sum().clamp(
            min=1)
        loss = lh + offset_weight * lo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        sched.step()
        step += 1
        seen += batch
        totals += (float(lh), float(lo))
        if step % 100 == 0 or step == steps:
            n = 100 if step % 100 == 0 else step % 100
            print(f'{step} heat {totals[0] / n:.3f} offset '
                  f'{totals[1] / n:.3f} '
                  f'{seen / (time.time() - t):.1f} frames/s', flush=True)
            t, seen, totals = time.time(), 0, np.zeros(2)
        if step % 500 == 0 or step == steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), step=step),
                       str(state) + '.tmp')
            os.replace(str(state) + '.tmp', state)
        if step % every == 0 or step == steps:
            net.eval()
            r: dict = dict(step=step)
            for scale, (ims, ts) in val.items():
                r[f'scale {scale:g}'] = score(net, ims, ts, dev, threshold)
            print('EVAL', json.dumps(r), flush=True)
            with open(out / 'eval.jsonl', 'a') as f:
                f.write(json.dumps(r) + '\n')
            t, seen = time.time(), 0
    torch.save(net.state_dict(), out / 'weights.pt')


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('out', help='directory of the model and the logs')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--init', default=None,
                        help='fine-tune these weights (a model or a '
                        'directory)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (1e-3, or 3e-4 with --init)')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--every', type=int, default=1000)
    parser.add_argument('--scales', type=float, nargs=2, default=(1., 2.5))
    parser.add_argument('--shape', type=int, nargs=2, default=(1024, 1280))
    parser.add_argument('--seed', type=int, default=7900)
    parser.add_argument('--nval', type=int, default=40)
    parser.add_argument('--device', default=None)
    a = parser.parse_args(argv)
    train(a.out, a.steps, a.init, a.lr, a.batch, a.every,
          Ranges(shape=tuple(a.shape), scales=tuple(a.scales)), a.seed,
          a.nval, device=a.device)


if __name__ == '__main__':  # pragma: no cover
    main()
