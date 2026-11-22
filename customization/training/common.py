'''What the two training commands have in common

A run writes a model directory (``weights.pt`` and ``configuration.json``), a
checkpoint (``state.pt``, from which the same command resumes) and the log of
its evaluations (``eval.jsonl``).
'''
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from CATCH.Localizer import model_directory


def restore(net: torch.nn.Module, opt: torch.optim.Optimizer, state: Path,
            kind: str, init: str | None, device: torch.device,
            sched: Any = None) -> int:
    '''Resume a run from its checkpoint, or start from trained weights

    Parameters
    ----------
    net : torch.nn.Module
        The network, which receives the weights.
    opt : torch.optim.Optimizer
        The optimizer, which receives its state if the run resumes.
    state : pathlib.Path
        The checkpoint of the run.
    kind : str
        ``'localizer'`` or ``'estimator'``.
    init : str | None
        Trained weights (a model or a directory) to fine-tune, if the run
        does not resume.
    device : torch.device
        Where the tensors are loaded.
    sched : torch.optim.lr_scheduler.LRScheduler | None
        The scheduler, which receives its state if the run resumes.
        Default: None, there is none.

    Returns
    -------
    step : int
        The step that the run is at: 0, or where the checkpoint was made.
    '''
    if state.exists():
        saved = torch.load(state, map_location=device, weights_only=False)
        net.load_state_dict(saved['net'])
        opt.load_state_dict(saved['opt'])
        if sched is not None:
            sched.load_state_dict(saved['sched'])
        print('resumed at step', saved['step'], flush=True)
        return int(saved['step'])
    if init:
        weights = model_directory(kind, init) / 'weights.pt'
        print('fine-tuning', weights, flush=True)
        net.load_state_dict(torch.load(weights, map_location=device,
                                       weights_only=True))
    return 0


def configure(out: Path, kind: str, steps: int, init: str | None,
              **changes: Any) -> None:
    '''Write the configuration of a model: the shipped one, changed

    Parameters
    ----------
    out : pathlib.Path
        The directory of the model.
    kind : str
        ``'localizer'`` or ``'estimator'``.
    steps : int
        Number of steps of the run, for the description.
    init : str | None
        The weights that were fine-tuned, for the description.
    **changes
        The entries of the configuration that differ from the shipped model.
    '''
    config = json.loads((model_directory(kind, None)
                         / 'configuration.json').read_text())
    config.update(description='trained by '
                  f'CATCH.customization.training.{kind}, {steps} steps'
                  + (f' from {init}' if init else ''), **changes)
    (out / 'configuration.json').write_text(json.dumps(config, indent=2))


def checkpoint(state: Path, **contents: Any) -> None:
    '''Save the state of a run so that no interruption leaves half a file'''
    torch.save(contents, str(state) + '.tmp')
    os.replace(str(state) + '.tmp', state)


def log_evaluation(out: Path, record: dict[str, Any]) -> None:
    '''Print an evaluation and append it to ``eval.jsonl``'''
    print('EVAL', json.dumps(record), flush=True)
    with open(out / 'eval.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
