'''Models with random weights, for tests that must not load trained ones'''
import json
import shutil
from pathlib import Path

import torch

from CATCH.networks import DetectorNet, ProfileNet

PACKAGE = Path(__file__).resolve().parent.parent


def write_models(directory: str | Path, seed: int = 0,
                 **changes) -> tuple[Path, Path]:
    '''Directories of a Localizer and an Estimator with random weights

    The configurations are those of the default models, with ``changes``
    (keys of either configuration) applied.
    '''
    torch.manual_seed(seed)
    out = []
    for kind, net in (('localizer', DetectorNet()),
                      ('estimator', ProfileNet())):
        path = Path(directory) / kind
        path.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path / 'weights.pt')
        source = PACKAGE / f'cfg_{kind}' / 'default' / 'configuration.json'
        config = json.loads(source.read_text())
        config.update({k: v for k, v in changes.items() if k in config})
        (path / 'configuration.json').write_text(json.dumps(config))
        out.append(path)
    return out[0], out[1]


def remove(directory: str | Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
