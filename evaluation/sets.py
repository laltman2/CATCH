'''The scenes that versions of CATCH are compared on

There are two sets, both made by ``scenes.make_scenes`` from a seed, so that
anyone can make the same scenes:

``dev``
    For development: try things, look at the failures.  These are the scenes
    of the baseline of the original CATCH.
``test``
    Frozen.  Use it to compare finished versions, and not to choose among
    ideas, or the numbers stop being an honest measure of how a version does
    on scenes that it has not been tuned to.  It is larger than ``dev``
    because it should detect small differences.

The truth about every particle is stored in ``sets/<name>_truth.csv``, so
that a change to the code that makes the scenes, which would invalidate the
comparisons, is noticed (``verify``).
'''
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from CATCH.evaluation.scenes import SHAPE, Scene, make_scenes

DIRECTORY = Path(__file__).parent / 'sets'

#: the columns of the truth that identify a particle in a scene
KEYS = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')


@dataclass(frozen=True)
class SceneSet:
    '''How to make a set of scenes (see ``scenes.make_scenes``)'''
    name: str
    nscenes: int
    seed: int
    nparticles: tuple[int, int] = (1, 6)
    shape: tuple[int, int] = SHAPE
    noise: float = 0.05
    margin: float = 0.
    min_separation: float = 20.

    def make(self, nscenes: int | None = None) -> list[Scene]:
        '''Makes the scenes, or the first ``nscenes`` of them

        The scenes are made in turn from one random number generator, so the
        first ``n`` scenes are the same however many are asked for.
        '''
        count = self.nscenes if nscenes is None else nscenes
        if not 0 < count <= self.nscenes:
            raise ValueError(f'the set {self.name!r} has {self.nscenes} '
                             f'scenes, not {count}')
        return make_scenes(count, self.nparticles, self.shape, self.noise,
                           self.seed, margin=self.margin,
                           min_separation=self.min_separation)

    def path(self) -> Path:
        '''File that holds the truth of the set'''
        return DIRECTORY / f'{self.name}_truth.csv'


SETS = {
    'dev': SceneSet('dev', nscenes=200, seed=2026),
    'test': SceneSet('test', nscenes=400, seed=31415),
}


def truth_table(scenes: list[Scene]) -> pd.DataFrame:
    '''The truth about the particles in the scenes, with the ``scene``'''
    tables = [scene.truth.assign(scene=scene.index) for scene in scenes]
    return pd.concat(tables, ignore_index=True)


def write_truth(name: str) -> Path:
    '''Makes the scenes of a set and stores the truth about them'''
    scene_set = SETS[name]
    DIRECTORY.mkdir(exist_ok=True)
    path = scene_set.path()
    truth_table(scene_set.make()).to_csv(path, index=False)
    return path


def verify(name: str, nscenes: int = 3) -> None:
    '''Checks that the code still makes the scenes that were stored

    Parameters
    ----------
    name : str
        ``'dev'`` or ``'test'``.
    nscenes : int
        Number of scenes to make and compare.  The first scenes of a set do
        not depend on how many there are, so a few are a good check, and
        they are made quickly.  Default: 3

    Raises
    ------
    ValueError
        If a particle differs from the stored truth.
    '''
    scene_set = SETS[name]
    stored = pd.read_csv(scene_set.path())
    made = truth_table(scene_set.make(nscenes))
    wanted = list(range(nscenes))
    stored = pd.DataFrame(
        stored[stored['scene'].isin(wanted)]).reset_index(drop=True)
    columns = [*KEYS, 'edge', 'neighbor', 'scene']
    if len(made) != len(stored):
        raise ValueError(f'the set {name!r} has {len(made)} particles in '
                         f'its first {nscenes} scenes, not {len(stored)}')
    if not np.allclose(made[columns].to_numpy(dtype=float),
                       stored[columns].to_numpy(dtype=float),
                       rtol=0, atol=1e-6, equal_nan=True):
        raise ValueError(f'the scenes of the set {name!r} are not the ones '
                         'that were stored; the code that makes them has '
                         'changed')


def get(name: str) -> SceneSet:
    '''The set with a name, or ``ValueError``'''
    if name not in SETS:
        raise ValueError(f'unknown set {name!r}; the sets are {list(SETS)}')
    return replace(SETS[name])
