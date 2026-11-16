'''Where does an estimator go wrong? Neighbors, edges, or the particle

The same particles of a set of scenes are given to an estimator under four
conditions:

``scene``
    the crop of the scene, as ``estimator_on_truth`` makes it: the fringes of
    the neighbors are in it, and near an edge it is shifted so that the
    particle is off center;
``padded``
    the crop of the scene centered on the particle and padded with background
    where it leaves the image (``pipelines.pad_around``): neighbors, but no
    shift;
``alone``
    the particle alone in an image of the same shape at the same position:
    the same crop, shifted the same way, without the neighbors;
``centered``
    the particle alone at the center of the image: a clean crop, as in
    training.

The difference of ``scene`` and ``alone`` is the effect of the neighbors and
the shift; ``alone`` against ``centered`` is the shift alone, ``padded``
against ``scene`` what padding in place of shifting recovers; what is left in
``centered`` is the estimator itself.

Usage::

    python -m CATCH.evaluation.diagnose [--set dev] [--nscenes N]
        [--output PREFIX]
'''
import argparse
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from CATCH.evaluation.cases import INSTRUMENT
from CATCH.evaluation.pipelines import crop_around, extent, pad_around
from CATCH.evaluation.scenes import Scene, describe, render

CONDITIONS = ('scene', 'padded', 'alone', 'centered')
#: bins of the quantities that the errors are tabulated by
BINS = {'a_p': (0., 1., 2., 3., 4., np.inf),
        'z_p': (0., 150., 250., 350., 450., np.inf),
        'side': (0., 100., 200., 300., np.inf),
        'neighbor': (0., 150., 400., np.inf),
        'edge': (0., 25., 100., np.inf)}

Estimate = Callable[[list[np.ndarray]], Any]


def alone(scene: Scene, row: pd.Series, centered: bool,
          seed: int) -> np.ndarray:
    '''The image of one particle of a scene, by itself

    Parameters
    ----------
    scene : Scene
        The scene; its shape and noise are used.
    row : pandas.Series
        The particle (``x_p``, ``y_p``, ``z_p``, ``a_p``, ``n_p``).
    centered : bool
        Put the particle at the center of the image.
    seed : int
        Seed of the noise.
    '''
    rows, columns = scene.image.shape
    particle = pd.DataFrame([row[['x_p', 'y_p', 'z_p', 'a_p', 'n_p']]])
    if centered:
        particle['x_p'], particle['y_p'] = columns / 2., rows / 2.
    particle = describe(particle, (rows, columns))
    return render(particle, (rows, columns), scene.noise,
                  np.random.default_rng(seed))


def crops(scene: Scene, minimum: int = 31, seed: int = 0
          ) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    '''The crops of the particles inside a scene, under the three conditions

    Returns
    -------
    truth : pandas.DataFrame
        The particles inside the image, with ``side`` of their crops.
    crops : dict
        For each condition, a list of crops, in the order of ``truth``.
    '''
    truth = pd.DataFrame(scene.truth[scene.truth.inside])
    out: dict[str, list[np.ndarray]] = {c: [] for c in CONDITIONS}
    sides = []
    for k, (_, p) in enumerate(truth.iterrows()):
        side = max(int(2 * extent(p.z_p, p.a_p, p.n_p)), minimum)
        sides.append(min(side, *scene.image.shape))
        out['scene'].append(crop_around(scene.image, p.x_p, p.y_p, side)[0])
        out['padded'].append(pad_around(scene.image, p.x_p, p.y_p, side))
        for centered in (False, True):
            image = alone(scene, p, centered, seed + 1000 * scene.index + k)
            name = 'centered' if centered else 'alone'
            out[name].append(
                crop_around(image, image.shape[1] / 2. if centered else p.x_p,
                            image.shape[0] / 2. if centered else p.y_p,
                            side)[0])
    truth['side'] = sides
    return truth, out


def errors(estimate: Estimate, scenes: list[Scene],
           n_m: float = INSTRUMENT['n_m']) -> pd.DataFrame:
    '''Relative errors of an estimator for each particle and condition

    Parameters
    ----------
    estimate : callable
        Takes a list of crops and returns a DataFrame of ``z_p``, ``a_p``
        and ``n_p``, such as ``CATCH.Estimator``.
    scenes : list of Scene
    n_m : float
        Refractive index of the medium.

    Returns
    -------
    errors : pandas.DataFrame
        A row for each particle and condition: the true ``a_p``, ``z_p``,
        ``n_p``, ``side``, ``neighbor``, ``edge``, and the errors ``da``,
        ``dz`` (relative) and ``dn`` (relative to ``n_p - n_m``).
    '''
    frames = []
    for scene in scenes:
        truth, made = crops(scene)
        if truth.empty:
            continue
        for condition in CONDITIONS:
            found = estimate(made[condition])
            frame = truth[['a_p', 'z_p', 'n_p', 'side', 'neighbor',
                           'edge']].reset_index(drop=True).copy()
            frame['condition'] = condition
            frame['scene'] = scene.index
            frame['da'] = (found['a_p'].to_numpy() - frame.a_p) / frame.a_p
            frame['dz'] = (found['z_p'].to_numpy() - frame.z_p) / frame.z_p
            frame['dn'] = ((found['n_p'].to_numpy() - frame.n_p)
                           / (frame.n_p - n_m))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    '''Median absolute errors and the fraction of good estimates

    ``z<10%`` and ``a<5%`` are the fractions of the estimates within those
    relative errors, ``both`` of the estimates that are within both (the
    start from which a fit tends to lock on).

    Parameters
    ----------
    table : pandas.DataFrame
        The result of ``errors``.
    by : str, optional
        One of the keys of ``BINS``: tabulate for each bin of it.
        Default: all the particles.
    '''
    frame = table.assign(
        az=table.dz.abs(), aa=table.da.abs(), an=table.dn.abs(),
        **{'z<10%': table.dz.abs() < 0.1, 'a<5%': table.da.abs() < 0.05})
    frame['both'] = frame['z<10%'] & frame['a<5%']
    keys: list[Any] = ['condition']
    if by:
        frame['bin'] = pd.cut(frame[by], BINS[by], right=False)
        keys = ['bin', 'condition']
    grouped = frame.groupby(keys, observed=True)
    out = grouped[['az', 'aa', 'an']].median()
    out.columns = pd.Index(['|dz|', '|da|', '|dn|'])
    for column in ('z<10%', 'a<5%', 'both'):
        out[column] = grouped[column].mean()
    out.insert(0, 'n', grouped.size())
    order = {c: i for i, c in enumerate(CONDITIONS)}
    return pd.DataFrame(out.sort_index(
        key=lambda idx: idx.map(order) if idx.name == 'condition' else idx))


def main() -> None:  # pragma: no cover
    from CATCH import Estimator
    from CATCH.evaluation.sets import SETS

    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--set', choices=list(SETS), default='dev',
                        dest='scene_set')
    parser.add_argument('--nscenes', type=int, default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--output', help='write the table of errors to '
                        '<output>_errors.csv')
    args = parser.parse_args()
    scenes = SETS[args.scene_set].make(args.nscenes)
    estimator = Estimator(model=args.model)
    table = errors(estimator, scenes)
    if args.output:
        table.to_csv(f'{args.output}_errors.csv', index=False)
    pd.options.display.float_format = '{:.3f}'.format
    print('all particles')
    print(summarize(table).to_string())
    for key in BINS:
        print(f'\nby {key}')
        print(summarize(table, key).to_string())


if __name__ == '__main__':  # pragma: no cover
    main()
