'''Evaluate a pipeline on scenes with several particles

For each scene the pipeline finds the particles; they are matched to the
truth; a fit that starts from each match, on a crop of the scene, says whether
the particle was found *and* characterized well enough for the fit to lock on.
The neighbors of a particle are in its crop, as the fringes of other
particles, so crowding counts against it.

Usage::

    python -m CATCH.evaluation.scene_run [--pipeline oracle|catch]
        [--nscenes 20] [--nparticles 1 5] [--nprocs 4] [--output prefix]
'''
import argparse
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from CATCH.evaluation.fitting import fit_particle, locks_on
from CATCH.evaluation.matching import match
from CATCH.evaluation.pipelines import (COLUMNS, Pipeline, catch_pipeline,
                                        crop_around, estimator_on_truth,
                                        extent, recorded_pipeline,
                                        truth_pipeline)
from CATCH.evaluation.scenes import SHAPE, Scene, make_scenes, write_images
from CATCH.evaluation.sets import SETS

#: edges of the bins of the summaries, by the property of the particle
BINS = {'nparticles': (1, 2, 3, 4, 6, np.inf),
        'edge': (-np.inf, 0, 25, 100, 250, np.inf),
        'neighbor': (0, 50, 150, 400, np.inf)}


@dataclass(frozen=True)
class SceneResults:
    '''The results of evaluating a pipeline on scenes

    Attributes
    ----------
    particles : pandas.DataFrame
        A row for each true particle: the truth (``scene``, ``index``,
        ``x_p``, ``y_p``, ``z_p``, ``a_p``, ``n_p``, ``nparticles``,
        ``edge``, ``neighbor``, ``inside``); ``found`` and the detection
        (``x_det``, ``y_det``, ``center_err``); its estimates (``z_est``,
        ``a_est``, ``n_est``) and their relative errors (``z_rel``,
        ``a_rel``, ``n_rel``); the fit (``x_fit``, ``y_fit``, ``z_fit``,
        ``a_fit``, ``n_fit``, ``redchi``); and ``ok``: the particle was
        found and the fit locked on.
    scenes : pandas.DataFrame
        A row for each scene: ``nparticles``, ``ndetected``, the true
        positives ``tp``, false positives ``fp`` and false negatives
        ``fn``, and the time of the pipeline ``seconds``.
    '''
    particles: pd.DataFrame
    scenes: pd.DataFrame


@dataclass(frozen=True)
class SceneJob:
    '''A fit of one particle on a crop of a scene'''
    row: int
    data: np.ndarray
    noise: float
    start: dict[str, float]
    x0: int
    y0: int


def fit_job(job: SceneJob) -> dict[str, float]:
    '''Makes the fit of a job; the position is in the image'''
    result = fit_particle(job.data, job.noise, job.start)
    result['x_p'] += job.x0
    result['y_p'] += job.y0
    return result


def make_job(row: int, scene: Scene, detection: pd.Series,
             minimum: int = 101) -> SceneJob | None:
    '''The fit that starts from a detection, or None if it has no start

    The crop is as large as twice the extent of the feature of the
    estimated particle.
    '''
    values = np.array([detection[key] for key in COLUMNS], dtype=float)
    if not (np.isfinite(values).all() and (values[2:] > 0).all()):
        return None
    side = max(int(2 * extent(*values[2:])), minimum)
    crop, x0, y0 = crop_around(scene.image, values[0], values[1], side)
    start = {key: float(value)
             for key, value in zip(COLUMNS, values, strict=True)}
    start['x_p'] -= x0
    start['y_p'] -= y0
    return SceneJob(row, crop, scene.noise, start, x0, y0)


def evaluate_scenes(pipeline: Pipeline,
                    scenes: Sequence[Scene],
                    gate: float = 15.,
                    nprocs: int = 1,
                    chunk: int | None = None) -> SceneResults:
    '''Runs a pipeline on scenes, matches, and fits from what it found

    The fits of a scene need its image and the crops for the fits are kept
    until they are made, so the memory that this takes grows with the number
    of scenes; ``chunk`` bounds it.  The results are the same.
    '''
    if not chunk or chunk >= len(scenes):
        return _evaluate(pipeline, scenes, gate, nprocs)
    parts = [_evaluate(pipeline, scenes[i:i + chunk], gate, nprocs)
             for i in range(0, len(scenes), chunk)]
    return SceneResults(
        pd.concat([p.particles for p in parts], ignore_index=True),
        pd.concat([p.scenes for p in parts], ignore_index=True))


def _evaluate(pipeline: Pipeline,
              scenes: Sequence[Scene],
              gate: float = 15.,
              nprocs: int = 1) -> SceneResults:
    '''Runs a pipeline on scenes, matches, and fits from what it found

    Parameters
    ----------
    pipeline : callable
        Takes a ``Scene`` and returns a DataFrame with a row for each
        particle that it found and the columns ``x_p``, ``y_p``, ``z_p``,
        ``a_p`` and ``n_p`` (see ``CATCH.evaluation.pipelines``).
    scenes : sequence of Scene
        The scenes, from ``make_scenes``.
    gate : float
        A detection is of a particle if it is within this distance
        [pixels].  Default: 15, which is how far a fit can move the center.
    nprocs : int
        Number of processes for the fits.  Default: 1, which fits in this
        process.

    Returns
    -------
    results : SceneResults
        A row for each true particle and for each scene.
    '''
    rows: list[dict] = []
    summary: list[dict] = []
    jobs: list[SceneJob] = []
    for scene in scenes:
        i = scene.index
        start = time.perf_counter()
        found = pipeline(scene).reset_index(drop=True)
        seconds = time.perf_counter() - start
        missing = [c for c in COLUMNS if c not in found.columns]
        if missing:
            raise ValueError(f'the pipeline did not return {missing}')
        truth = scene.truth
        matched = match(truth[['x_p', 'y_p']].to_numpy(),
                        found[['x_p', 'y_p']].to_numpy(), gate)
        summary.append(dict(scene=i, nparticles=len(truth),
                            ndetected=len(found), tp=len(matched.truth),
                            fp=len(matched.spurious),
                            fn=len(matched.missed), seconds=seconds))
        first = len(rows)
        for index, particle in truth.iterrows():
            rows.append(dict(scene=i, index=index, **particle.to_dict(),
                             found=False))
        for k, j, distance in zip(matched.truth, matched.detection,
                                  matched.distance, strict=True):
            row = rows[first + int(k)]
            detection = found.iloc[int(j)]
            row.update(found=True, center_err=float(distance),
                       x_det=detection.x_p, y_det=detection.y_p,
                       z_est=detection.z_p, a_est=detection.a_p,
                       n_est=detection.n_p)
            job = make_job(first + int(k), scene, detection)
            if job is not None:
                jobs.append(job)
    if nprocs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(nprocs) as pool:
            fits = list(pool.map(fit_job, jobs, chunksize=2))
    else:
        fits = [fit_job(job) for job in jobs]
    for job, result in zip(jobs, fits, strict=True):
        rows[job.row].update(x_fit=result['x_p'], y_fit=result['y_p'],
                             z_fit=result['z_p'], a_fit=result['a_p'],
                             n_fit=result['n_p'], redchi=result['redchi'])
    return SceneResults(score(pd.DataFrame(rows)), pd.DataFrame(summary))


def score(particles: pd.DataFrame) -> pd.DataFrame:
    '''Adds the relative errors of the estimates and whether the fit locked on
    '''
    df = particles.copy()
    for key in ('center_err', 'z_est', 'a_est', 'n_est', 'a_fit', 'n_fit',
                'z_fit', 'x_fit', 'y_fit', 'redchi'):
        if key not in df:
            df[key] = np.nan
    for key, truth in (('z', 'z_p'), ('a', 'a_p'), ('n', 'n_p')):
        df[key + '_rel'] = (df[key + '_est'] - df[truth]).abs() / df[truth]
    df['ok'] = [bool(found) and locks_on(fit, a_p)
                for found, fit, a_p in zip(df['found'], df['a_fit'],
                                           df['a_p'], strict=True)]
    return df


def summarize_scenes(results: SceneResults,
                     inside_only: bool = True) -> pd.Series:
    '''Detection, estimates and fits, over all the particles

    Parameters
    ----------
    results : SceneResults
        From ``evaluate_scenes``.
    inside_only : bool
        Count only the particles whose centers are in the image, which is
        where a detector can be expected to find them.  The false positives
        are counted for every scene.  Default: True

    Returns
    -------
    summary : pandas.Series
        ``precision`` and ``recall`` of the detections and their ``f1``;
        ``lock_on``, the fraction of all the particles that were found *and*
        fitted, and ``lock_on_found``, of those that were found;
        ``center_med`` [pixels] and ``z_med``, ``a_med`` and ``n_med``, the
        median errors of the center and the relative errors of the estimates
        of those that were found; ``seconds``, the time of the pipeline
        for a scene.
    '''
    p = select(results.particles, inside_only)
    found = pd.DataFrame(p[p['found']])
    tp, fn = len(found), len(p) - len(found)
    fp = int(np.sum(results.scenes['fp'].to_numpy()))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else np.nan)
    return pd.Series(dict(
        scenes=len(results.scenes), particles=len(p),
        precision=precision, recall=recall, f1=f1,
        lock_on=_mean(pd.Series(p['ok'])),
        lock_on_found=_mean(pd.Series(found['ok'])),
        center_med=_col(found, 'center_err').median(),
        z_med=_col(found, 'z_rel').median(),
        a_med=_col(found, 'a_rel').median(),
        n_med=_col(found, 'n_rel').median(),
        seconds=results.scenes['seconds'].mean())).round(3)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    '''A column of a DataFrame, as a Series'''
    return pd.Series(frame[name])


def _mean(values: pd.Series) -> float:
    '''Mean of the values, nan if there are none'''
    return float(np.mean(values.to_numpy(dtype=float))) if len(values) \
        else np.nan


def select(particles: pd.DataFrame, inside_only: bool) -> pd.DataFrame:
    '''The particles, or only those whose centers are in the image'''
    if not inside_only:
        return particles
    return pd.DataFrame(particles[particles['inside']])


def binned(particles: pd.DataFrame, by: str,
           bins: Sequence[float] | None = None,
           inside_only: bool = True) -> pd.DataFrame:
    '''The recall and the fits of the particles, in bins of a property

    Parameters
    ----------
    particles : pandas.DataFrame
        The ``particles`` of ``SceneResults``.
    by : str
        ``'nparticles'`` (in the scene), ``'edge'`` (distance of the center
        from the nearest edge, negative outside) or ``'neighbor'`` (lateral
        distance to the nearest other particle), or another column.
    bins : sequence of float
        Edges of the bins, each bin including its lower edge.
        Default: ``BINS[by]``
    inside_only : bool
        Count only the particles whose centers are in the image.
        Default: True

    Returns
    -------
    table : pandas.DataFrame
        For each bin: ``n`` (particles), ``recall``, ``lock_on`` (of all the
        particles in the bin), ``lock_on_found`` and ``center_med`` (of the
        error of the center of those found).
    '''
    p = select(particles, inside_only)
    edges = list(BINS[by] if bins is None else bins)
    values = np.nan_to_num(p[by].to_numpy(dtype=float), posinf=1e12)
    groups = pd.cut(pd.Series(values, index=p.index), edges, right=False)
    rows = {}
    for interval, group in p.groupby(groups, observed=True):
        found = pd.DataFrame(group[group['found']])
        rows[str(interval)] = dict(
            n=len(group), recall=group['found'].mean(),
            lock_on=group['ok'].mean(),
            lock_on_found=found['ok'].mean() if len(found) else np.nan,
            center_med=_col(found, 'center_err').median())
    return pd.DataFrame.from_dict(rows, orient='index').round(3)


def report(results: SceneResults) -> str:
    '''The summary and the tables by number of particles, distance from the
    edge and distance to the nearest neighbor, as text'''
    parts = [summarize_scenes(results).to_string()]
    parts.extend(f'\nby {by}:\n' + binned(results.particles, by).to_string()
                 for by in BINS)
    return '\n'.join(parts)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pipeline',
                        choices=('oracle', 'truth', 'catch', 'recorded'),
                        default='oracle',
                        help='oracle: the true centers, with the sizes of '
                        'the crops, and the Estimator; truth: the true '
                        'particles (the best any pipeline can do); catch: '
                        'CATCHapp (needs yolov5); recorded: the detections '
                        'in the CSV file of --detections')
    parser.add_argument('--detections', help='CSV file of the particles '
                        'that a program found (see detect_images.py)')
    parser.add_argument('--write-images', metavar='DIR', help='write the '
                        'images of the scenes to this directory, and stop')
    parser.add_argument('--set', choices=list(SETS), dest='scene_set',
                        help='the scenes of a set, which fixes the options '
                        'below except --nscenes, which can ask for the '
                        'first scenes only.  dev: for development; test: '
                        'frozen, for comparing finished versions')
    parser.add_argument('--nscenes', type=int, default=None,
                        help='default: 20, or all of the set')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--nparticles', type=int, nargs=2, default=(1, 5),
                        metavar=('FEWEST', 'MOST'))
    parser.add_argument('--shape', type=int, nargs=2, default=SHAPE,
                        metavar=('ROWS', 'COLUMNS'))
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.,
                        help='centers can be this far outside the image')
    parser.add_argument('--gate', type=float, default=15.)
    parser.add_argument('--nprocs', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=50,
                        help='scenes to fit at a time, to bound the memory')
    parser.add_argument('--model', default=None,
                        help='name of the Estimator model')
    parser.add_argument('--output', help='write the tables to '
                        'OUTPUT_particles.csv and OUTPUT_scenes.csv')
    args = parser.parse_args(argv)

    if args.scene_set:
        if args.scene_set == 'test':
            print('The test set is frozen: compare finished versions on it, '
                  'and tune on the dev set.', file=sys.stderr)
        scenes = SETS[args.scene_set].make(args.nscenes)
    else:
        scenes = make_scenes(args.nscenes or 20, tuple(args.nparticles),
                             tuple(args.shape), args.noise, args.seed,
                             margin=args.margin)
    if args.write_images:
        paths = write_images(scenes, args.write_images)
        print(f'wrote {len(paths)} images to {args.write_images}')
        return
    if args.pipeline == 'recorded':
        if not args.detections:
            parser.error('--pipeline recorded needs --detections')
        pipeline = recorded_pipeline(args.detections)
    elif args.pipeline == 'truth':
        pipeline = truth_pipeline()
    elif args.pipeline == 'oracle':
        from CATCH import Estimator
        pipeline = estimator_on_truth(Estimator(model=args.model))
    else:
        from CATCH import CATCHapp
        pipeline = catch_pipeline(CATCHapp(emodel=args.model))
    results = evaluate_scenes(pipeline, scenes, args.gate, args.nprocs,
                              args.chunk)
    if args.output:
        results.particles.to_csv(f'{args.output}_particles.csv', index=False)
        results.scenes.to_csv(f'{args.output}_scenes.csv', index=False)
    print(report(results))


if __name__ == '__main__':  # pragma: no cover
    main()
