'''Evaluate a pipeline on scenes with several particles

For each scene the pipeline finds the particles; they are matched to the
truth; a fit that starts from each match, on a crop of the scene, says whether
the particle was found *and* characterized well enough for the fit to lock on.
The neighbors of a particle are in its crop, as the fringes of other
particles, so crowding counts against it.

Usage::

    python -m CATCH.customization.evaluation.scene_run \
        [--pipeline oracle|catch] [--nscenes 20] [--nparticles 1 5] \
        [--nprocs 4] [--output prefix]
'''
import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from CATCH.customization.evaluation.cases import INSTRUMENT
from CATCH.customization.evaluation.fitting import (
    GRID, GRID_N, LOSSES, RELATIVE, SCENE_LOSS, SPREAD, fit_limits,
    fit_particle, locks_on, spread_start)
from CATCH.customization.evaluation.matching import match
from CATCH.customization.evaluation.pipelines import (
    COLUMNS, Pipeline, catch_pipeline, crop_around, crops_on_truth,
    estimator_on_truth, extent, recorded_pipeline, truth_pipeline)
from CATCH.customization.evaluation.references import (
    NAMES, catch_estimator, pylorenzmie_estimator)
from CATCH.customization.evaluation.scan import Scan, scan as scan_grid
from CATCH.customization.evaluation.scenes import (
    CAMERA_LEVEL, SHAPE, Scene, add_instrument_option,
    instrument_from_option, make_scenes, record, write_images)
from CATCH.customization.evaluation.sets import SETS

#: classes of particles to fit, for ``--only``
ONLY: dict[str, Callable[[Mapping[str, Any]], bool] | None] = {
    'all': None,
    'large-high': lambda p: p['a_p'] >= 3. and p['n_p'] > 1.5}
#: edges of the bins of the summaries, by the property of the particle
BINS = {'nparticles': (1, 2, 3, 4, 6, np.inf),
        'saturated': (0, 1e-6, 0.01, 0.05, 0.2, np.inf),
        'snr': (0, 50, 100, 200, 400, 800, 1600, np.inf),
        'edge': (-np.inf, 0, 25, 100, 250, np.inf),
        'neighbor': (0, 50, 150, 400, np.inf),
        'a_p': (0, 1, 2, 3, 4, np.inf)}


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
        ``a_fit``, ``n_fit``, ``redchi``, and ``redchi0``, the reduced
        chi-squared of the start before the fit); ``ok``: the particle was
        found and the fit locked on (the radius within 5 nm); and ``close``:
        found and the radius within 1% (the criterion that takes the size
        of the particle into account).  For a fit from several starts the
        fit is the one with the least ``redchi``, and ``a_fit_0``,
        ``redchi_0``, ``a_fit_1``, ... are those of each start.  ``ok_any``
        says that some start locked on: what the best choice among the
        starts could achieve.
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
    spread: tuple[tuple[float, float, float], ...] = ((1., 1., 1.),)
    loss: str = SCENE_LOSS
    stages: tuple[float, ...] = ()
    scan: Scan | None = None
    instrument: Mapping[str, float] = field(
        default_factory=lambda: dict(INSTRUMENT))
    saturation: float | None = None


def fit_start(task: tuple[SceneJob, tuple[float, float, float]]
              ) -> dict[str, float]:
    '''One fit of a job, from the estimate with the radius, ``z_p`` and the
    contrast of the index scaled

    The unit of work for a process, so that the starts of one particle run
    side by side: the fits of large spheres take seconds, and the fits of
    one such particle would otherwise be the last to finish.  The position
    is in the image.
    '''
    job, (factor, z_factor, contrast) = task
    result = fit_particle(job.data, job.noise,
                          spread_start(job.start, factor,
                                       job.instrument['n_m'], z_factor,
                                       contrast), job.loss, job.stages,
                          job.instrument, fit_limits(job.instrument),
                          saturation=job.saturation)
    result['x_p'] += job.x0
    result['y_p'] += job.y0
    return result


def fit_scanned(job: SceneJob) -> dict[str, float]:
    '''The fits of a job from the estimate and from the scan around it

    The scan renders the model on a grid around the estimate and returns
    the local minima of the chi-squared
    (``CATCH.customization.evaluation.scan``); each is the start of a fit.
    The fit with the least reduced chi-squared is the result, and the time
    of the scan is ``scan_seconds``.
    '''
    assert job.scan is not None
    begin = time.perf_counter()
    starts, chi = scan_grid(job.data, job.noise, job.start, job.scan,
                            instrument=job.instrument)
    seconds = time.perf_counter() - begin
    limits = fit_limits(job.instrument)
    results = [fit_particle(job.data, job.noise, start, job.loss, job.stages,
                            job.instrument, limits, saturation=job.saturation)
               for start in [job.start, *starts]]
    for result in results:
        result['x_p'] += job.x0
        result['y_p'] += job.y0
    out = combine(results)
    out['scan_seconds'] = seconds
    out['scan_starts'] = float(len(starts))
    out['scan_chi'] = float(np.min(chi)) if np.isfinite(chi).any() \
        else float('nan')
    return out


def combine(results: Sequence[dict[str, float]]) -> dict[str, float]:
    '''The fit with the least reduced chi-squared of the fits of a particle

    The first of the results is the fit from the estimate, which sets
    ``redchi0``.  With several, the radius, the axial position and the
    chi-squared of each are returned as ``a_fit_i``, ``z_fit_i`` and
    ``redchi_i``.
    '''
    chi = np.array([r['redchi'] for r in results], dtype=float)
    best = results[int(np.nanargmin(chi))] if np.isfinite(chi).any() \
        else results[0]
    result = dict(best)
    result['redchi0'] = results[0]['redchi0']
    if len(results) > 1:
        for i, r in enumerate(results):
            result[f'a_fit_{i}'] = r['a_p']
            result[f'z_fit_{i}'] = r['z_p']
            result[f'n_fit_{i}'] = r['n_p']
            result[f'redchi_{i}'] = r['redchi']
    return result


def fit_job(job: SceneJob) -> dict[str, float]:
    '''Makes the fits of a job: the one with the least chi-squared'''
    return combine([fit_start((job, factor)) for factor in job.spread])


def make_job(row: int, scene: Scene, detection: pd.Series,
             minimum: int = 101,
             spread: tuple[tuple[float, float, float], ...] = ((1., 1., 1.),),
             loss: str = SCENE_LOSS,
             stages: tuple[float, ...] = (),
             scan: Scan | None = None,
             mask_saturated: bool = False) -> SceneJob | None:
    '''The fit that starts from a detection, or None if it has no start

    The crop is as large as twice the extent of the feature of the
    estimated particle.  With ``mask_saturated``, the pixels that a camera
    saturated (``scene.saturation``) are left out of the fit.
    '''
    values = np.array([detection[key] for key in COLUMNS], dtype=float)
    if not (np.isfinite(values).all() and (values[2:] > 0).all()):
        return None
    z_p, a_p, n_p = values[2:]
    side = max(int(2 * extent(z_p, a_p, n_p, scene.instrument)), minimum)
    crop, x0, y0 = crop_around(scene.image, values[0], values[1], side)
    start = {key: float(value)
             for key, value in zip(COLUMNS, values, strict=True)}
    start['x_p'] -= x0
    start['y_p'] -= y0
    return SceneJob(row, crop, scene.noise, start, x0, y0, spread, loss,
                    stages, scan, scene.instrument,
                    scene.saturation if mask_saturated else None)


def saturated_fraction(scene: Scene, particle: Mapping[str, float]
                       ) -> float:
    '''Fraction of the crop of a particle that a camera would saturate

    The crop is the one of the fits, twice the extent of the feature of the
    true particle.  A pixel is saturated if it is at or above the limit of
    the camera that recorded the scene (``scene.saturation``), or
    ``CAMERA_LEVEL`` for an image that was not recorded by ``record``, which
    says what would be lost.

    Parameters
    ----------
    scene : Scene
        The scene.
    particle : mapping
        The true ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``.

    Returns
    -------
    fraction : float
    '''
    side = max(int(2 * extent(particle['z_p'], particle['a_p'],
                              particle['n_p'], scene.instrument)), 101)
    crop = crop_around(scene.image, particle['x_p'], particle['y_p'],
                       side)[0]
    level = CAMERA_LEVEL if scene.saturation is None else scene.saturation
    return float(np.mean(crop >= level - 1e-9))


def _triple(start: Sequence[float]) -> tuple[float, float, float]:
    '''Factors of the radius, ``z_p`` and the contrast; the last may be left
    out'''
    a, z, c = (*(float(f) for f in start), 1., 1.)[:3]
    return a, z, c


def evaluate_scenes(pipeline: Pipeline,
                    scenes: Sequence[Scene],
                    gate: float = 15.,
                    nprocs: int = 1,
                    chunk: int | None = None,
                    starts: int | Sequence[Sequence[float]] = 1,
                    loss: str = SCENE_LOSS,
                    stages: Sequence[float] = (),
                    scan: Scan | None = None,
                    only: Callable[[Mapping[str, Any]], bool] | None = None,
                    mask_saturated: bool = False
                    ) -> SceneResults:
    '''Runs a pipeline on scenes, matches, and fits from what it found

    The fits of a scene need its image and the crops for the fits are kept
    until they are made, so the memory that this takes grows with the number
    of scenes; ``chunk`` bounds it.  The results are the same.

    ``starts`` is the number of fits of each particle, from the estimate and
    from other radii along the degeneracy of the radius and the index
    (``CATCH.customization.evaluation.fitting.SPREAD``, at most 9); the best by
    reduced chi-squared is the fit, and the others are in the table.
    ``starts`` can also be a sequence of pairs of factors of the radius and
    of ``z_p``, the first being the estimate (``fitting.GRID``), or of
    triples that add a factor of the contrast of the index
    (``fitting.GRID_N``).  ``loss`` is the loss of the fits
    (``fitting.LOSSES``).  ``stages`` are radii (as
    fractions of half the side of the crop) of fits that come before the fit
    of everything, from the inner fringes outward.  ``mask_saturated`` leaves
    out of the fits the pixels that a camera saturated (``scenes.record``).
    '''
    if loss not in LOSSES:
        raise ValueError(f'loss must be one of {LOSSES}, not {loss!r}')
    if not chunk or chunk >= len(scenes):
        return _evaluate(pipeline, scenes, gate, nprocs, starts, loss,
                         tuple(stages), scan, only, mask_saturated)
    parts = [_evaluate(pipeline, scenes[i:i + chunk], gate, nprocs, starts,
                       loss, tuple(stages), scan, only, mask_saturated)
             for i in range(0, len(scenes), chunk)]
    return SceneResults(
        pd.concat([p.particles for p in parts], ignore_index=True),
        pd.concat([p.scenes for p in parts], ignore_index=True))


def _evaluate(pipeline: Pipeline,
              scenes: Sequence[Scene],
              gate: float = 15.,
              nprocs: int = 1,
              starts: int | Sequence[Sequence[float]] = 1,
              loss: str = SCENE_LOSS,
              stages: tuple[float, ...] = (),
              scan: Scan | None = None,
              only: Callable[[Mapping[str, Any]], bool] | None = None,
              mask_saturated: bool = False
              ) -> SceneResults:
    '''Runs a pipeline on scenes, matches, and fits from what it found

    Parameters
    ----------
    pipeline : callable
        Takes a ``Scene`` and returns a DataFrame with a row for each
        particle that it found and the columns ``x_p``, ``y_p``, ``z_p``,
        ``a_p`` and ``n_p`` (see ``CATCH.customization.evaluation.pipelines``).
    scenes : sequence of Scene
        The scenes, from ``make_scenes``.
    gate : float
        A detection is of a particle if it is within this distance
        [pixels].  Default: 15, which is how far a fit can move the center.
    nprocs : int
        Number of processes for the fits.  Default: 1, which fits in this
        process.
    starts : int or sequence of pairs
        Number of fits of each particle, or the pairs of factors of the
        radius and of ``z_p`` of the starts.  Default: 1
    loss : str
        Loss of the fits: ``'linear'``, ``'cauchy'`` or ``'both'``.
        Default: ``SCENE_LOSS`` (``'both'``)
    stages : tuple of float
        Radii of fits before the last, as fractions of half the side of the
        crop.  Default: none
    scan : Scan or None
        Starts from the scan of a grid, in place of ``starts``.
        Default: None
    mask_saturated : bool
        Leave out of the fits the pixels that a camera saturated (the
        scenes must have been recorded by ``scenes.record``).
        Default: False

    Returns
    -------
    results : SceneResults
        A row for each true particle and for each scene.
    '''
    if isinstance(starts, int):
        if not 1 <= starts <= len(SPREAD):
            raise ValueError(f'starts must be from 1 to {len(SPREAD)}')
        spread = tuple((factor, 1., 1.) for factor in SPREAD[:starts])
    else:
        spread = tuple(_triple(start) for start in starts)
        if not spread or spread[0] != (1., 1., 1.) or len(set(spread)) \
                != len(spread):
            raise ValueError('the starts are distinct pairs of factors '
                             'and the first is (1, 1), the estimate')
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
                             saturated=saturated_fraction(scene,
                                                          particle.to_dict()),
                             found=False))
        for k, j, distance in zip(matched.truth, matched.detection,
                                  matched.distance, strict=True):
            row = rows[first + int(k)]
            detection = found.iloc[int(j)]
            row.update(found=True, center_err=float(distance),
                       x_det=detection.x_p, y_det=detection.y_p,
                       z_est=detection.z_p, a_est=detection.a_p,
                       n_est=detection.n_p,
                       snr=detection.get('snr', np.nan))
            if only is not None and not only(row):
                continue
            job = make_job(first + int(k), scene, detection, spread=spread,
                           loss=loss, stages=stages, scan=scan,
                           mask_saturated=mask_saturated)
            if job is not None:
                jobs.append(job)
    if scan is not None:
        if nprocs > 1 and len(jobs) > 1:
            order = sorted(range(len(jobs)),
                           key=lambda k: -jobs[k].start['a_p'])
            with ProcessPoolExecutor(nprocs) as pool:
                done = list(pool.map(fit_scanned, [jobs[k] for k in order],
                                     chunksize=1))
            fits = [{} for _ in jobs]
            for k, result in zip(order, done, strict=True):
                fits[k] = result
        else:
            fits = [fit_scanned(job) for job in jobs]
    elif nprocs > 1 and len(jobs) * len(spread) > 1:
        # the slowest first, the radius being the best guess of the time
        tasks = sorted(((k, job, factor) for k, job in enumerate(jobs)
                        for factor in job.spread),
                       key=lambda task: -task[1].start['a_p'])
        with ProcessPoolExecutor(nprocs) as pool:
            done = list(pool.map(fit_start, [t[1:] for t in tasks],
                                 chunksize=1))
        results = {(k, factor): result
                   for (k, _, factor), result in zip(tasks, done,
                                                     strict=True)}
        fits = [combine([results[k, factor] for factor in spread])
                for k in range(len(jobs))]
    else:
        fits = [fit_job(job) for job in jobs]
    for job, result in zip(jobs, fits, strict=True):
        rows[job.row].update(x_fit=result['x_p'], y_fit=result['y_p'],
                             z_fit=result['z_p'], a_fit=result['a_p'],
                             n_fit=result['n_p'], redchi=result['redchi'],
                             redchi0=result['redchi0'])
        for key in result:
            if key.startswith(('a_fit_', 'z_fit_', 'n_fit_', 'redchi_',
                               'scan_')):
                rows[job.row][key] = result[key]
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
    if 'redchi0' not in df:
        df['redchi0'] = np.nan
    df['ok'] = [bool(found) and locks_on(fit, a_p)
                for found, fit, a_p in zip(df['found'], df['a_fit'],
                                           df['a_p'], strict=True)]
    df['close'] = [bool(found) and locks_on(fit, a_p, relative=RELATIVE)
                   for found, fit, a_p in zip(df['found'], df['a_fit'],
                                              df['a_p'], strict=True)]
    fits = [c for c in df if c.startswith('a_fit_')]
    if fits:
        df['ok_any'] = [bool(found) and any(
            locks_on(fit, a_p) for fit in row if np.isfinite(fit))
            for found, a_p, row in zip(df['found'], df['a_p'],
                                       df[fits].to_numpy(), strict=True)]
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
        fitted, and ``lock_on_found``, of those that were found; ``close``,
        the fraction found and fitted to within 1% of the radius;
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
        close=_mean(pd.Series(p['close'])),
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
        from the nearest edge, negative outside), ``'neighbor'`` (lateral
        distance to the nearest other particle) or ``'a_p'`` (radius), or
        another column.
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
        particles in the bin), ``close`` (the radius within 1%, of all the
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
            lock_on=group['ok'].mean(), close=group['close'].mean(),
            lock_on_found=found['ok'].mean() if len(found) else np.nan,
            center_med=_col(found, 'center_err').median())
    return pd.DataFrame.from_dict(rows, orient='index').round(3)


def report(results: SceneResults) -> str:
    '''The summary and the tables by number of particles, saturated
    fraction, distance from the edge and distance to the nearest neighbor,
    as text (a table whose column the results lack, as in an older file,
    is left out)'''
    parts = [summarize_scenes(results).to_string()]
    parts.extend(f'\nby {by}:\n' + binned(results.particles, by).to_string()
                 for by in BINS if by in results.particles)
    return '\n'.join(parts)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    '''Evaluate a pipeline on a set of scenes

    Parameters
    ----------
    argv : list[str] | None
        Command line.  Default: None, ``sys.argv``
    '''
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pipeline',
                        choices=('oracle', 'truth', 'catch', 'recorded'),
                        default='oracle',
                        help='oracle: the true centers, with the sizes of '
                        'the crops, and the Estimator; truth: the true '
                        'particles (the best any pipeline can do); catch: '
                        'CATCHapp; recorded: the detections '
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
    add_instrument_option(parser)
    parser.add_argument('--camera', nargs='?', type=float, const=CAMERA_LEVEL,
                        default=None, metavar='LEVEL',
                        help='record the scenes as a camera does: pixels '
                        'above LEVEL times the background saturate (an '
                        '8-bit camera: 2.55, the default of the option)')
    parser.add_argument('--mask-saturated', action='store_true',
                        help='with --camera: leave the saturated pixels '
                        'out of the fits')
    parser.add_argument('--margin', type=float, default=0.,
                        help='centers can be this far outside the image')
    parser.add_argument('--gate', type=float, default=15.)
    parser.add_argument('--nprocs', type=int, default=4)
    parser.add_argument('--starts', type=int, default=1,
                        help='fits of each particle, from the estimate and '
                        'from other radii along the degeneracy of the '
                        'radius and the index (at most 9)')
    parser.add_argument('--grid', action='store_true',
                        help='fit from the 25 starts of a grid of radii and '
                        'of axial positions, in place of --starts')
    parser.add_argument('--grid-n', action='store_true',
                        help='fit from the 45 starts of a grid of radii, '
                        'of axial positions and of the contrast of the '
                        'index, in place of --starts')
    parser.add_argument('--stages', type=float, nargs='+', default=(),
                        metavar='FRACTION',
                        help='fit first the pixels within these fractions '
                        'of half the side of the crop, in turn, then all')
    parser.add_argument('--scan', action='store_true',
                        help='fit from the local minima of the chi-squared '
                        'of a grid around the estimate, in place of '
                        '--starts')
    parser.add_argument('--scan-spacing', type=float, default=Scan.spacing)
    parser.add_argument('--scan-keep', type=int, default=Scan.keep)
    parser.add_argument('--scan-span', type=float, nargs=3,
                        default=(Scan.z_span, Scan.a_span, Scan.c_span),
                        metavar=('Z', 'A', 'C'))
    parser.add_argument('--only', choices=('all', 'large-high'),
                        default='all',
                        help='large-high: only the particles of 3 um and '
                        'more with an index above 1.5 (a quick look at the '
                        'hard case)')
    parser.add_argument('--pad', action='store_true',
                        help='oracle pipeline: the estimator gets crops '
                        'centered on the particle and padded with background '
                        'at the edges, not shifted inside the image')
    parser.add_argument('--loss', choices=LOSSES, default=SCENE_LOSS,
                        help='loss of the fits: least squares, the robust '
                        'Cauchy loss, or least squares and then Cauchy')
    parser.add_argument('--chunk', type=int, default=50,
                        help='scenes to fit at a time, to bound the memory')
    parser.add_argument('--lmodel', default=None,
                        help='name or directory of the Localizer model '
                        '(catch pipeline)')
    parser.add_argument('--model', default=None,
                        help='name or directory of the Estimator model')
    parser.add_argument('--estimator', choices=('catch', *NAMES),
                        default='catch',
                        help='the estimator of the oracle pipeline: the '
                        'network, or a start that needs none (analytic: '
                        'the azimuthal profile, with a fixed index; de: '
                        'differential evolution)')
    parser.add_argument('--output', help='write the tables to '
                        'OUTPUT_particles.csv and OUTPUT_scenes.csv')
    args = parser.parse_args(argv)
    instrument = instrument_from_option(args.instrument)
    if (instrument and args.pipeline == 'oracle'
            and args.estimator != 'catch'):
        parser.error('--estimator analytic|de are for the reference '
                     'instrument; use --estimator catch with --instrument')

    if args.scene_set:
        if args.scene_set == 'test':
            print('The test set is frozen: compare finished versions on it, '
                  'and tune on the dev set.', file=sys.stderr)
        scenes = SETS[args.scene_set].make(args.nscenes, instrument)
    else:
        scenes = make_scenes(args.nscenes or 20, tuple(args.nparticles),
                             tuple(args.shape), args.noise, args.seed,
                             margin=args.margin, instrument=instrument)
    if args.camera is not None:
        scenes = [record(scene, args.camera) for scene in scenes]
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
        if args.estimator == 'catch':
            pipeline = estimator_on_truth(catch_estimator(args.model),
                                       args.mask_saturated)
        else:
            pipeline = crops_on_truth(pylorenzmie_estimator(args.estimator),
                                      pad=args.pad)
    else:
        from CATCH import CATCHapp
        pipeline = catch_pipeline(CATCHapp(
            lmodel=args.lmodel, emodel=args.model, instrument=instrument))
    results = evaluate_scenes(pipeline, scenes, args.gate, args.nprocs,
                              args.chunk,
                              GRID_N if args.grid_n else GRID if args.grid
                              else args.starts, args.loss, args.stages,
                              Scan(*args.scan_span, spacing=args.scan_spacing,
                                   keep=args.scan_keep) if args.scan
                              else None, ONLY[args.only], args.mask_saturated)
    if args.output:
        results.particles.to_csv(f'{args.output}_particles.csv', index=False)
        results.scenes.to_csv(f'{args.output}_scenes.csv', index=False)
    print(report(results))


if __name__ == '__main__':  # pragma: no cover
    main()
