'''How well does a Localizer find the centers, and where does it fail?

A localizer (a function of an 8-bit image returning a DataFrame with
``x_p``, ``y_p``, ``bbox``, ``edge`` and ``confidence``, such as
``CATCH.CATCHapp``, whose box is sized from the estimate) is run on the
scenes of a set, and each true particle is matched to the detections twice:
by the center that the localizer returns and by the middle of the box.
The table says, for each particle, whether it was found, how far the center
is, and whether its box is cut by the border of the image (``clipped``), and
on both axes (``corner``), where the size of the feature cannot be read from
the box.

Usage::

    python -m CATCH.customization.evaluation.centers [--set dev] [--nscenes N]
        [--output FILE]
'''
import argparse
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from CATCH.customization.evaluation.matching import match
from CATCH.customization.evaluation.pipelines import extent
from CATCH.customization.evaluation.scenes import Scene, to_uint8

Localize = Callable[[np.ndarray], pd.DataFrame | None]
#: bins of the distance from a particle to the nearest edge [pixels]
EDGE_BINS = (0., 25., 100., 250., np.inf)


def centers(localize: Localize, scenes: list[Scene],
            gate: float = 100., tolerance: int = 3) -> pd.DataFrame:
    '''For each particle inside the scenes: what the localizer found

    Parameters
    ----------
    localize : callable
        Takes an 8-bit image, returns the detections (see above).
    scenes : list of Scene
    gate : float
        Largest distance [pixels] for a match: wide, so that the detections
        whose center is far off are counted, with their error.  Default: 100
    tolerance : int
        A box whose edge is within this of the border is clipped [pixels].
        Default: 3

    Returns
    -------
    table : pandas.DataFrame
        A row for each particle: ``scene``, ``edge``, ``a_p``, ``extent``,
        ``found`` and ``err`` (distance of the returned center), ``raw_found``
        and ``raw_err`` (of the middle of the box), and for the matched
        detection ``flag`` (``edge``), ``clipped``, ``corner``, ``side``
        (``max(w, h)``) and ``ratio`` (``side`` over twice the extent).
    '''
    rows = []
    for scene in scenes:
        truth = scene.truth[scene.truth.inside].reset_index(drop=True)
        height, width = scene.image.shape
        found = localize(to_uint8(scene.image))
        if found is None or len(found) == 0:
            found = pd.DataFrame(columns=['x_p', 'y_p', 'bbox', 'edge'])
        boxes = [(x0, y0, w, h) for (x0, y0), w, h in found.bbox]
        box = np.array(boxes, dtype=float).reshape(-1, 4)
        middle = np.column_stack([box[:, 0] + box[:, 2] / 2.,
                                  box[:, 1] + box[:, 3] / 2.])
        corrected = np.asarray(found[['x_p', 'y_p']],
                               dtype=float).reshape(-1, 2)
        xy = np.asarray(truth[['x_p', 'y_p']], dtype=float)
        by_center = match(xy, corrected, gate)
        by_middle = match(xy, middle, gate)
        for i, p in truth.iterrows():
            row: dict[str, Any] = dict(
                scene=scene.index, edge=p.edge, a_p=p.a_p,
                extent=extent(p.z_p, p.a_p, p.n_p), found=False,
                err=np.nan, raw_found=False, raw_err=np.nan)
            hit = np.flatnonzero(by_center.truth == i)
            if hit.size:
                j = int(by_center.detection[hit[0]])
                x0, y0, w, h = box[j]
                clip_x = x0 <= tolerance or x0 + w >= width - tolerance
                clip_y = y0 <= tolerance or y0 + h >= height - tolerance
                row.update(found=True, err=by_center.distance[hit[0]],
                           flag=bool(found.edge.iloc[j]),
                           clipped=bool(clip_x or clip_y),
                           corner=bool(clip_x and clip_y),
                           side=max(w, h),
                           ratio=max(w, h) / (2. * row['extent']))
            hit = np.flatnonzero(by_middle.truth == i)
            if hit.size:
                row.update(raw_found=True, raw_err=by_middle.distance[hit[0]])
            rows.append(row)
    return pd.DataFrame(rows)


def recall(table: pd.DataFrame, gate: float = 15.,
           raw: bool = False) -> pd.Series:
    '''Fraction of the particles found within a gate, by distance to the edge

    Parameters
    ----------
    table : pandas.DataFrame
        The result of ``centers``.
    gate : float
        Largest error of the center [pixels].  Default: 15
    raw : bool
        Use the middle of the box, not the center that the localizer
        returns.  Default: False
    '''
    prefix = 'raw_' if raw else ''
    hit = table[prefix + 'found'].fillna(False).astype(bool) \
        & (table[prefix + 'err'] <= gate)
    bins = pd.cut(table.edge, EDGE_BINS, right=False)
    return pd.Series(hit.groupby(bins, observed=True).mean())


def main() -> None:  # pragma: no cover
    '''Measure the recall and precision of the Localizer
    '''
    from CATCH import CATCHapp
    from CATCH.customization.evaluation.sets import SETS

    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--set', choices=list(SETS), default='dev',
                        dest='scene_set')
    parser.add_argument('--nscenes', type=int, default=None)
    parser.add_argument('--output', help='write the table to a CSV file')
    args = parser.parse_args()
    table = centers(CATCHapp(), SETS[args.scene_set].make(
        args.nscenes))
    if args.output:
        table.to_csv(args.output, index=False)
    pd.options.display.float_format = '{:.3f}'.format
    print('recall by distance to the edge')
    print(pd.DataFrame({'15 px': recall(table, 15.),
                        '100 px': recall(table, 100.),
                        'middle of the box, 15 px': recall(table, 15, True),
                        'middle of the box, 100 px': recall(table, 100, True)}
                       ).to_string())
    hit = pd.DataFrame(table[table.found.fillna(False).astype(bool)])
    clipped = pd.DataFrame(hit[hit.clipped.astype(bool)])
    print(f'\nclipped boxes: {len(clipped)}; corners: '
          f'{int(clipped.corner.sum())}')
    corner = clipped.corner.astype(bool)
    for name, group in (('one axis', pd.DataFrame(clipped[~corner])),
                        ('corner', pd.DataFrame(clipped[corner]))):
        print(f'  {name}: median error {group.err.median():.1f} px, '
              f'{(group.err > 15).mean():.0%} over 15 px, side/(2 extent) '
              f'{group.ratio.median():.2f}')


if __name__ == '__main__':  # pragma: no cover
    main()
