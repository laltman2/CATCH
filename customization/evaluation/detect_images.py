'''Find the particles in the images of scenes with CATCHapp, and record them

This script does not import ``CATCH.customization.evaluation``, so that it
can run with any version of CATCH: put the directory that holds the
checkout to be tested (named ``CATCH``) on ``PYTHONPATH``::

    PYTHONPATH=/path/to/parent python detect_images.py IMAGES detections.csv

``IMAGES`` is the directory that ``scene_run --write-images`` wrote.  The file
that it writes is scored by ``scene_run --pipeline recorded``.
'''
import argparse
import re
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd

COLUMNS = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p']


def detect(directory: str | Path,
           app: Callable) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Runs an app on the images ``scene_NNNN.png`` of a directory

    Parameters
    ----------
    directory : str or pathlib.Path
        The images.
    app : callable
        Takes a grayscale image and returns a DataFrame of the particles
        with the columns ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``
        (and ``confidence``), or nothing: ``CATCH.CATCHapp``.

    Returns
    -------
    detections, times : pandas.DataFrame, pandas.DataFrame
        The particles that were found, with the number of the scene, and the
        time that the app took for each scene [seconds].
    '''
    rows, times = [], []
    paths = sorted(Path(directory).glob('scene_*.png'))
    if not paths:
        raise FileNotFoundError(f'no scene_*.png in {directory}')
    for path in paths:
        number = re.search(r'scene_(\d+)', path.name)
        if number is None:
            continue
        scene = int(number.group(1))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        found = app(image)
        times.append(dict(scene=scene, seconds=time.perf_counter() - start))
        if len(found):
            columns = [c for c in COLUMNS + ['confidence']
                       if c in found.columns]
            rows.append(found[columns].assign(scene=scene))
    detections = (pd.concat(rows, ignore_index=True) if rows
                  else pd.DataFrame(columns=['scene'] + COLUMNS))
    return detections, pd.DataFrame(times)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    '''Detect the particles in a directory of scenes

    Parameters
    ----------
    argv : list[str] | None
        Command line.  Default: None, ``sys.argv``
    '''
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').split('\n')[0])
    parser.add_argument('images', help='directory of scene_NNNN.png')
    parser.add_argument('output', help='CSV file to write')
    parser.add_argument('--lmodel', default=None,
                        help='name of the Localizer model')
    parser.add_argument('--emodel', default=None,
                        help='name of the Estimator model')
    parser.add_argument('--instrument', type=float, nargs=3, default=None,
                        metavar=('WAVELENGTH', 'MAGNIFICATION', 'N_M'),
                        help='the microscope that made the images (see '
                        'scene_run.py).  Default: the reference instrument '
                        'of the networks')
    args = parser.parse_args(argv)
    instrument = (None if args.instrument is None else dict(zip(
        ('wavelength', 'magnification', 'n_m'), args.instrument,
        strict=True)))

    from CATCH.CATCHapp import CATCHapp
    detections, times = detect(args.images,
                               CATCHapp(lmodel=args.lmodel,
                                        emodel=args.emodel,
                                        instrument=instrument))
    detections.to_csv(args.output, index=False)
    times.to_csv(Path(args.output).with_suffix('.seconds.csv'), index=False)
    print(f'{len(detections)} particles in {len(times)} images, '
          f'{times.seconds.mean():.2f} s per image')


if __name__ == '__main__':  # pragma: no cover
    main()
