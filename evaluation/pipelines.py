'''Pipelines that find and characterize the particles in a scene

A pipeline is a function of a ``Scene`` that returns a DataFrame with one row
for each particle that it found and the columns ``x_p``, ``y_p`` (pixels),
``z_p`` (pixels), ``a_p`` (micrometers) and ``n_p``.  It must use only
``scene.image``; the pipelines that cheat by reading ``scene.truth`` say so.
'''
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from CATCH.evaluation.cases import INSTRUMENT
from CATCH.evaluation.scenes import Scene

Pipeline = Callable[[Scene], pd.DataFrame]

COLUMNS = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')


def crop_around(image: np.ndarray, x: float, y: float,
                side: int) -> tuple[np.ndarray, int, int]:
    '''A square crop around a point, shifted inside the image if it must be

    This is how ``CATCHapp.crop`` treats a feature that crosses an edge.

    Parameters
    ----------
    image : numpy.ndarray
        The image.
    x, y : float
        The center of the crop [pixels].
    side : int
        The length of a side [pixels], at most the shorter side of the image.

    Returns
    -------
    crop, x0, y0 : numpy.ndarray, int, int
        The crop and the coordinates in the image of its corner.
    '''
    rows, columns = image.shape[:2]
    side = int(min(side, rows, columns))
    x0 = int(np.clip(round(x - side / 2), 0, columns - side))
    y0 = int(np.clip(round(y - side / 2), 0, rows - side))
    return image[y0:y0 + side, x0:x0 + side], x0, y0


def extent(z_p: float, a_p: float, n_p: float) -> float:
    '''Radius [pixels] of the feature of a particle'''
    from pylorenzmie.theory import Sphere

    from CATCH.utilities.mtd import feature_extent
    sphere = Sphere(a_p=a_p, n_p=n_p, z_p=z_p)
    return feature_extent(sphere, {'instrument': INSTRUMENT})


def truth_pipeline() -> Pipeline:
    '''A pipeline that cheats completely: it returns the truth

    Fitting from the true particles measures the best that any pipeline
    can do in a scene: the neighbors and the edges can still keep a fit from
    locking on.
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        return scene.truth.loc[scene.truth.inside, list(COLUMNS)].copy()
    return pipeline


def estimator_on_truth(estimator: Callable[[list[np.ndarray]], Any],
                       minimum: int = 31) -> Pipeline:
    '''A pipeline that cheats: it knows where the particles are

    It takes the true centers of the particles that are in the image as the
    detections, and crops around each as large as twice the extent of its
    feature (which the Localizer does not know), as in training, and gives the
    crops to the estimator.  It measures the estimator, with the effects of
    the neighbors and of the edge, apart from the Localizer.

    Parameters
    ----------
    estimator : callable
        Takes a list of crops and returns a DataFrame of ``z_p``, ``a_p``
        and ``n_p``, such as ``CATCH.Estimator``.
    minimum : int
        Smallest side of a crop [pixels].  Default: 31
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        truth = scene.truth[scene.truth.inside]
        crops = []
        for _, p in truth.iterrows():
            side = max(int(2 * extent(p.z_p, p.a_p, p.n_p)), minimum)
            crops.append(crop_around(scene.image, p.x_p, p.y_p, side)[0])
        if not crops:
            return pd.DataFrame(columns=list(COLUMNS))
        estimates = estimator(crops)
        out = pd.DataFrame(dict(x_p=truth.x_p.to_numpy(),
                                y_p=truth.y_p.to_numpy()))
        for key in ('z_p', 'a_p', 'n_p'):
            out[key] = estimates[key].to_numpy()
        return out
    return pipeline


def catch_pipeline(app: Callable[[np.ndarray], pd.DataFrame]) -> Pipeline:
    '''The pipeline of CATCH: ``CATCHapp``, which needs yolov5

    The image is given to the app as an 8-bit image with a mean of 100, as
    the Localizer expects it.

    Parameters
    ----------
    app : callable
        Takes an image and returns a DataFrame of features with the columns
        ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``, such as
        ``CATCH.CATCHapp``.
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        image = np.clip(100. * scene.image, 0, 255).astype(np.uint8)
        found = app(image)
        if len(found) == 0:
            return pd.DataFrame(columns=list(COLUMNS))
        return pd.DataFrame(found[list(COLUMNS)]).reset_index(drop=True)
    return pipeline
