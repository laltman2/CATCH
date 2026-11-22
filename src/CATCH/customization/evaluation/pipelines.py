'''Pipelines that find and characterize the particles in a scene

A pipeline is a function of a ``Scene`` that returns a DataFrame with one row
for each particle that it found and the columns ``x_p``, ``y_p`` (pixels),
``z_p`` (pixels), ``a_p`` (micrometers) and ``n_p``.  It must use only
``scene.image``; the pipelines that cheat by reading ``scene.truth`` say so.
'''
from collections.abc import Callable, Mapping
from typing import Any

from pathlib import Path

import numpy as np
import pandas as pd

from CATCH.customization.evaluation.cases import INSTRUMENT
from CATCH.customization.evaluation.scenes import Scene, to_uint8
from CATCH.lib.fitter import crop_square

Pipeline = Callable[[Scene], pd.DataFrame]

COLUMNS = ('x_p', 'y_p', 'z_p', 'a_p', 'n_p')


def pad_around(image: np.ndarray, x: float, y: float, side: int,
               value: float = 1.) -> np.ndarray:
    '''A square crop centered on a point, padded where it leaves the image

    Unlike ``CATCH.lib.fitter.crop_square`` the particle stays at the
    center of the crop, which is what the Estimator was trained on; what is
    outside the image is background (``value``, the mean of a normalized
    hologram).

    Parameters
    ----------
    image : numpy.ndarray
        The image.
    x, y : float
        The center of the crop [pixels].
    side : int
        The length of a side [pixels].
    value : float
        What fills the part outside the image.  Default: 1
    '''
    rows, columns = image.shape[:2]
    x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
    out = np.full((side, side), value, dtype=float)
    top, left = max(y0, 0), max(x0, 0)
    bottom, right = min(y0 + side, rows), min(x0 + side, columns)
    if bottom > top and right > left:
        out[top - y0:bottom - y0, left - x0:right - x0] = \
            image[top:bottom, left:right]
    return out


def extent(z_p: float, a_p: float, n_p: float,
           instrument: Mapping[str, float] | None = None) -> float:
    '''Radius [pixels] of the feature of a particle

    Parameters
    ----------
    z_p, a_p, n_p : float
        Axial position [pixels of the instrument], radius [um] and
        refractive index.
    instrument : mapping | None
        ``wavelength`` and ``magnification`` [um] and ``n_m``.  Default:
        None, the reference instrument.

    Returns
    -------
    extent : float
        Radius of the feature [pixels of the instrument].
    '''
    from pylorenzmie.theory import Sphere

    from CATCH.customization.evaluation.extent import feature_extent
    sphere = Sphere(a_p=a_p, n_p=n_p, z_p=z_p)
    return feature_extent(sphere,
                          {'instrument': dict(instrument or INSTRUMENT)})


def truth_pipeline() -> Pipeline:
    '''A pipeline that cheats completely: it returns the truth

    Fitting from the true particles measures the best that any pipeline
    can do in a scene: the neighbors and the edges can still keep a fit from
    locking on.
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        return scene.truth.loc[scene.truth.inside, list(COLUMNS)].copy()
    return pipeline


def estimator_on_truth(estimator: Callable[..., pd.DataFrame],
                       mask_saturated: bool = False) -> Pipeline:
    '''A pipeline that cheats: it knows where the particles are

    It takes the true centers of the particles that are in the image as the
    detections and gives them, with the image, to the estimator.  It
    measures the estimator, with the effects of the neighbors and of the
    edge, apart from the Localizer.

    Parameters
    ----------
    estimator : callable
        Takes the image, the centers ``x`` and ``y`` [pixels] and the
        keywords ``instrument`` and ``noise`` (those of the scene), and
        returns a DataFrame of ``z_p``, ``a_p`` and ``n_p`` (and ``snr``, if
        it has it), such as ``CATCH.Estimator``.
    mask_saturated : bool
        Also give the estimator the keyword ``saturation``, the brightest
        pixel that the camera of the scene records (``scenes.record``), so
        that it leaves the saturated pixels out.  Default: False
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        truth = scene.truth[scene.truth.inside]
        if len(truth) == 0:
            return pd.DataFrame(columns=list(COLUMNS))
        options: dict[str, Any] = dict(instrument=scene.instrument,
                                       noise=scene.noise)
        if mask_saturated and scene.saturation is not None:
            options['saturation'] = scene.saturation
        estimates = estimator(scene.image, truth.x_p.to_numpy(),
                              truth.y_p.to_numpy(), **options)
        out = pd.DataFrame(dict(x_p=truth.x_p.to_numpy(),
                                y_p=truth.y_p.to_numpy()))
        for key in ('z_p', 'a_p', 'n_p', 'snr'):
            if key in estimates:
                out[key] = estimates[key].to_numpy()
        return out
    return pipeline


def crops_on_truth(estimator: Callable[[list[np.ndarray]], Any],
                       minimum: int = 31, pad: bool = False) -> Pipeline:
    '''A pipeline that cheats: it knows where the particles are

    It takes the true centers of the particles that are in the image as the
    detections, and crops around each as large as twice the extent of its
    feature (which a detector does not know), and gives the crops to an
    estimator of crops, such as the references of pylorenzmie.  It
    measures the estimator, with the effects of the neighbors and of the
    edge, apart from a detector.

    Parameters
    ----------
    estimator : callable
        Takes a list of crops and returns a DataFrame of ``z_p``, ``a_p``
        and ``n_p``, such as the estimators of ``references``, which are
        for the reference instrument: the pipeline refuses the scenes of
        another.
    minimum : int
        Smallest side of a crop [pixels].  Default: 31
    pad : bool
        Give the estimator crops that are centered on the particle and
        padded with background where they leave the image (``pad_around``),
        in place of crops that are shifted inside it.  Default: False
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        if dict(scene.instrument) != INSTRUMENT:
            raise ValueError('the estimators of the references are for the '
                             'reference instrument, not '
                             f'{dict(scene.instrument)}')
        truth = scene.truth[scene.truth.inside]
        crops = []
        for _, p in truth.iterrows():
            side = max(int(2 * extent(p.z_p, p.a_p, p.n_p)), minimum)
            if pad:
                crops.append(pad_around(scene.image, p.x_p, p.y_p, side))
            else:
                crops.append(
                    crop_square(scene.image, p.x_p, p.y_p, side)[0])
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
    '''The pipeline of CATCH: ``CATCHapp``

    The image is given to the app as an 8-bit image with a mean of 100, as
    it would be read from a file (the way that the baseline of CATCH 2 was
    measured).

    Parameters
    ----------
    app : callable
        Takes an image and returns a DataFrame of features with the columns
        ``x_p``, ``y_p``, ``z_p``, ``a_p`` and ``n_p``, such as
        ``CATCH.CATCHapp``.
    '''
    def pipeline(scene: Scene) -> pd.DataFrame:
        found = app(to_uint8(scene.image))
        if len(found) == 0:
            return pd.DataFrame(columns=list(COLUMNS))
        keep = list(COLUMNS) + [c for c in ('snr', 'saturated')
                                if c in found.columns]
        return pd.DataFrame(found[keep]).reset_index(drop=True)
    return pipeline


def recorded_pipeline(path: str | Path) -> Pipeline:
    '''A pipeline that returns what a program found, from a CSV file

    This evaluates a program that cannot run in this process, such as another
    version of CATCH (which is also called ``CATCH``): write the images with
    ``scenes.write_images``, run the program on them
    (``CATCH.customization.evaluation.detect_images`` does it for
    ``CATCHapp``), and score the file that it writes.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with a row for each particle that was found, and the
        columns ``scene`` (the ``index`` of the scene) and ``x_p``, ``y_p``,
        ``z_p``, ``a_p`` and ``n_p``.
    '''
    recorded = pd.read_csv(path)
    missing = [c for c in ('scene', *COLUMNS) if c not in recorded.columns]
    if missing:
        raise ValueError(f'{path} has no column {missing}')

    def pipeline(scene: Scene) -> pd.DataFrame:
        rows = recorded[recorded['scene'] == scene.index]
        return pd.DataFrame(rows[list(COLUMNS)]).reset_index(drop=True)
    return pipeline
