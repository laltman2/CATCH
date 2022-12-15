from CATCH.version import __version__
from CATCH.Localizer import Localizer
from CATCH.Estimator import Estimator
import numpy as np
import pandas as pd
from typing import (Optional, Union, List)


Images = Union[List[np.ndarray], np.ndarray]


def crop_frame(image: np.ndarray,
               features: pd.DataFrame) -> List[np.ndarray]:
    crops = []
    img_cols, img_rows = image.shape[:2]
    for n, feature in features.iterrows():
        xc, yc = int(round(feature.x_p)), int(round(feature.y_p))
        _, w, h = feature.bbox
        cropsize = max(w, h)
        right_top = int(np.ceil(cropsize/2.))
        left_bot = int(np.floor(cropsize/2.))
        xbot = xc - left_bot
        xtop = xc + right_top
        ybot = yc - left_bot
        ytop = yc + right_top
        if xbot < 0:
            xbot = 0
            xtop = cropsize
        if ybot < 0:
            ybot = 0
            ytop = cropsize
        if xtop > img_rows:
            xtop = img_rows
            xbot = img_rows - cropsize
        if ytop > img_cols:
            ytop = img_cols
            ybot = img_cols - cropsize
        crop = image[ybot:ytop, xbot:xtop]
        crops.append(crop)
    return crops


class CATCHapp(object):
    '''Detect, localize and characterize colloidal particles
    in holographic video microscopy images

    CATCH is an end-to-end convolutional neural network that
    analyzes normalized holograms of colloidal particle using
    predictions based on the Lorenz-Mie theory of light scattering.

    ...

    Properties
    ----------
    lmodel : Optional[str]
        Name of training model for localizing features
    emodel : Optional[str]
        Name of training model for estimating parameters

    Methods
    -------
    analyze(image: numpy.ndarray) : pandas.DataFrame
        Detects features in image and uses each feature to
        esimate the three-dimensional position, radius
        and refractive index of the associated colloidal particle.

    '''

    def __init__(self,
                 lmodel: Optional[str] = None,
                 emodel: Optional[str] = None) -> None:
        self.localizer = Localizer(model=lmodel)
        self.estimator = Estimator(model=emodel)

    def __call__(self, images: Images = []) -> pd.DataFrame:
        return self.analyze(images)

    def analyze(self, images: Images = []) -> pd.DataFrame:
        if not isinstance(images, list):
            images = [images]
        results = []
        featurelist = self.localizer(images)
        for n, (image, features) in enumerate(zip(images, featurelist)):
            if len(features) == 0:
                continue
            features['framenum'] = [n]*len(features)
            crops = crop_frame(image, features)
            estimates = self.estimator(crops)
            result = pd.concat([features, estimates], axis=1)
            results.append(result)
        return pd.concat(results) if results else pd.DataFrame()


def example():
    import cv2

    catch = CATCHapp()

    directory = catch.localizer.directory()
    image_file = directory / 'examples' / 'test_image_large.png'
    print(image_file)
    image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    results = catch(image)
    print(results)


if __name__ == '__main__':
    example()
