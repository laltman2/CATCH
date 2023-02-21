from CATCH.version import __version__
from CATCH.Localizer import Localizer
from CATCH.Estimator import Estimator
import numpy as np
import pandas as pd
from typing import (Optional, Union, List)


Images = Union[List[np.ndarray], np.ndarray]


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
            crops = self.crop(image, features)
            estimates = self.estimator(crops)
            result = pd.concat([features, estimates], axis=1)
            results.append(result)
        return pd.concat(results) if results else pd.DataFrame()

    @staticmethod
    def crop(image: np.ndarray, features: pd.DataFrame) -> List[np.ndarray]:

        def cropone(image: np.ndarray, feature: pd.Series) -> np.ndarray:
            xc, yc = int(round(feature.x_p)), int(round(feature.y_p))
            _, w, h = feature.bbox
            cropsize = max(w, h)
            right_top = int(np.ceil(cropsize/2.))
            left_bot = int(np.floor(cropsize/2.))
            x0, x1 = xc - left_bot, xc + right_top
            y0, y1 = yc - left_bot, yc + right_top
            if feature.edge:
                height, width = image.shape[:2]
                if x0 < 0:
                    x0, x1 = 0, cropsize
                if y0 < 0:
                    y0, y1 = 0, cropsize
                if x1 > width:
                    x0, x1 = width - cropsize, width
                if y1 > height:
                    y0, y1 = height - cropsize, height
            return image[y0:y1, x0:x1]

        if isinstance(features, pd.Series):
            return [cropone(image, features)]
        return [cropone(image, feature) for _, feature in features.iterrows()]


def example():
    import cv2

    # create a CATCH instance
    catch = CATCHapp()

    # read a normalized hologram image
    directory = catch.localizer.directory()
    image_file = str(directory / 'examples' / 'test_image_large.png')
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    print(image_file)

    # analyze the hologram and report the results
    results = catch(image)
    print(results)


if __name__ == '__main__':
    example()
