from CATCH.Localizer import Localizer
from CATCH.Estimator import Estimator
import numpy as np
import pandas as pd
from typing import (Optional, Union, List)


Images = Union[List[np.ndarray], np.ndarray]
Features = Union[List[pd.DataFrame], pd.DataFrame]


class CATCHapp(object):
    '''Detect, localize and characterize colloidal particles
    in holographic video microscopy images.

    CATCH analyzes normalized holograms of colloidal particles with a
    detector (the Localizer) and a convolutional neural network (the
    Estimator) that is trained with the Lorenz-Mie theory of light
    scattering.

    Parameters
    ----------
    lmodel : str | None
        Name of the trained model that localizes features.
        Default: None, which selects the default Localizer model.
    emodel : str | None
        Name of the trained model that estimates parameters.
        Default: None, which selects the default Estimator model.

    Attributes
    ----------
    localizer : Localizer
        Finds features in images.
    estimator : Estimator
        Estimates the properties of the particle in each feature.
    '''

    def __init__(self,
                 lmodel: Optional[str] = None,
                 emodel: Optional[str] = None) -> None:
        self.localizer = Localizer(model=lmodel)
        self.estimator = Estimator(model=emodel)

    def __call__(self, images: Optional[Images] = None) -> pd.DataFrame:
        return self.analyze(images)

    def analyze(self, images: Optional[Images] = None) -> pd.DataFrame:
        '''Detect the features in images and characterize their particles

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray] | None
            Normalized hologram, or list of normalized holograms.
            Default: None, which returns an empty DataFrame.

        Returns
        -------
        results : pandas.DataFrame
            One row for each detected feature, with the columns that the
            Localizer reports (``label``, ``confidence``, ``x_p``,
            ``y_p``, ``bbox``, ``edge``), the estimates ``z_p`` (pixels),
            ``a_p`` (micrometers) and ``n_p``, and ``framenum``, the
            index of the image in ``images``.
        '''
        if images is None:
            return pd.DataFrame()
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return pd.DataFrame()
        results = []
        featurelist = self.localizer(images)
        pairs = zip(images, featurelist, strict=True)
        for n, (image, features) in enumerate(pairs):
            if len(features) == 0:
                continue
            features['framenum'] = [n]*len(features)
            crops = self.crop(image, features)
            estimates = self.estimator(crops)
            result = pd.concat([features, estimates], axis=1)
            results.append(result)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    @staticmethod
    def crop(image: np.ndarray, features: pd.DataFrame) -> List[np.ndarray]:
        '''Crop a square around each feature

        Parameters
        ----------
        image : numpy.ndarray
            Image that contains the features.
        features : pandas.DataFrame
            Features found by the Localizer.  The side of each crop is the
            larger of the width and the height of the bounding box, and a
            feature that crosses an edge of the image is shifted inward.

        Returns
        -------
        crops : list[numpy.ndarray]
            One crop for each row of ``features``.
        '''

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

        return [cropone(image, feature) for _, feature in features.iterrows()]


def example():  # pragma: no cover
    import cv2

    # create a CATCH instance
    catch = CATCHapp()

    # read a normalized hologram image
    directory = catch.localizer.directory()
    image_file = str(directory / 'docs' / 'images' / 'test_image_large.png')
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f'{image_file}: the example images are in the repository, '
            'not in an installed package')
    print(image_file)

    # analyze the hologram and report the results
    results = catch(image)
    print(results)


if __name__ == '__main__':
    example()
