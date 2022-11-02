from CATCH.version import __version__
from CATCH.Localizer import Localizer
from CATCH.Estimator import Estimator
import numpy as np
import pandas as pd
from typing import (Optional, List)


def crop_frame(image: np.ndarray, features: pd.DataFrame) -> List:
    crops = []
    img_cols, img_rows = image.shape[:2]
    for n, feature in features.iterrows():
        xc, yc = map(lambda v: int(round(v)), [feature.x_p, feature.y_p])
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


class CATCH(object):

    def __init__(self,
                 localizer: Optional[Localizer] = None,
                 estimator: Optional[Estimator] = None) -> None:
        self.localizer = localizer or Localizer()
        self.estimator = estimator or Estimator()

    def analyze(self, images: List = []) -> pd.DataFrame:
        results = []
        featurelist = self.localizer.detect(images)
        for n, (image, features) in enumerate(zip(images, featurelist)):
            if len(features) == 0:
                continue
            crops = crop_frame(image, features)
            predictions = self.estimator.predict(crops)
            p_data = pd.DataFrame(predictions)
            f_data = pd.DataFrame(features)
            f_data['framenum'] = [n]*len(features)
            result = pd.concat([f_data, p_data], axis=1)
            results.append(result)
        return pd.concat(results) if results else pd.DataFrame()


def example():
    import os
    import cv2

    catch = CATCH()

    basedir = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(basedir, 'examples', 'test_image_large.png')
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    results = catch.analyze([img])
    print(results)


if __name__ == '__main__':
    example()
