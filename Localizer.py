# from CATCH.version import __version__
import yolov5
from packaging import version
from pathlib import Path
import numpy as np
import pandas as pd
from typing import (Optional, Union, List, Tuple)

# suppress pytorch warnings about CUDA
import warnings


warnings.filterwarnings("ignore")


if version.parse(yolov5.__version__) > version.parse('6.0.7'):
    raise ImportError('Localizer requires yolov5.__version__ <= 6.0.7')


class Localizer(yolov5.YOLOv5):
    '''
    Find features of interest in holograms

    ...

    Properties
    __________
    model_path : str | None
        Name of trained configuration file
        default: None -- use standard hologram localizer
    device : str or None
        Computational device: 'cpu' or '0' for gpu
        default: None -- choose automatically
    threshold : float
        Confidence threshold for feature detection
        default: 0.5

    Methods
    _______
    localize(images: List[numpy.ndarray]) : list[pandas.DataFrame]
        Returns a DataFrame describing the features found
        in each input image.
    detect():
        Synonym for localize() for backward compatibility
    '''

    default_model = 'small_noisy'

    def __init__(self,
                 model: Optional[str] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5) -> None:
        self.model = model or self._default_model
        self.threshold = threshold
        self.predict = self.localize
        super().__init__(self._model_path(), device)

    def directory(self) -> str:
        '''Returns path to package directory'''
        return Path(__file__).parent.resolve()

    def _model_path(self) -> None:
        '''Returns path to configuration file'''
        return self.directory() / 'cfg_yolov5' / self.model / 'weights.pt'

    def find_center(self, p, shape) -> Tuple[float, float, bool]:
        '''Returns center of feature corrected for edge crossing'''
        x_p, y_p = (p.xmax + p.xmin)/2., (p.ymax + p.ymin)/2.

        ext = np.max([int(p.ymax - p.ymin), int(p.xmax - p.xmin)])
        h, w, _ = shape

        left, right = (p.xmax - ext < 0), (p.xmin + ext > w)
        bottom, top = (p.ymax - ext < 0), (p.ymin + ext > h)
        if left:
            x_p = p.xmax - ext/2.
        if right:
            x_p = p.xmin + ext/2.
        if bottom:
            y_p = p.ymax - ext/2.
        if top:
            y_p = p.ymin + ext/2.

        return x_p, y_p, left | right | bottom | top

    def localize(self,
                 images: Union[List[np.ndarray], np.ndarray] = []) -> List:
        '''Detect and localize features in an image

        Arguments
        ---------
        images: list[numpy.ndarray] | numpy.ndarray
           images to be analyzed

        Returns
        -------
        predictions: list[pandas.DataFrame] | pandas.DataFrame
            Each prediction describes the features in one image
            label: str
                Type of feature: Default: hologram
            x_p, y_p: float
                Coordinates of feature centroid
            bbox: ((x_0, y_0), w, h)
                Bounding box:
                x_0, y_0 : int
                    Lower-right corner
                w, h : int
                    Width and height of bounding box
            edge: bool
                True if bounding box crosses an edge of the image.
        '''
        is_list = isinstance(images, list)
        if not is_list:
            images = [images]
        images = [x*100. if np.max(x) < 100 else x for x in images]
        size = np.max([np.max(image.shape) for image in images])
        results = self.predict(images, size=size).pandas().xyxy
        predictions = []
        for image, result in zip(images, results):
            prediction = []
            for _, p in result.iterrows():
                if p.confidence < self.threshold:
                    continue
                w = int(p.xmax - p.xmin)
                h = int(p.ymax - p.ymin)
                bbox = ((int(p.xmin), int(p.ymin)), w, h)
                x_p, y_p, edge = self.find_center(p, image.shape)
                prediction.append({'label': p.name,
                                   'confidence': p.confidence,
                                   'x_p': x_p,
                                   'y_p': y_p,
                                   'bbox': bbox,
                                   'edge': edge})
            predictions.append(pd.DataFrame(prediction))
        return predictions if is_list else predictions[0]


def example():
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Create a Localizer
    localizer = Localizer()

    # Normalized hologram
    basedir = localizer.directory()
    img_file = str(basedir / 'examples' / 'test_image_large.png')
    print(img_file)
    b = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    # Use Localizer to identify features in the hologram
    features = localizer.localize(b)
    print(features)

    # Show and report results
    style = dict(fill=False, linewidth=3, edgecolor='r')
    matplotlib.use('Qt5Agg')
    fig, ax = plt.subplots()
    ax.imshow(b, cmap='gray')
    for bbox in features.bbox:
        ax.add_patch(Rectangle(*bbox, **style))
    plt.show()


if __name__ == '__main__':
    example()
