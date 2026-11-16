# from CATCH.version import __version__
import yolov5
from packaging import version
from pathlib import Path
import numpy as np
import pandas as pd
from typing import (Optional, Union, List, Tuple, overload)

# suppress pytorch warnings about CUDA
# import warnings
# warnings.filterwarnings("ignore")


if version.parse(yolov5.__version__) > version.parse('6.0.7'):
    raise ImportError('Localizer requires yolov5.__version__ <= 6.0.7')


Images = Union[List[np.ndarray], np.ndarray]
Predictions = Union[List[pd.DataFrame], pd.DataFrame]


class Localizer(yolov5.YOLOv5):
    '''Find features of interest in holograms

    Parameters
    ----------
    model : str | None
        Name of the trained model.  Its weights are read from
        ``cfg_yolov5/<model>/weights.pt`` in the package directory.
        Default: None, which selects ``'small_noisy'``.
    device : str | None
        Computational device: ``'cpu'``, or ``'0'`` for a GPU.
        Default: None, which chooses automatically.
    threshold : float
        Features detected with less confidence are discarded.
        Default: 0.5

    Notes
    -----
    Requires ``yolov5 <= 6.0.7``.  ``detect`` is a synonym for
    :meth:`localize`.
    '''

    default_model = 'small_noisy'

    def __init__(self,
                 model: Optional[str] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5) -> None:
        self.model = model or self.default_model
        self.threshold = threshold
        self.detect = self.localize
        super().__init__(self._model_path(), device)

    @overload
    def __call__(self, images: np.ndarray) -> pd.DataFrame: ...

    @overload
    def __call__(self, images: List[np.ndarray]) -> List[pd.DataFrame]: ...

    def __call__(self, images: Images) -> Predictions:
        return self.localize(images)

    def directory(self) -> Path:
        '''Returns the directory of the CATCH package'''
        return Path(__file__).parent.resolve()

    def _model_path(self) -> Path:
        '''Returns path to model weights'''
        return self.directory() / 'cfg_yolov5' / self.model / 'weights.pt'

    def find_center(self, p: pd.Series,
                    shape: Tuple[int, ...]) -> Tuple[float, float, bool]:
        '''Returns center of feature corrected for edge crossing'''
        x_p, y_p = (p.xmax + p.xmin)/2., (p.ymax + p.ymin)/2.

        ext = max(int(p.ymax - p.ymin), int(p.xmax - p.xmin))
        h, w = shape[:2]

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

    @overload
    def localize(self, images: np.ndarray) -> pd.DataFrame: ...

    @overload
    def localize(self,
                 images: List[np.ndarray]) -> List[pd.DataFrame]: ...

    def localize(self, images: Images) -> Predictions:
        '''Detect and localize features in images

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray]
            Normalized holograms to analyze.  An image whose maximum is
            below 100 is multiplied by 100.

        Returns
        -------
        predictions : pandas.DataFrame | list[pandas.DataFrame]
            A DataFrame for an image, or a list of them for a list of
            images, with one row for each feature and the columns

            ``label``
                Class name of the feature.
            ``confidence``
                Confidence of the detection.
            ``x_p``, ``y_p``
                Coordinates of the center of the feature, in pixels.  The
                center of a feature that crosses an edge of the image is
                shifted to allow for the part that is cut off, and so may
                lie outside the image.
            ``bbox``
                ``((x_0, y_0), w, h)``: the corner of the bounding box with
                the smallest coordinates, and its width and height, in
                pixels.
            ``edge``
                True if the bounding box crosses an edge of the image.
        '''
        is_list = isinstance(images, list)
        if not is_list:
            images = [images]
        images = [x*100. if np.max(x) < 100 else x for x in images]
        size = np.max([np.max(image.shape) for image in images])
        results = self.predict(images, size=size).pandas().xyxy
        predictions = []
        for image, result in zip(images, results, strict=True):
            prediction = []
            for _, p in result.iterrows():
                if p.confidence < self.threshold:
                    continue
                w = int(p.xmax - p.xmin)
                h = int(p.ymax - p.ymin)
                bbox = ((int(p.xmin), int(p.ymin)), w, h)
                x_p, y_p, edge = self.find_center(p, image.shape)
                prediction.append({'label': p['name'],
                                   'confidence': p.confidence,
                                   'x_p': x_p,
                                   'y_p': y_p,
                                   'bbox': bbox,
                                   'edge': edge})
            predictions.append(pd.DataFrame(prediction))
        return predictions if is_list else predictions[0]


def example():  # pragma: no cover
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Create a Localizer
    localizer = Localizer()

    # Normalized hologram
    basedir = localizer.directory()
    image_file = str(basedir / 'docs' / 'images' / 'test_image_large.png')
    b = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if b is None:
        raise FileNotFoundError(
            f'{image_file}: the example images are in the repository, '
            'not in an installed package')
    print(image_file)

    # Use Localizer to identify features in the hologram
    features = localizer(b)
    print(features)

    # Show and report results
    style = dict(fill=False, linewidth=3, edgecolor='r')
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.imshow(b, cmap='gray')
    for bbox in features.bbox:
        ax.add_patch(Rectangle(*bbox, **style))
    plt.show()


if __name__ == '__main__':
    example()
