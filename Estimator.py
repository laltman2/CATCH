import torch
from CATCH.TorchEstimator import TorchEstimator
from CATCH.ParamScale import ParamScale
import torchvision.transforms as trf
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import (Optional, Union, List, Dict, cast)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


Images = Union[List[np.ndarray], np.ndarray]


class Estimator(TorchEstimator):
    '''Estimate properties of scatterers from their holograms

    Estimator is a convolutional neural network that is trained
    with the Lorenz-Mie theory of light scattering
    to estimate the radius (a_p), refractive index (n_p) and
    axial position (z_p) of a micrometer-scale colloidal sphere
    based on its in-line holographic microscopy image.

    Parameters
    ----------
    model : str | None
        Name of the model to load.  Its weights are read from
        ``cfg_estimator/<model>/weights.pt`` and its configuration from
        ``cfg_estimator/<model>/configuration.json`` in the package
        directory (see :meth:`directory`).
        Default: None, which selects ``'default'``.
    device : str | None
        Name of the Torch device that performs the computation,
        such as ``'cpu'`` or ``'cuda'``.
        Default: None, which selects ``'cpu'``.

    Notes
    -----
    ``predict`` is a synonym for :meth:`estimate`.
    '''

    default_model = 'default'

    def __init__(self,
                 model: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        super().__init__()
        self.model = model or self.default_model
        self.device = torch.device(device or 'cpu')
        self._load_model()
        self.config = self._load_config()
        self.shape = tuple(self.config['shape'])
        self.scale = ParamScale(self.config).unnormalize
        transforms = [trf.ToTensor(), trf.Resize(self.shape)]
        self.transform = trf.Compose(transforms)
        self.predict = self.estimate
        self.eval()

    def __call__(self, images: Images) -> pd.DataFrame:
        return self.estimate(images)

    def directory(self) -> Path:
        '''Returns the directory of the CATCH package'''
        return Path(__file__).parent.resolve()

    def _model_path(self) -> Path:
        '''Returns path to Estimator model weights'''
        return self.directory() / 'cfg_estimator' / self.model

    def _load_model(self) -> None:
        '''Returns CNN that estimates parameters from holograms'''
        weights = self._model_path() / 'weights.pt'
        checkpoint = torch.load(weights, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.device.type != 'cpu':
            self.to(self.device)

    def _load_config(self) -> Dict:
        '''Returns dictionary of model configuration parameters'''
        path = self._model_path() / 'configuration.json'
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    def _load_image(self, image: np.ndarray) -> torch.Tensor:
        '''Transforms image into form expected by CNN'''
        if image.shape[0] != image.shape[1]:
            logger.warning('image crops must be square')
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        if np.mean(image) != 100:
            image = np.clip(image/np.mean(image)*100., 0, 255)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = image[:, :, 0]
        # the transform is untyped, so say what it returns
        tensor = cast(torch.Tensor, self.transform(image))
        return tensor.unsqueeze(0)

    def estimate(self, images: Images, **kwargs) -> pd.DataFrame:
        '''Estimates particle properties for each cropped image

        Parameters
        ----------
        images : numpy.ndarray | list[numpy.ndarray]
            Square crops of a normalized hologram, each capturing one
            particle, such as those that ``CATCHapp.crop`` makes from the
            features that the Localizer finds.  A crop may have any size.

        Returns
        -------
        estimates : pandas.DataFrame
            One row for each crop, with columns ``z_p`` (pixels), ``a_p``
            (micrometers) and ``n_p``.
        '''
        if not isinstance(images, list):
            images = [images]
        scale_list, image_list = [], []
        for image in images:
            scale_list.append(image.shape[0]/self.shape[0])
            image_list.append(self._load_image(image))
        scale = torch.tensor(scale_list).unsqueeze(1)
        image = torch.cat(image_list)

        if self.device.type != 'cpu':
            image = image.to(self.device)
            scale = scale.to(self.device)

        with torch.no_grad():
            # change image.float() to image to revert
            estimates = super().__call__(image=image.float(),
                                         scale=scale,
                                         **kwargs)
        keys = ['z_p', 'a_p', 'n_p']
        results = [{k: v.item()
                    for k, v in zip(keys, self.scale(s), strict=True)}
                   for s in estimates]
        return pd.DataFrame(results)


def example():  # pragma: no cover
    import cv2

    estimator = Estimator()
    image_file = (estimator.directory() / 'docs' / 'images' /
                  'test_image_crop.png')
    print(image_file)
    image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_file)
    image = cv2.rotate(image, cv2.ROTATE_180)
    results = estimator(image)
    print(results)


if __name__ == '__main__':
    example()
