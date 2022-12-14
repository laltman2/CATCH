from CATCH.version import __version__
import torch
from CATCH.training.torch_estimator_arch import TorchEstimator
from CATCH.training.ParamScale import ParamScale
import torchvision.transforms as trf
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import (Optional, List)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Estimator(TorchEstimator):
    '''Estimate properties of scatterers from their holograms

    Estimator is a convolutional neural network that is trained
    with the Lorenz-Mie theory of light scattering
    to estimate the radius (a_p), refractive index (n_p) and
    axial position (z_p) of a micrometer-scale colloidal sphere
    based on its in-line holographic microscopy image.

    ...

    Inherits
    --------
    torch.nn.Module

    Properties
    ----------
    model: Optional[str]
        Name of light-scattering model to load
        Weights for the trained convolutional neural network should be
        stored in DIRECTORY/cfg_estimator/MODEL/weights.pt
        where DIRECTORY can be obtained with the directory() method.
        The configuration used for training should be stored in
        DIRECTORY/cfg_estimator/MODEL/config.json

        Default: 'default'

    device: Optional[str]
        Name of the Torch device used to perform computation
        'cpu' or 'gpu'

        Default: 'cpu'

    Methods
    -------
    directory(): str
        Fully qualified path to this package

    estimate(images: List[numpy.ndarray]): pandas.DataFrame
        Estimates z_p, a_p and n_p for each cropped image
        in the list of images.

    predict:
        Synonym for estimate for backward compatibility
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

    def __call__(self, images):
        return self.estimate(images)

    def directory(self) -> str:
        '''Returns path to this file'''
        return Path(__file__).parent.resolve()

    def _model_path(self) -> str:
        '''Returns path to Estimator model weights'''
        return self.directory() / 'cfg_estimator' / self.model

    def _load_model(self) -> None:
        '''Returns CNN that estimates parameters from holograms'''
        weights = self._model_path() / 'weights.pt'
        checkpoint = torch.load(weights, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.device != 'cpu':
            self.to(self.device)

    def _load_config(self) -> dict:
        '''Returns dictionary of model configuration parameters'''
        config = self._model_path() / 'configuration.json'
        with open(config, 'r') as f:
            config = json.load(f)
        return config

    def _load_image(self, image: np.ndarray) -> np.ndarray:
        '''Transforms image into form expected by CNN'''
        if image.shape[0] != image.shape[1]:
            logger.warn('image crops must be square')
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        if np.mean(image) != 100:
            image = np.clip(image/np.mean(image)*100., 0, 255)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = image[:, :, 0]
        return self.transform(image).unsqueeze(0)

    def estimate(self, images: List[np.ndarray] = [],
                 **kwargs) -> pd.DataFrame:
        '''Estimates particle properties for each cropped image

        Arguments
        ---------
        images: List[numpy.ndarray]
            List of cropped images, each capturing one particle
            This list can be obtained by applying Localizer
            to a normalized hologram.

        '''
        scale_list, image_list = [], []
        for image in images:
            scale_list.append(image.shape[0]/self.shape[0])
            image_list.append(self._load_image(image))
        scale = torch.tensor(scale_list).unsqueeze(1)
        image = torch.cat(image_list)

        if self.device != 'cpu':
            image = image.to(self.device)
            scale = scale.to(self.device)

        with torch.no_grad():
            # change image.float() to image to revert
            estimates = super().__call__(image=image.float(),
                                         scale=scale,
                                         **kwargs)
        keys = ['z_p', 'a_p', 'n_p']
        results = [{k: v.item() for k, v in zip(keys, self.scale(s))}
                   for s in estimates]
        return pd.DataFrame(results)


def example():
    import cv2

    estimator = Estimator()
    image_file = estimator.directory() / 'examples' / 'test_image_crop.png'
    image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    image = cv2.rotate(image, cv2.ROTATE_180)
    results = estimator([image])
    print(results)


if __name__ == '__main__':
    example()
