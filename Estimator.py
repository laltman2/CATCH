from CATCH.version import __version__
import torch
from CATCH.training.torch_estimator_arch import TorchEstimator
from CATCH.training.ParamScale import ParamScale
import torchvision.transforms as trf
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
from typing import (Optional, List)
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Estimator(TorchEstimator):

    default_configuration = 'autotrain_300p'
    default_weights = 'best'

    '''
    Estimate properties of scatterers from their holograms

    ...

    Properties
    ----------

    Methods
    -------

    '''

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        super().__init__()
        self.model_path = model_path or self._default_path()
        self.device = torch.device(device or 'cpu')
        self._load_model()
        self.config = self._load_config()
        self.shape = tuple(self.config['shape'])
        self.scale = ParamScale(self.config).unnormalize
        self.transform = trf.Compose([trf.ToTensor(),
                                      trf.Resize(self.shape)])
        self.eval()

    def _default_path(self) -> str:
        '''Returns path to Estimator model weights'''
        basedir = Path(__file__).parent.resolve()
        return str(basedir / 'cfg_estimator' /
                   f'{self.default_configuration}_checkpoints' /
                   f'{self.default_weights}.pt')

    def _load_model(self) -> None:
        '''Returns CNN that estimates parameters from holograms'''
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.device != 'cpu':
            self.to(self.device)

    def _load_config(self) -> dict:
        '''Returns dictionary of model configuration parameters'''
        config_path = re.sub(r'_check.*', r'.json', self.model_path)
        with open(config_path, 'r') as f:
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

    def predict(self, images: List[np.ndarray] = [], **kwargs) -> List:
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
            predictions = self(image=image.float(),
                               scale=scale, **kwargs)
        keys = ['z_p', 'a_p', 'n_p']
        results = [{k: v.item() for k, v in zip(keys, self.scale(p))}
                   for p in predictions]
        return pd.DataFrame(results)


def example():
    import cv2

    est = Estimator()

    basedir = Path(__file__).parent.resolve()
    img_file = str(basedir / 'examples' / 'test_image_crop.png')
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.rotate(img, cv2.ROTATE_180)
    results = est.predict([img])
    print(results)


if __name__ == '__main__':
    example()
