import torch
import os
import json
import numpy as np
import pandas as pd
from CATCH.training.torch_estimator_arch import TorchEstimator
from CATCH.training.ParamScale import ParamScale
import torchvision.transforms as trf
from typing import (Optional, List)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Estimator(object):

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
                 model_path: Optional[str]: None,
                 device: Optional[str]: None) -> None:
        self.model_path = model_path or self._default_path()
        self.device = torch.device(device or 'cpu')
        self.model = self._load_model()
        self.config = self._load_config()
        self.shape = tuple(self.config['shape'])
        self.transform = trf.Compose([trf.ToTensor(), trf.Resize(self.shape)
        self.model.eval()

    def _default_model(self) -> str:
        '''Returns path to Estimator model weights'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        data = f'{self.default_configuration}_checkpoints'
        path = (basedir, 'cfg_estimator', data, f'{self.weights}.pt')
        return os.path.join(*path)

    def _load_model(self) -> TorchEstimator:
        '''Returns CNN that estimates parameters from holograms'''
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = TorchEstimator()
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        if self.device != 'cpu':
            model.to(self.device)
        return model

    def _load_config(self) -> dict:
        '''Returns dictionary of model configuration parameters'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        cfg_name = self.configuration + '.json'
        path = (basedir, 'cfg_estimator', cfg_name)
        cfg_path = os.path.join(*path)
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        return config

    def load_image(self, image: np.ndarray) -> np.ndarray:
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
            image_list.append(self.load_image(image))
        scale = torch.tensor(scale_list).unsqueeze(1)
        image = torch.cat(image_list)

        if self.device != 'cpu':
            image = image.to(self.device)
            scale = scale.to(self.device)

        with torch.no_grad():
            # change image.float() to image to revert
            predictions = self.model(image=image.float(), scale=scale, **kwargs)
        keys = ['z_p', 'a_p', 'n_p']
        results = [{k: v.item() for k, v in zip(keys, self.scale(p))}
                   for p in predictions]
        return pd.DataFrame(results)


def example():
    import cv2

    est = Estimator()

    img_file = os.path.join('examples', 'test_image_crop.png')
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.rotate(img, cv2.ROTATE_180)
    results = est.predict([img])
    print(results)


if __name__ == '__main__':
    example()
