import torch
import os, json, cv2
import numpy as np
from CATCH.training.torch_estimator_arch import TorchEstimator
from CATCH.training.Torch_DataLoader import ParamScale
from torchvision import transforms as trf

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Estimator(object):

    def __init__(self,
                 configuration='test',
                 device='cpu'):

        self.device = device
        self.configuration = configuration

        self.model = self.load_model()
        self.config = self.load_config()
        self.shape = tuple(self.config['shape'])
        print(self.shape, type(self.shape))
        self.scaleparams = ParamScale(self.config)

    def load_model(self):
        '''Returns CNN that performs regression on holograms'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        data = self.configuration + '_checkpoints'
        path = os.path.join(basedir, 'cfg_estimator', data, 'best.pt')
        dev = torch.device(self.device)
        checkpoint = torch.load(path, map_location=dev)
        model = TorchEstimator()
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        if self.device != 'cpu':
            model.to(dev)

        model.eval()
        return model

    def load_config(self):
        '''Returns dictionary of model configuration parameters'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        cfg_name = self.configuration + '.json'
        path = os.path.join(basedir, 'cfg_estimator', cfg_name)
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    def rescale(self, image):
        if image.shape[0] != image.shape[1]:
            logger.warn('image crops must be square')
        return cv2.resize(image, self.shape)
        
    def predict(self, images=[]):
        scale_list = [image.shape[0]/self.shape[0] for image in images]
        scale = torch.tensor(scale_list).unsqueeze(1)
        
        images = list(map(self.rescale, images))
        transforms = [trf.ToTensor(), trf.Grayscale(num_output_channels=1)]
        loader = trf.Compose(transforms)
        image_list = [loader(image).unsqueeze(0) for image in images]
        image = torch.cat(image_list)
        
        if self.device != 'cpu':
            image = image.to(self.device)
            scale = scale.to(self.device)

        with torch.no_grad():
            predictions = self.model(image=image, scale=scale)

        results = []
        keys = ['z_p', 'a_p', 'n_p']
        for prediction in predictions:
            values = self.scaleparams.unnormalize(prediction)
            results.append({k: v.item() for k, v in zip(keys, values)})
        return results


if __name__ == '__main__':
    est = Estimator()

    # Read hologram (not normalized)
    img_file = os.path.join('examples', 'test_image_crop.png')
    img = cv2.imread(img_file)
    #img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    results = est.predict([img])[0]

    print(results)
