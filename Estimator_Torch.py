import torch
import os
import json
import numpy as np
from CATCH.training.torch_estimator_arch import TorchEstimator
from CATCH.training.ParamScale import ParamScale
import torchvision.transforms as trf

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Estimator(object):

    def __init__(self,
                 configuration='overlaps_fullrange',
                 device='cpu',
                 weights='best',
                 **kwargs):

        self.device = device
        self.configuration = configuration
        self.weights = weights

        self.model = self.load_model()
        self.config = self.load_config()
        self.scale = ParamScale(self.config).unnormalize
        self.shape = tuple(self.config['shape'])
        self.transform = trf.Compose([trf.ToTensor(),
                                      #trf.Grayscale(),
                                      trf.Resize(self.shape)])
        self.model.eval()

    def load_model(self):
        '''Returns CNN that performs regression on holograms'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        data = self.configuration + '_checkpoints'
        path = os.path.join(basedir, 'cfg_estimator', data, '{}.pt'.format(self.weights))
        dev = torch.device(self.device)
        checkpoint = torch.load(path, map_location=dev)

        model = TorchEstimator()
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        if self.device != 'cpu':
            model.to(dev)
        return model

    def load_config(self):
        '''Returns dictionary of model configuration parameters'''
        basedir = os.path.dirname(os.path.abspath(__file__))
        cfg_name = self.configuration + '.json'
        path = os.path.join(basedir, 'cfg_estimator', cfg_name)
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    def load_image(self, image):
        if image.shape[0] != image.shape[1]:
            logger.warn('image crops must be square')
        if np.max(image) < 100:
            image = image*100.
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        image = image[:,:,0]
        return self.transform(image).unsqueeze(0)
        
    def predict(self, images=[], **kwargs):
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
            predictions = self.model(image=image.float(), scale=scale, **kwargs) #change image.float() to image to revert
        keys = ['z_p', 'a_p', 'n_p']
        results = [{k: v.item() for k, v in zip(keys, self.scale(p))}
                   for p in predictions]
        return results


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    
    est = Estimator()

    img_file = os.path.join('examples', 'test_image_crop.png')
    #img = cv2.imread(img_file)[:,:,0]
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    results = est.predict([img])#, save_intermediate=True)
    
    print(results)
