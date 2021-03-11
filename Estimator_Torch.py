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

    def __init__(self, configuration='test',
                 device = 'cpu'):

        self.device = device
        dev = torch.device(device)

        basedir = os.path.dirname(os.path.abspath(__file__))
        fname = configuration + '_checkpoints/best.pt'
        modelpath = os.path.join(basedir, 'cfg_estimator', fname)
        self.model = self.load_checkpoint(modelpath, dev)

        cfg_name = configuration + '.json'
        cfg_path = os.path.join(basedir, 'cfg_estimator', cfg_name)
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        self.config = config

        self.scaleparams = ParamScale(config)

    def load_checkpoint(self, filepath, dev):
        checkpoint = torch.load(filepath, map_location=dev)
        model = TorchEstimator()
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        if self.device != 'cpu':
            model.to(dev)

        model.eval()
        return model

    def predict(self, img_list=[]):
        self.model.eval()

        new_shape = self.config['shape'][0]
        scale_list = []
        for i in range(len(img_list)):
            img = img_list[i]
            if img.shape[0] != img.shape[1]:
                logger.warn('image crops must be square ... skipping')
            else:
                og_shape = img.shape[0]
                sc = og_shape / new_shape
                scale_list.append(sc)
                img_list[i] = cv2.resize(img, (new_shape, new_shape))

        tlist = [trf.ToTensor(), trf.Grayscale(num_output_channels=1)]
        loader = trf.Compose(tlist)
        image = [loader(x).unsqueeze(0) for x in img_list]
        image = torch.cat(image)
        
        scale = torch.tensor(scale_list).unsqueeze(1)

        if self.device != 'cpu':
            image = image.to(self.device)
            scale = scale.to(self.device)

        with torch.no_grad():
            pred = self.model(image=image, scale=scale)

        predictions = []
        for img in pred:
            outputs = self.scaleparams.unnormalize(img)
            z, a, n = [x.item() for x in outputs]
            predictions.append({'z_p':z, 'a_p':a, 'n_p':n})

        self.model.train() #unclear to me how necessary this is
        return predictions


if __name__ == '__main__':
    img = cv2.imread('./examples/test_image_crop.png')
    
    est = Estimator()
    results = est.predict(img_list = [img])[0]

    print(results)
