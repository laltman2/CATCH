import torch
import os, json
import numpy as np
from training.torch_estimator_arch import TorchEstimator
from training.Torch_DataLoader import ParamScale
from torchvision import transforms

class Estimator(object):

    def __init__(self, configuration='test',
                 device = 'cpu'):

        self.device = device
        dev = torch.device(device)

        basedir = os.path.dirname(os.path.abspath(__file__))
        fname = configuration + '.pt'
        modelpath = os.path.join(basedir, 'cfg_estimator', fname)
        self.model = TorchEstimator()
        self.model.load_state_dict(torch.load(modelpath))#, map_location=dev)

        cfg_name = configuration + '.json'
        cfg_path = os.path.join(basedir, 'cfg_estimator', cfg_name)
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        self.config = config

        self.scaleparams = ParamScale(config)

    def predict(self, img_list=[], scale_list=[]):
        self.model.eval()
        
        loader = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
        #image = loader(img_list)
        image = [loader(x).unsqueeze(0) for x in img_list]
        image = torch.cat(image)
        
        scale = torch.IntTensor(scale_list).unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(image = image, scale = scale)

        predictions = []
        for img in pred:
            outputs = self.scaleparams.unnormalize(img)
            z,a,n = outputs
            predictions.append({'z_p':z, 'a_p':a, 'n_p':n})

        self.model.train() #unclear to me how necessary this is
        return predictions


if __name__ == '__main__':
    import cv2

    img = cv2.imread('./examples/test_image_crop_201.png')
    
    est = Estimator(configuration='scale_float')
    results = est.predict(img_list = [img], scale_list = [1])

    print(results)
