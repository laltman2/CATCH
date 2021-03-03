import json
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F



#configfile='keras_train_config.json'
#with open(configfile, 'r') as f:
#    config = json.load(f)


class TorchEstimator(nn.Module):
    def __init__(self, config={}):
         super(TorchEstimator, self).__init__()

         out_names = ['z', 'a', 'n']
         drop_rates = [0.005, 0.005, 0.005]
         regularizer_rates = [0.3, 0.3, 0.3]
         dense_nodes = [20, 40, 100]
         
         self.conv1 = nn.Conv2d(1, 32, 3)
         self.conv2 = nn.Conv2d(32, 32, 3)
         self.conv3 = nn.Conv2d(32, 16, 3)
         self.pool1 = nn.MaxPool2d(2,2)
         self.pool2 = nn.MaxPool2d(4,4)
         self.dense1 = nn.Linear(401, 20)
         self.densez = nn.Linear(20, dense_nodes[0])
         self.densea = nn.Linear(20, dense_nodes[1])
         self.densen = nn.Linear(20, dense_nodes[2])
         self.relu = nn.ReLU()
         self.dropout = nn.Dropout(0.01)
         self.outz = nn.Linear(dense_nodes[0], 1)
         self.outa = nn.Linear(dense_nodes[1], 1)
         self.outn = nn.Linear(dense_nodes[2], 1)
         

    def forward(self, image, scale):
        #inputs
        x1 = image
        x2 = scale

        #conv layers
        x1 = self.conv1(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.pool1(x1)
        x1 = self.conv3(x1)
        x1 = self.pool2(x1)
        x1 = torch.flatten(x1, start_dim=1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.dense1(x))
        
        #split outputs
        z = self.relu(self.densez(x))
        z = self.dropout(z)
        z = self.outz(z)
        
        a = self.relu(self.densea(x))
        a = self.dropout(a)
        a = self.outa(a)
        
        n = self.relu(self.densen(x))
        n = self.dropout(n)
        n = self.outn(n)

        #outputs
        outputs = torch.cat((z, a, n), dim=1)
        return outputs
        


if __name__ == '__main__':
    import cv2
    from torchvision import transforms
    from torch.autograd import Variable

    imsize = 201
    loader = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    
    net = TorchEstimator()

    img = loader(cv2.imread('../examples/test_image_crop_201.png')).unsqueeze(0)
    print(img.shape)

    scale = torch.IntTensor([1]).unsqueeze(0)
    print(net(img, scale))
    
    
