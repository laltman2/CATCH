import json
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from CNNLorenzMie.Estimator import format_image


configfile='keras_train_config.json'
with open(configfile, 'r') as f:
    config = json.load(f)


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
        x1 = torch.flatten(x1)

        
        x = torch.cat((x1, x2), dim=0)
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
        return z, a, n
        

#Image dimensions
#_, input_shape = format_image(np.array([]), config['shape'])


if __name__ == '__main__':
    import PIL.Image as Image
    from torchvision import transforms
    from torch.autograd import Variable

    imsize = 201
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image#.cuda()  #assumes that you're using GPU
    
    net = TorchEstimator()

    img = image_loader('../examples/test_image_crop_201.png')

    scale = torch.ByteTensor(1)
    print(net(img, scale))
    
    
