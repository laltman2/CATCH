from torch import nn
import torch


class TorchEstimator(nn.Module):
    def __init__(self):
        super().__init__()

        # out_names = ['z', 'a', 'n']
        # drop_rates = [0.005, 0.005, 0.005]
        # regularizer_rates = [0.3, 0.3, 0.3]
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
        self.dropout_z = nn.Dropout(0.1)
        self.outz = nn.Linear(dense_nodes[0], 1)
        self.outa = nn.Linear(dense_nodes[1], 1)
        self.outn = nn.Linear(dense_nodes[2], 1)

    def forward(self, image, scale):
        # inputs
        x1 = image
        x2 = scale

        # conv layers
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

        # split outputs
        z = self.relu(self.densez(x))
        z = self.dropout(z)
        z = self.outz(z)

        a = self.relu(self.densea(x))
        a = self.dropout(a)
        a = self.outa(a)

        n = self.relu(self.densen(x))
        n = self.dropout(n)
        n = self.outn(n)

        # outputs
        outputs = torch.cat((z, a, n), dim=1)
        return outputs

    def freeze_shared(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.dense1.weight.requires_grad = False
        self.dense1.bias.requires_grad = False

    def freeze_z(self):
        self.densez.weight.requires_grad = False
        self.densez.bias.requires_grad = False
        self.outz.weight.requires_grad = False
        self.outz.bias.requires_grad = False

    def freeze_a(self):
        self.densea.weight.requires_grad = False
        self.densea.bias.requires_grad = False
        self.outa.weight.requires_grad = False
        self.outa.bias.requires_grad = False

    def freeze_n(self):
        self.densen.weight.requires_grad = False
        self.densen.bias.requires_grad = False
        self.outn.weight.requires_grad = False
        self.outn.bias.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    import cv2
    from torchvision import transforms

    shape = (201, 201)
    tlist = [transforms.ToTensor(),
             transforms.Grayscale(num_output_channels=1),
             transforms.Resize(shape)]
    loader = transforms.Compose(tlist)

    net = TorchEstimator()

    img = cv2.imread('../examples/test_image_crop.png')
    img = loader(img).unsqueeze(0)

    scale = img.shape[0]/shape[0]
    scale = torch.tensor([scale]).unsqueeze(0)

    print(net(img, scale))
