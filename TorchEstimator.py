from torch import nn
import torch


class TorchEstimator(nn.Module):
    '''Convolutional network that estimates ``z_p``, ``a_p`` and ``n_p``

    The network analyzes a 201 x 201 normalized crop of a hologram, and the
    ratio of the crop's original size to 201, which is its second input.  It
    returns three numbers for each crop, scaled into [0, 1] (see
    :class:`~CATCH.ParamScale.ParamScale`).  The ``freeze_*`` methods hold
    parts of the network fixed for training.
    '''

    def __init__(self, squeeze: int = 20, scale_nodes: int = 0,
                 film: bool = False) -> None:
        super().__init__()
        self.build(squeeze, scale_nodes, film)

    def build(self, squeeze: int = 20, scale_nodes: int = 0,
              film: bool = False) -> None:
        '''Makes the layers (the defaults are the network of CATCH 2)

        Parameters
        ----------
        squeeze : int
            Width of the dense layer that all three outputs share.
            Default: 20
        scale_nodes : int
            If positive, the scale and its logarithm pass through a dense
            layer of this many units before they join the image features,
            so that they do not weigh as one number against hundreds.
            Default: 0 (the scale joins as it is)
        film : bool
            If True, the scale scales and shifts the channels of the
            convolutional layers (starting from no change).
            Default: False
        '''
        # out_names = ['z', 'a', 'n']
        # drop_rates = [0.005, 0.005, 0.005]
        # regularizer_rates = [0.3, 0.3, 0.3]
        dense_nodes = [20, 40, 100]
        self.options = dict(squeeze=squeeze, scale_nodes=scale_nodes,
                            film=film)

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        features = 400
        if scale_nodes:
            self.scale_dense = nn.Linear(2, scale_nodes)
            features += scale_nodes
        else:
            features += 1
        if film:
            # channels of the four convolutions (conv2 is used twice)
            self.film_channels = (32, 32, 32, 16)
            self.film = nn.Linear(2, 2 * sum(self.film_channels))
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.dense1 = nn.Linear(features, squeeze)
        self.densez = nn.Linear(squeeze, dense_nodes[0])
        self.densea = nn.Linear(squeeze, dense_nodes[1])
        self.densen = nn.Linear(squeeze, dense_nodes[2])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.01)
        self.dropout_z = nn.Dropout(0.1)
        self.outz = nn.Linear(dense_nodes[0], 1)
        self.outa = nn.Linear(dense_nodes[1], 1)
        self.outn = nn.Linear(dense_nodes[2], 1)

    def forward(self, image: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        '''Returns the scaled ``(z_p, a_p, n_p)`` for each image

        Parameters
        ----------
        image : torch.Tensor
            Batch of crops with shape ``(N, 1, 201, 201)``.
        scale : torch.Tensor
            Batch of crop-size ratios with shape ``(N, 1)``.
        '''
        # inputs
        x1 = image
        x2 = scale
        both = torch.cat((scale, torch.log(scale)), dim=1)
        modulation = iter(self.film(both).split(
            [c for c in self.film_channels for _ in (0, 1)], dim=1)) \
            if 'film' in self._modules else None

        def modulate(x: torch.Tensor) -> torch.Tensor:
            if modulation is None:
                return x
            gain, shift = next(modulation), next(modulation)
            return x * (1. + gain[:, :, None, None]) + shift[:, :, None, None]

        # conv layers
        x1 = modulate(self.conv1(x1))
        x1 = self.pool1(x1)
        x1 = modulate(self.conv2(x1))
        x1 = self.pool1(x1)
        x1 = modulate(self.conv2(x1))
        x1 = self.pool1(x1)
        x1 = modulate(self.conv3(x1))
        x1 = self.pool2(x1)
        x1 = torch.flatten(x1, start_dim=1)

        if 'scale_dense' in self._modules:
            x2 = self.relu(self.scale_dense(both))
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

    @staticmethod
    def _freeze(*layers: nn.Module) -> None:
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def freeze_shared(self) -> None:
        self._freeze(self.conv1, self.conv2, self.conv3, self.dense1,
                     *(m for k, m in self._modules.items()
                       if k in ('scale_dense', 'film') and m is not None))

    def freeze_z(self) -> None:
        self._freeze(self.densez, self.outz)

    def freeze_a(self) -> None:
        self._freeze(self.densea, self.outa)

    def freeze_n(self) -> None:
        self._freeze(self.densen, self.outn)

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


if __name__ == '__main__':  # pragma: no cover
    from typing import cast

    import cv2
    from torchvision import transforms

    shape = (201, 201)
    tlist = [transforms.ToTensor(),
             transforms.Grayscale(num_output_channels=1),
             transforms.Resize(shape)]
    loader = transforms.Compose(tlist)

    net = TorchEstimator()

    img = cv2.imread('../docs/images/test_image_crop.png')
    if img is None:
        raise FileNotFoundError('test_image_crop.png')
    tensor = cast(torch.Tensor, loader(img)).unsqueeze(0)

    scale = torch.tensor([tensor.shape[0]/shape[0]]).unsqueeze(0)

    print(net(tensor, scale))
