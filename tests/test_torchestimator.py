import unittest

import torch

from CATCH.TorchEstimator import TorchEstimator


class TestTorchEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.net = TorchEstimator().eval()

    def test_forward_shape(self) -> None:
        image = torch.rand(2, 1, 201, 201)
        scale = torch.ones(2, 1)
        with torch.no_grad():
            out = self.net(image, scale)
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_freeze_and_unfreeze(self) -> None:
        self.net.freeze_shared()
        self.assertFalse(self.net.conv1.weight.requires_grad)
        self.assertFalse(self.net.dense1.bias.requires_grad)
        self.assertTrue(self.net.densez.weight.requires_grad)
        self.net.freeze_z()
        self.net.freeze_a()
        self.net.freeze_n()
        for layer in (self.net.outz, self.net.outa, self.net.outn):
            self.assertFalse(layer.weight.requires_grad)
        self.net.unfreeze_all()
        self.assertTrue(all(p.requires_grad for p in self.net.parameters()))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
