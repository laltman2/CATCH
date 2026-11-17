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


class TestOptions(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.image = torch.rand(2, 1, 201, 201)
        self.scale = torch.tensor([[0.5], [2.]])

    def run_net(self, net: TorchEstimator) -> torch.Tensor:
        with torch.no_grad():
            return net.eval()(self.image, self.scale)

    def test_default_is_unchanged(self) -> None:
        net = TorchEstimator()
        self.assertEqual(net.dense1.in_features, 401)
        self.assertEqual(net.dense1.out_features, 20)
        self.assertFalse({'scale_dense', 'film'} & set(net.state_dict()))

    def test_squeeze(self) -> None:
        net = TorchEstimator(squeeze=64)
        self.assertEqual(net.densez.in_features, 64)
        self.assertEqual(tuple(self.run_net(net).shape), (2, 3))

    def test_scale_nodes(self) -> None:
        net = TorchEstimator(scale_nodes=16)
        self.assertEqual(net.dense1.in_features, 416)
        self.assertEqual(tuple(self.run_net(net).shape), (2, 3))

    def test_film_starts_as_no_change(self) -> None:
        plain = TorchEstimator()
        film = TorchEstimator(film=True)
        film.load_state_dict(plain.state_dict(), strict=False)
        self.assertTrue(torch.allclose(self.run_net(plain),
                                       self.run_net(film)))

    def test_film_learns(self) -> None:
        net = TorchEstimator(film=True)
        before = self.run_net(net)
        torch.nn.init.normal_(net.film.weight, std=0.5)
        self.assertFalse(torch.allclose(before, self.run_net(net)))

    def test_freeze_keeps_the_options_trainable(self) -> None:
        net = TorchEstimator(scale_nodes=8, film=True)
        net.freeze_shared()
        self.assertFalse(net.scale_dense.weight.requires_grad)
        self.assertFalse(net.film.weight.requires_grad)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
