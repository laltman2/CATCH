import unittest

from CATCH.ParamScale import ParamScale

CONFIG = {'particle': {'z_p': [50, 600],
                       'a_p': [0.2, 5.0],
                       'n_p': [1.38, 2.5]}}


class TestParamScale(unittest.TestCase):

    def setUp(self) -> None:
        self.scale = ParamScale(CONFIG)

    def test_ranges(self) -> None:
        self.assertEqual(self.scale.ranges,
                         [[50, 600], [0.2, 5.0], [1.38, 2.5]])

    def test_normalize_endpoints(self) -> None:
        self.assertEqual(self.scale.normalize([50, 0.2, 1.38]),
                         [0., 0., 0.])
        self.assertEqual(self.scale.normalize([600, 5.0, 2.5]),
                         [1., 1., 1.])

    def test_unnormalize_midpoint(self) -> None:
        z, a, n = self.scale.unnormalize([0.5, 0.5, 0.5])
        self.assertAlmostEqual(z, 325.)
        self.assertAlmostEqual(a, 2.6)
        self.assertAlmostEqual(n, 1.94)

    def test_round_trip(self) -> None:
        params = [123., 1.7, 1.59]
        back = self.scale.unnormalize(self.scale.normalize(params))
        for x, y in zip(params, back, strict=True):
            self.assertAlmostEqual(x, y)

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.scale.normalize([1., 2.])


class TestLogScaling(unittest.TestCase):

    def setUp(self) -> None:
        config = dict(CONFIG, instrument={'n_m': 1.34},
                      scaling={'a_p': 'log', 'n_p': 'log_contrast'})
        self.scale = ParamScale(config)

    def test_endpoints(self) -> None:
        for low, high in (([50, 0.2, 1.38], [600, 5.0, 2.5]),):
            for got, want in zip(self.scale.normalize(low), (0, 0, 0),
                                 strict=True):
                self.assertAlmostEqual(got, want)
            for got, want in zip(self.scale.normalize(high), (1, 1, 1),
                                 strict=True):
                self.assertAlmostEqual(got, want)

    def test_equal_relative_steps(self) -> None:
        a = self.scale.normalize([100, 0.5, 1.5])[1]
        b = self.scale.normalize([100, 1.0, 1.5])[1]
        c = self.scale.normalize([100, 2.0, 1.5])[1]
        self.assertAlmostEqual(b - a, c - b)

    def test_round_trip(self) -> None:
        params = [123., 1.7, 1.59]
        back = self.scale.unnormalize(self.scale.normalize(params))
        for x, y in zip(params, back, strict=True):
            self.assertAlmostEqual(x, y)

    def test_tensors(self) -> None:
        import torch
        z, a, n = (torch.tensor([300., 400.]), torch.tensor([1., 3.]),
                   torch.tensor([1.5, 2.]))
        back = self.scale.unnormalize(self.scale.normalize([z, a, n]))
        for x, y in zip((z, a, n), back, strict=True):
            self.assertTrue(torch.allclose(x, y, atol=1e-5))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
