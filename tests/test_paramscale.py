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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
