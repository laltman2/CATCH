import unittest

import numpy as np
import torch
import torch.nn.functional as F

from CATCH.lib.resample import resize


class TestResize(unittest.TestCase):

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.image = (1. + .1 * rng.standard_normal((40, 60))
                      ).astype(np.float32)

    def test_the_size_and_the_type(self) -> None:
        for shrink, size in ((True, (30, 17)), (False, (90, 71))):
            out = resize(self.image, size, shrink)
            self.assertEqual(out.shape, (size[1], size[0]))
            self.assertEqual(out.dtype, np.float32)

    def test_an_integer_factor_averages_blocks(self) -> None:
        out = resize(self.image, (30, 20), shrink=True)
        blocks = self.image.reshape(20, 2, 30, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(out, blocks, atol=1e-6)

    def test_shrinking_keeps_the_mean(self) -> None:
        for size in ((37, 23), (25, 16)):
            out = resize(self.image, size, shrink=True)
            self.assertAlmostEqual(float(out.mean()),
                                   float(self.image.mean()), places=4)

    def test_a_constant_stays_constant(self) -> None:
        flat = np.full((40, 60), 1.3, np.float32)
        for shrink, size in ((True, (37, 23)), (False, (91, 77))):
            np.testing.assert_allclose(resize(flat, size, shrink), 1.3,
                                       atol=1e-6)

    def test_enlarging_is_bicubic(self) -> None:
        out = resize(self.image, (90, 71), shrink=False)
        t = torch.from_numpy(self.image)[None, None].double()
        expected = F.interpolate(t, size=(71, 90), mode='bicubic',
                                 align_corners=False)[0, 0].numpy()
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_enlarging_follows_a_ramp(self) -> None:
        x = np.linspace(0., 1., 60, dtype=np.float32)
        ramp = np.tile(x, (40, 1))
        out = resize(ramp, (120, 40), shrink=False)
        self.assertTrue(np.all(np.diff(out[0, 4:-4]) > 0))
        self.assertAlmostEqual(float(out[0, 60]), .5, delta=.01)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
