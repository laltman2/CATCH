'''The two networks: shapes, decoding of the detector, the profile inputs'''
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CATCH.lib import networks as N


class TestDetector(unittest.TestCase):

    def test_inputs(self) -> None:
        images = torch.ones(2, 64, 80)
        x = N.detector_inputs(images)
        self.assertEqual(tuple(x.shape), (2, 3, 64, 80))
        self.assertTrue(torch.all(x[:, 0] == 0))      # the background
        self.assertEqual(float(x[0, 1, 0, 0]), 0.)    # at the edge
        self.assertEqual(float(x[0, 1, 0, 30]), 30 / N.EDGE_SCALE)

    def test_outputs(self) -> None:
        net = N.DetectorNet().eval()
        with torch.no_grad():
            heat, offset = net(torch.ones(1, 64, 96))
        self.assertEqual(tuple(heat.shape), (1, 16, 24))
        self.assertEqual(tuple(offset.shape), (1, 2, 16, 24))
        self.assertTrue(torch.all((offset >= 0) & (offset <= 1)))
        self.assertEqual(N.grid_shape((64, 96)), (16, 24))

    def test_peaks_decode_the_offsets(self) -> None:
        heat = torch.full((1, 8, 10), -10.)
        heat[0, 3, 5] = 5.
        offset = torch.zeros(1, 2, 8, 10)
        offset[0, 0, 3, 5], offset[0, 1, 3, 5] = 0.25, 0.75
        net = N.DetectorNet()
        with patch.object(net, 'forward', return_value=(heat, offset)):
            x, y, s = N.peaks(net, torch.ones(1, 32, 40), count=3)
        self.assertEqual(x.shape, (1, 3))
        self.assertAlmostEqual(float(x[0, 0]), (5 + 0.25) * N.STRIDE)
        self.assertAlmostEqual(float(y[0, 0]), (3 + 0.75) * N.STRIDE)
        self.assertGreater(s[0, 0], 0.99)
        self.assertLess(s[0, 1], 0.01)


class TestProfileNet(unittest.TestCase):

    def test_features(self) -> None:
        b = torch.ones(3, N.NRADII) * 1.1
        m = torch.ones(3, N.NRADII)
        m[:, -10:] = 0
        f = N.profile_features(b, m)
        self.assertEqual(tuple(f.shape), (3, 4, N.NRADII))
        self.assertAlmostEqual(float(f[0, 0, 0]), 0.3, places=5)
        self.assertEqual(float(f[0, 1, 0]), 3.)       # clipped
        self.assertTrue(torch.all(f[:, :3, -10:] == 0))
        self.assertTrue(torch.all(f[:, 3] == m))

    def test_candidates(self) -> None:
        net = N.ProfileNet().eval()
        with torch.no_grad():
            c = net.candidates(torch.zeros(2, 4, N.NRADII))
        self.assertEqual(tuple(c.shape), (2, N.NCANDIDATES, 3))
        self.assertTrue(torch.all(c > 0))
        self.assertEqual(len(N.RADII), N.NRADII)
        np.testing.assert_allclose(N.RADII[:2], [2., 4.])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
