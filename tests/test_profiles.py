'''Radial profiles measured from a frame'''
import unittest
from unittest.mock import patch

import numpy as np
import torch

from CATCH import profiles
from CATCH.networks import NRADII, RADII


def rings(shape, x, y, scale):
    '''An image whose value is the index of the nearest ring + 1'''
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.hypot(xx - x, yy - y) * scale
    return (np.rint(r / 2.) + 1.).astype(np.float32)


class TestProfiler(unittest.TestCase):

    def setUp(self) -> None:
        self.profiler = profiles.Profiler('cpu', chunk=2, compile=False)

    def test_shapes_and_values(self) -> None:
        image = rings((200, 240), 100.3, 90.6, 2.5)
        t = torch.from_numpy(image)
        b, m, c = self.profiler(t, [100.3, 120.], [90.6, 100.], 2.5)
        self.assertEqual(tuple(b.shape), (2, NRADII))
        # the median of a ring is the ring's own value: radius / 2 + 1
        good = m[0] > 0
        np.testing.assert_allclose(b[0][good][:20].numpy(),
                                   RADII[good.numpy()][:20] / 2 + 1)
        # rings beyond the frame are not measured, and count no pixels
        self.assertEqual(float(m[0, -1]), 0.)
        self.assertEqual(float(c[0, -1]), 0.)
        self.assertEqual(float(b[0, -1]), 1.)

    def test_counts_follow_the_circumference(self) -> None:
        image = np.ones((400, 400), np.float32)
        b, m, c = self.profiler(torch.from_numpy(image), [200.], [200.], 1.)
        self.assertAlmostEqual(float(c[0, 9]),
                               min(2 * np.pi * RADII[9], profiles.ANGLES),
                               places=2)
        self.assertTrue(torch.all(b[0][m[0] > 0] == 1))

    def test_a_center_outside_the_frame(self) -> None:
        image = np.ones((100, 100), np.float32)
        b, m, c = self.profiler(torch.from_numpy(image), [-5.], [50.], 1.)
        self.assertGreater(float(m.sum()), 0)

    def test_no_particles(self) -> None:
        image = torch.ones(50, 50)
        b, m, c = self.profiler(image, [], [], 1.)
        self.assertEqual(tuple(b.shape), (0, NRADII))

    def test_padding_for_a_compiled_sampler(self) -> None:
        image = torch.from_numpy(rings((120, 120), 60., 60., 1.))
        padded = profiles.Profiler('cpu', chunk=4, compile=False)
        def wrapped(flat, x, y, dx, dy, height, width):  # as if compiled,
            return profiles._sample(flat, x, y, dx, dy, height,  # so it
                                    width)                        # pads

        padded.sample = wrapped
        one = self.profiler(image, [60.], [60.], 1.)
        two = padded(image, [60.], [60.], 1.)
        for p, q in zip(one, two, strict=True):
            self.assertTrue(torch.equal(p, q))

    def test_compile_failure_falls_back(self) -> None:
        with patch.object(torch, 'compile', side_effect=RuntimeError), \
                self.assertWarns(UserWarning):
            p = profiles.Profiler('cpu')
        self.assertIs(p.sample, profiles._sample)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
