'''The Estimator: candidates checked against the profile, and the starts'''
import importlib
import tempfile
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import random_models
from CATCH.instruments import REFERENCE, pixel_ratio

E = importlib.import_module('CATCH.Estimator')
HAVE_PYLORENZMIE = find_spec('pylorenzmie') is not None
SHAPE = (480, 480)
TRUTH = dict(z=2500., a=1.0, n=1.55)          # reference pixels, um


def render(instrument=REFERENCE, z=TRUTH['z'], a=TRUTH['a'], n=TRUTH['n']):
    '''A frame with one particle at its center'''
    from pylorenzmie.theory import LorenzMie, Sphere
    model = LorenzMie()
    model.instrument.properties = dict(instrument)
    model.coordinates = model.meshgrid(SHAPE)
    model.particle = Sphere(x_p=240., y_p=240., z_p=z, a_p=a, n_p=n)
    return np.asarray(model.hologram()).reshape(SHAPE).astype(np.float32)


def candidates(rows):
    '''A replacement for ProfileNet.candidates: these, for every particle'''
    c = torch.tensor([[z, a, n - 1.34] for z, a, n in rows])

    def fake(features):
        return c[None].expand(features.shape[0], -1, -1).clone()
    return fake


class TestRing(unittest.TestCase):

    def test_ring_radius(self) -> None:
        r1 = E.ring_radius(2000., 1.)
        self.assertGreater(E.ring_radius(4000., 1.), r1)
        self.assertAlmostEqual(E.ring_radius(2000., 2.), r1 / 2)


class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.mkdtemp()
        _, epath = random_models.write_models(cls.directory)
        cls.estimator = E.Estimator(epath, device='cpu', compile=False)

    @classmethod
    def tearDownClass(cls) -> None:
        random_models.remove(cls.directory)

    def test_configuration(self) -> None:
        self.assertEqual(self.estimator.top, 3)
        self.assertEqual(self.estimator.standard, (0.0, 1.8))
        self.assertEqual(self.estimator.reference, dict(REFERENCE))

    def test_no_particles(self) -> None:
        out = self.estimator(np.ones(SHAPE), [], [])
        self.assertEqual(list(out.columns), list(E.COLUMNS))
        self.assertEqual(len(out), 0)
        self.assertEqual(len(self.estimator.misfits(
            np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0),
            None, None, None)), 0)

    def test_random_network_gives_starts(self) -> None:
        image = np.ones(SHAPE, np.float32) * 100.     # the 2.x scale
        out = self.estimator(image, [240.], [240.])
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out.starts[0], list)

    @unittest.skipUnless(HAVE_PYLORENZMIE, 'pylorenzmie is not installed')
    def test_the_truth_is_the_best_candidate(self) -> None:
        image = render()
        rows = [(TRUTH['z'] * f, TRUTH['a'] * g, TRUTH['n'])
                for f, g in ((1.1, 1.), (1., 1.15), (0.9, 0.9))]
        rows.insert(2, (TRUTH['z'], TRUTH['a'], TRUTH['n']))
        rows.append((TRUTH['z'], TRUTH['a'], 2.3))      # high index
        with patch.object(self.estimator.net, 'candidates',
                          side_effect=candidates(rows)):
            out = self.estimator(image, [240.], [240.])
        best = out.starts[0][0]
        self.assertAlmostEqual(best[0], TRUTH['z'], places=2)
        self.assertAlmostEqual(best[1], TRUTH['a'], places=4)
        self.assertEqual(out.z_p[0], best[0])
        self.assertLess(out.misfit[0], 1.)
        self.assertEqual(len(out.starts[0]), 4)     # top 3, 1 more inside

    @unittest.skipUnless(HAVE_PYLORENZMIE, 'pylorenzmie is not installed')
    def test_units_of_another_instrument(self) -> None:
        coarse = dict(REFERENCE, magnification=0.12)
        scale = pixel_ratio(coarse)
        image = render(coarse, TRUTH['z'] / scale)
        rows = [(TRUTH['z'], TRUTH['a'], TRUTH['n'])] * 3
        with patch.object(self.estimator.net, 'candidates',
                          side_effect=candidates(rows)):
            out = self.estimator(image, [240.], [240.], coarse)
        self.assertAlmostEqual(out.z_p[0], TRUTH['z'] / scale, places=2)
        self.assertAlmostEqual(out.a_p[0], TRUTH['a'], places=4)

    def test_scan_supplies_starts_in_the_preferred_range(self) -> None:
        image = np.ones(SHAPE, np.float32)
        rows = [(2000., 1., 2.2), (2100., 1.1, 2.3), (1900., 0.9, 2.4)]
        with patch.object(self.estimator.net, 'candidates',
                          side_effect=candidates(rows)):
            out = self.estimator(image, [240.], [240.])
            starts = out.starts[0]
            self.assertEqual(len(starts), 5)
            self.assertTrue(all(n <= 1.8 for _, _, n in starts[3:]))
            self.assertGreater(abs(starts[3][2] - starts[4][2]), 0.09)
            prior = self.estimator(image, [240.], [240.],
                                   nrange=(1.55, 1.65))
            self.assertTrue(all(1.55 <= n <= 1.65
                                for _, _, n in prior.starts[0][3:]))
            more = self.estimator(image, [240.], [240.],
                                  nrange=(1.55, 1.65), explore=3)
            extra = [a for _, a, _ in more.starts[0][3:]]
            self.assertEqual(len(extra), 3)
            self.assertTrue(all(abs(a / b - 1) > .03
                                for i, a in enumerate(extra)
                                for b in extra[:i]))

    def test_a_range_too_narrow_to_scan(self) -> None:
        image = np.ones(SHAPE, np.float32)
        rows = [(2000., 1., 2.2)] * 3
        with patch.object(self.estimator.net, 'candidates',
                          side_effect=candidates(rows)):
            out = self.estimator(image, [240.], [240.], nrange=(1.0, 1.3))
        self.assertEqual(len(out.starts[0]), 3)

    def test_candidates_inside_the_sphere_are_dropped(self) -> None:
        image = np.ones(SHAPE, np.float32)
        rows = [(10., 4., 1.5)] * 3        # z_p of 0.48 um: inside
        with patch.object(self.estimator.net, 'candidates',
                          side_effect=candidates(rows)):
            out = self.estimator(image, [240.], [240.])
        self.assertTrue(np.isnan(out.z_p[0]))
        self.assertEqual(out.starts[0], [])
        self.assertIsInstance(out, pd.DataFrame)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
