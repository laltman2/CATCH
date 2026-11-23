'''The scan of the chi-squared around an estimate'''
import unittest
from importlib.util import find_spec

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np

from CATCH.customization.evaluation import pipelines, scan, scene_run, scenes
from CATCH.customization.evaluation.cases import INSTRUMENT
from CATCH.lib.fitter import crop_square

SMALL = scan.Scan(z_span=0.1, a_span=0.1, c_span=0.1, spacing=0.05,
                  keep=3, npixels=500)


class TestGrid(unittest.TestCase):

    def test_axis_has_the_estimate(self) -> None:
        factors = scan.axis(0.1, 0.025)
        self.assertEqual(len(factors), 9)
        self.assertAlmostEqual(factors[4], 1.)
        self.assertAlmostEqual(factors[0], 0.9)

    def test_local_minima(self) -> None:
        chi = np.full((5, 5, 5), 10.)
        chi[2, 2, 2] = 1.
        chi[0, 0, 0] = 2.            # a corner is a minimum too
        chi[4, 4, 4] = np.inf        # not computed
        low = scan.local_minima(chi)
        self.assertTrue(low[2, 2, 2])
        self.assertTrue(low[0, 0, 0])
        self.assertFalse(low[4, 4, 4])
        self.assertFalse(low[1, 1, 1])


class TestScan(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = scenes.make_scenes(
            1, (1, 1), shape=(256, 320), radii=(1.,), indexes={'ps': 1.59},
            z_range=(250., 300.))[0]
        truth = cls.scene.truth.iloc[0]
        cls.truth = truth
        values = np.array([truth.x_p, truth.y_p, truth.z_p, truth.a_p,
                           truth.n_p])
        side = max(int(2 * pipelines.extent(*values[2:])), 101)
        cls.crop, cls.x0, cls.y0 = crop_square(
            cls.scene.image, values[0], values[1], side)
        cls.start = dict(x_p=truth.x_p - cls.x0, y_p=truth.y_p - cls.y0,
                         z_p=truth.z_p, a_p=truth.a_p, n_p=truth.n_p)

    def test_the_best_cell_is_at_the_truth(self) -> None:
        starts, chi = scan.scan(self.crop, self.scene.noise, self.start,
                                SMALL, INSTRUMENT['n_m'],
                                INSTRUMENT['magnification'])
        self.assertEqual(chi.shape, (5, 5, 5))
        self.assertLessEqual(len(starts), 3)
        self.assertEqual(np.unravel_index(np.argmin(chi), chi.shape),
                         (2, 2, 2))
        self.assertAlmostEqual(starts[0]['a_p'], self.truth.a_p)
        self.assertLess(chi[2, 2, 2], 3.)

    def test_skips_points_inside_the_sphere(self) -> None:
        start = dict(self.start, z_p=30.)      # 1.4 um above the plane
        _, chi = scan.scan(self.crop, self.scene.noise, start, SMALL,
                           INSTRUMENT['n_m'], INSTRUMENT['magnification'])
        self.assertTrue(np.isinf(chi).any())


class TestSceneRun(unittest.TestCase):

    def test_fit_from_the_scan(self) -> None:
        made = scenes.make_scenes(
            1, (1, 1), shape=(256, 320), radii=(1.,), indexes={'ps': 1.59},
            z_range=(250., 300.))
        results = scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), made, scan=SMALL,
            only=lambda p: p['a_p'] > 0)
        p = results.particles
        self.assertTrue(np.isfinite(p.redchi).all())
        self.assertGreater(p.scan_seconds[0], 0.)
        self.assertIn('a_fit_1', p.columns)

    def test_only_skips_the_others(self) -> None:
        made = scenes.make_scenes(
            1, (1, 1), shape=(256, 320), radii=(1.,), indexes={'ps': 1.59},
            z_range=(250., 300.))
        results = scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), made, only=lambda p: False)
        self.assertTrue(np.isnan(results.particles.redchi).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
