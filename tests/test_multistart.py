'''The criteria of success, fits from several starts, paired comparisons'''
import unittest
from importlib.util import find_spec
from typing import Any
from unittest.mock import patch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from CATCH.evaluation import (fitting, paired, pipelines,  # noqa: E402
                              references, scene_run, scenes)
from CATCH.evaluation.cases import INSTRUMENT  # noqa: E402

SHAPE = (256, 320)


def small_scenes(count=2):
    options: dict[str, Any] = dict(shape=SHAPE, radii=(1.,),
                                   indexes={'ps': 1.59},
                                   z_range=(250., 300.))
    return scenes.make_scenes(count, (1, 1), **options)


class TestCriteria(unittest.TestCase):

    def test_absolute_and_relative(self) -> None:
        self.assertTrue(fitting.locks_on(1.004, 1.))
        self.assertFalse(fitting.locks_on(1.006, 1.))
        # 5 nm is 0.1% of 5 um: 1% of the radius is much more lenient
        self.assertFalse(fitting.locks_on(5.02, 5.))
        self.assertTrue(fitting.locks_on(5.02, 5., relative=0.01))
        self.assertFalse(fitting.locks_on(1.02, 1., relative=0.01))

    def test_spread_start_keeps_the_scattering_strength(self) -> None:
        n_m = INSTRUMENT['n_m']
        start = dict(x_p=1., z_p=300., a_p=2., n_p=1.59)
        for factor in fitting.SPREAD:
            moved = fitting.spread_start(start, factor, n_m)
            self.assertAlmostEqual(moved['a_p'], 2. * factor)
            self.assertAlmostEqual((moved['n_p'] - n_m) * moved['a_p'],
                                   (1.59 - n_m) * 2.)
            self.assertEqual(moved['z_p'], 300.)
        self.assertEqual(start['a_p'], 2.)      # the start is not changed

    def test_the_first_start_is_the_estimate(self) -> None:
        self.assertEqual(fitting.SPREAD[0], 1.)
        self.assertEqual(len(set(fitting.SPREAD)), len(fitting.SPREAD))
        self.assertEqual(fitting.GRID[0], (1., 1.))
        self.assertEqual(len(fitting.GRID), 25)
        self.assertEqual(len(set(fitting.GRID)), 25)

    def test_the_grid_varies_the_axial_position(self) -> None:
        start = dict(x_p=1., z_p=300., a_p=2., n_p=1.59)
        moved = fitting.spread_start(start, 1., INSTRUMENT['n_m'], 1.3)
        self.assertAlmostEqual(moved['z_p'], 390.)
        self.assertEqual(moved['a_p'], 2.)
        zs = {z for _, z in fitting.GRID}
        self.assertEqual(zs, {1., 0.85, 1.15, 0.7, 1.3})


class TestStartRedchi(unittest.TestCase):

    def test_small_at_the_truth_and_large_far_from_it(self) -> None:
        near = scene_run.fit_job(self.job(1.0))
        far = scene_run.fit_job(self.job(1.5))
        self.assertLess(near['redchi0'], 2.)
        self.assertGreater(far['redchi0'], 5 * near['redchi0'])

    @staticmethod
    def job(factor):
        scene = small_scenes(1)[0]
        truth = scene.truth.iloc[0]
        detection = pd.Series(dict(x_p=truth.x_p, y_p=truth.y_p,
                                   z_p=truth.z_p, a_p=truth.a_p * factor,
                                   n_p=truth.n_p))
        job = scene_run.make_job(0, scene, detection)
        assert job is not None
        return job


class TestSeveralStarts(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.made = small_scenes(2)
        cls.results = scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), cls.made, starts=3)

    def test_columns_of_each_start(self) -> None:
        p = self.results.particles
        for i in range(3):
            self.assertIn(f'a_fit_{i}', p.columns)
            self.assertIn(f'redchi_{i}', p.columns)
        self.assertNotIn('a_fit_3', p.columns)
        for column in ('ok', 'close', 'ok_any', 'redchi0'):
            self.assertIn(column, p.columns)

    def test_the_fit_is_the_start_with_the_least_chi_squared(self) -> None:
        p = self.results.particles
        chis = p[[f'redchi_{i}' for i in range(3)]].to_numpy()
        np.testing.assert_allclose(p.redchi, chis.min(axis=1))

    def test_starts_from_the_truth_lock_on(self) -> None:
        p = self.results.particles
        self.assertTrue(p.ok.all())
        self.assertTrue(p.close.all())
        self.assertTrue(p.ok_any.all())

    def test_in_several_processes_gives_the_same(self) -> None:
        again = scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), self.made, nprocs=2, starts=3)
        for column in ('a_fit', 'redchi', 'a_fit_2', 'redchi0'):
            np.testing.assert_allclose(again.particles[column],
                                       self.results.particles[column])

    def test_one_start_has_no_columns_of_starts(self) -> None:
        one = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                        self.made)
        self.assertNotIn('a_fit_0', one.particles.columns)
        self.assertNotIn('ok_any', one.particles.columns)

    def test_a_grid_of_starts(self) -> None:
        grid = ((1., 1.), (1., 1.2), (1.1, 0.8))
        results = scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), self.made, starts=grid)
        p = results.particles
        self.assertIn('z_fit_2', p.columns)
        self.assertNotIn('a_fit_3', p.columns)
        self.assertTrue(p.ok_any.all())

    def test_the_grid_must_begin_with_the_estimate(self) -> None:
        for bad in (((1., 1.2),), ((1., 1.), (1., 1.)), ()):
            with self.assertRaises(ValueError):
                scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                          self.made, starts=bad)

    def test_number_of_starts(self) -> None:
        for starts in (0, len(fitting.SPREAD) + 1):
            with self.assertRaises(ValueError):
                scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                          self.made, starts=starts)

    def test_a_failed_fit_does_not_stop_the_others(self) -> None:
        bad = {'redchi': np.nan, 'redchi0': np.nan, 'a_p': np.nan,
               'x_p': 0., 'y_p': 0., 'z_p': 0., 'n_p': np.nan}
        good = dict(bad, redchi=2., redchi0=5., a_p=1.)
        combined = scene_run.combine([bad, good])
        self.assertEqual(combined['redchi'], 2.)
        self.assertTrue(np.isnan(combined['redchi0']))
        self.assertTrue(np.isnan(scene_run.combine([bad, bad])['redchi']))


class TestLoss(unittest.TestCase):
    '''Least squares, the robust Cauchy loss, and both in turn'''

    @classmethod
    def setUpClass(cls) -> None:
        cls.made = small_scenes(2)

    def run_loss(self, loss, **kwargs):
        return scene_run.evaluate_scenes(
            pipelines.truth_pipeline(), self.made, loss=loss, **kwargs)

    def test_every_loss_locks_on_from_the_truth(self) -> None:
        for loss in fitting.LOSSES:
            p = self.run_loss(loss).particles
            self.assertTrue(p.ok.all(), loss)

    def test_the_losses_are_different_fits(self) -> None:
        linear = self.run_loss('linear').particles
        cauchy = self.run_loss('cauchy').particles
        self.assertFalse(np.allclose(linear.redchi, cauchy.redchi,
                                     rtol=0, atol=1e-12))

    def test_both_is_a_robust_fit_from_the_least_squares_fit(self) -> None:
        both = self.run_loss('both').particles
        self.assertTrue(np.isfinite(both.redchi).all())

    def test_the_default_is_the_loss_of_the_scene_harness(self) -> None:
        self.assertEqual(fitting.SCENE_LOSS, 'both')
        default = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            self.made).particles
        both = self.run_loss('both').particles
        np.testing.assert_allclose(default.a_fit, both.a_fit)

    def test_the_single_particle_fit_is_least_squares(self) -> None:
        import inspect
        for function in (fitting.fit, fitting.fit_particle):
            default = inspect.signature(function).parameters['loss'].default
            self.assertEqual(default, 'linear')

    def test_in_several_processes_and_starts(self) -> None:
        one = self.run_loss('both', starts=2).particles
        two = self.run_loss('both', starts=2, nprocs=2).particles
        np.testing.assert_allclose(one.a_fit, two.a_fit)

    def test_unknown_loss(self) -> None:
        with self.assertRaises(ValueError):
            self.run_loss('huber')
        with self.assertRaises(ValueError):
            fitting.fit(None, None, loss='huber')


class TestBinnedByRadius(unittest.TestCase):

    def test_the_bins(self) -> None:
        results = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            small_scenes(2))
        table = scene_run.binned(results.particles, 'a_p')
        self.assertEqual(int(table.n.sum()), 2)
        self.assertIn('close', table.columns)
        self.assertIn('by a_p', scene_run.report(results))


def table(ok, inside=None):
    n = len(ok)
    return pd.DataFrame(dict(scene=np.arange(n) // 3, index=np.arange(n) % 3,
                             ok=ok,
                             inside=[True] * n if inside is None else inside))


class TestPaired(unittest.TestCase):

    def test_counts(self) -> None:
        a = table([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])
        b = table([1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
        result = paired.paired(a, b)
        self.assertEqual((result.n, result.both, result.neither,
                          result.a_only, result.b_only), (10, 4, 1, 1, 4))
        self.assertAlmostEqual(result.rate_a, 0.5)
        self.assertAlmostEqual(result.rate_b, 0.8)
        self.assertAlmostEqual(result.difference, 0.3)
        self.assertLess(result.low, result.difference)
        self.assertGreater(result.high, result.difference)
        self.assertAlmostEqual(result.pvalue, 0.375)   # exact, 1 against 4
        self.assertIn('B - A = +30.0%', str(result))

    def test_identical_versions(self) -> None:
        a = table([1, 0, 1, 0])
        result = paired.paired(a, a.copy())
        self.assertEqual(result.difference, 0.)
        self.assertEqual(result.pvalue, 1.)

    def test_a_clear_difference_is_significant(self) -> None:
        a = table([0] * 40)
        b = table([1] * 30 + [0] * 10)
        result = paired.paired(a, b)
        self.assertLess(result.pvalue, 1e-6)
        self.assertGreater(result.low, 0.)

    def test_particles_outside_the_image_are_left_out(self) -> None:
        inside = [True, True, True, False]
        a = table([1, 0, 0, 1], inside)
        b = table([1, 1, 0, 0], inside)
        self.assertEqual(paired.paired(a, b).n, 3)
        self.assertEqual(paired.paired(a, b, inside_only=False).n, 4)

    def test_other_scenes_are_refused(self) -> None:
        a = table([1, 0, 1])
        b = table([1, 0, 1]).assign(scene=7)
        with self.assertRaises(ValueError):
            paired.paired(a, b)


class TestReferences(unittest.TestCase):

    def test_the_analytic_estimate_has_the_interface(self) -> None:
        scene = small_scenes(1)[0]
        truth = scene.truth.iloc[0]
        crop, _, _ = pipelines.crop_around(scene.image, truth.x_p,
                                           truth.y_p, 301)
        found = references.pylorenzmie_estimator('analytic')([crop])
        self.assertEqual(list(found.columns), ['z_p', 'a_p', 'n_p'])
        self.assertEqual(len(found), 1)
        self.assertTrue(np.isfinite(found.to_numpy()).all())
        self.assertEqual(found.n_p[0], 1.5)     # it does not estimate this

    def test_an_estimate_that_fails_is_nan(self) -> None:
        with patch('pylorenzmie.analysis.Estimator.Estimator.estimate',
                   side_effect=ValueError):
            found = references.pylorenzmie_estimator('analytic')(
                [np.ones((9, 9))])
        self.assertEqual(found.shape, (1, 3))
        self.assertTrue(found.isna().all(axis=None))

    def test_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            references.pylorenzmie_estimator('guess')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
