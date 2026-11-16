'''The evaluation harness, against the installed pylorenzmie'''
import unittest
from importlib.util import find_spec
from unittest.mock import patch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from CATCH.evaluation import cases, fitting, run  # noqa: E402


def oracle(case_list):
    '''An estimator that knows the answer, for a list of cases'''
    def estimate(crops):
        return pd.DataFrame([dict(z_p=c.z_p, a_p=c.a_p, n_p=c.n_p)
                             for c in case_list])
    return estimate


class TestCases(unittest.TestCase):

    def test_number_and_balance(self) -> None:
        made = cases.make_cases(24)
        self.assertEqual(len(made), 24)
        counts = pd.Series([(c.a_p, c.n_p) for c in made]).value_counts()
        self.assertEqual(set(counts), {2})      # 24 / (6 radii x 2 indexes)
        self.assertEqual(len(counts), 12)

    def test_remainder_is_dropped(self) -> None:
        self.assertEqual(len(cases.make_cases(25)), 24)

    def test_ranges(self) -> None:
        for c in cases.make_cases(60):
            self.assertTrue(150. <= c.z_p <= 600.)
            self.assertIn(c.offset, cases.OFFSETS)
            self.assertIn(c.a_p, cases.RADII)

    def test_reproducible(self) -> None:
        self.assertEqual(cases.make_cases(12), cases.make_cases(12))
        self.assertNotEqual(cases.make_cases(12), cases.make_cases(12, 1))

    def test_first_case_is_the_one_of_the_baseline(self) -> None:
        '''The cases of the handoff baseline (seed 2026) are kept'''
        first = cases.make_cases(180)[0]
        self.assertEqual((first.a_p, first.n_p, first.offset, first.seed),
                         (0.5, 1.59, 0.0, 687101529))
        self.assertAlmostEqual(first.z_p, 230.52066615394628)

    def test_custom_ranges(self) -> None:
        made = cases.make_cases(4, radii=(1.,), indexes={'x': 1.5},
                                z_range=(200., 210.), offsets=(7.,))
        self.assertEqual(len(made), 4)
        self.assertTrue(all(c.offset == 7. and 200. <= c.z_p <= 210.
                            for c in made))


class TestSimulate(unittest.TestCase):

    def setUp(self) -> None:
        self.case = cases.Case(a_p=1., n_p=1.59, z_p=300., offset=3.,
                               seed=5)

    def test_fixed_crop(self) -> None:
        crop = cases.simulate(self.case, 'fixed')
        self.assertEqual(crop.data.shape, (201, 201))
        self.assertEqual((crop.center, crop.noise), (100., 0.02))
        # normalized: the mean is 1, to within the noise and the fringes
        self.assertAlmostEqual(crop.data.mean(), 1., delta=0.05)

    def test_training_crop_is_sized_by_the_feature(self) -> None:
        crop = cases.simulate(self.case, 'training')
        side = crop.data.shape[0]
        self.assertEqual(crop.data.shape, (side, side))
        self.assertEqual(crop.noise, 0.05)
        # twice the extent, to within the 5% random error
        self.assertAlmostEqual(side / (2*cases.extent(self.case)), 1.,
                               delta=0.25)

    def test_reproducible(self) -> None:
        a = cases.simulate(self.case, 'fixed')
        b = cases.simulate(self.case, 'fixed')
        self.assertTrue(np.array_equal(a.data, b.data))

    def test_unknown_condition(self) -> None:
        with self.assertRaises(ValueError):
            cases.simulate(self.case, 'other')   # type: ignore[arg-type]


class TestFitting(unittest.TestCase):

    def test_locks_on(self) -> None:
        self.assertTrue(fitting.locks_on(1.004, 1.0))
        self.assertTrue(fitting.locks_on(0.996, 1.0))
        self.assertFalse(fitting.locks_on(1.006, 1.0))
        self.assertFalse(fitting.locks_on(float('nan'), 1.0))
        self.assertTrue(fitting.locks_on(1.05, 1.0, tolerance=0.1))

    def test_fit_from_the_truth_locks_on(self) -> None:
        case = cases.Case(a_p=1., n_p=1.59, z_p=300., offset=0., seed=1)
        crop = cases.simulate(case, 'fixed')
        job = fitting.Job(case, crop, (case.z_p, case.a_p, case.n_p))
        a_fit, n_fit, redchi = fitting.refine(job)
        self.assertTrue(fitting.locks_on(a_fit, case.a_p))
        self.assertAlmostEqual(n_fit, case.n_p, delta=0.01)
        self.assertLess(redchi, 1.5)

    def test_fit_from_a_near_start_locks_on(self) -> None:
        case = cases.Case(a_p=1., n_p=1.59, z_p=300., offset=0., seed=1)
        crop = cases.simulate(case, 'fixed')
        job = fitting.Job(case, crop, (case.z_p * 1.01, case.a_p * 1.01,
                                       case.n_p * 1.005))
        a_fit, *_ = fitting.refine(job)
        self.assertTrue(fitting.locks_on(a_fit, case.a_p))

    def test_a_failed_fit_is_nan(self) -> None:
        case = cases.Case(a_p=1., n_p=1.59, z_p=300., offset=0., seed=1)
        job = fitting.Job(case, cases.simulate(case, 'fixed'),
                          (300., 1., 1.59))
        with patch.object(fitting, 'fit', side_effect=RuntimeError):
            result = fitting.refine(job)
        self.assertTrue(all(np.isnan(v) for v in result))


class TestRun(unittest.TestCase):

    def setUp(self) -> None:
        self.cases = cases.make_cases(12)[::6]      # two cases

    def test_evaluate_with_the_truth(self) -> None:
        df = run.evaluate(oracle(self.cases), self.cases, 'fixed')
        self.assertEqual(len(df), 2)
        for column in ('a_p', 'z_est', 'a_fit', 'redchi', 'z_rel', 'all2',
                       'all5', 'ok', 'size', 'est_seconds', 'condition'):
            self.assertIn(column, df.columns)
        self.assertTrue((df.z_rel == 0).all() and df.all2.all())
        self.assertTrue(df.ok.all())
        self.assertEqual(set(df.condition), {'fixed'})

    def test_evaluate_in_several_processes(self) -> None:
        df = run.evaluate(oracle(self.cases), self.cases, 'fixed', nprocs=2)
        self.assertEqual(len(df), 2)
        self.assertTrue(df.ok.all())

    def test_the_estimator_gets_the_crops(self) -> None:
        seen = []

        def estimate(crops):
            seen.extend(crops)
            return oracle(self.cases)(crops)

        run.evaluate(estimate, self.cases, 'training')
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(c.shape[0] == c.shape[1] for c in seen))

    def test_score(self) -> None:
        df = pd.DataFrame(dict(
            z_p=[100., 100.], a_p=[1., 1.], n_p=[1.5, 1.5],
            z_est=[101., 120.], a_est=[1.01, 1.], n_est=[1.5, 1.5],
            a_fit=[1.001, float('nan')]))
        scored = run.score(df)
        self.assertEqual(list(scored.all2), [True, False])
        self.assertEqual(list(scored.all5), [True, False])
        self.assertEqual(list(scored.ok), [True, False])

    def test_summaries(self) -> None:
        df = run.evaluate(oracle(self.cases), self.cases, 'fixed')
        summary = run.summarize(df)
        self.assertEqual(summary.loc['locks_on', 'fixed'], 1.0)
        self.assertEqual(summary.loc['within2', 'fixed'], 1.0)
        radii = run.by_radius(df)
        self.assertEqual(list(radii.index), sorted(set(df.a_p)))
        self.assertTrue((radii['fixed'] == 1.0).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
