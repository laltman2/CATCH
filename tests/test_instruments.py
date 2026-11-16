import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import torch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import cv2  # noqa: E402
from pylorenzmie.theory import LorenzMie, Sphere  # noqa: E402

from CATCH.Estimator import Estimator  # noqa: E402
from CATCH.instruments import (REFERENCE, from_reference,  # noqa: E402
                               pixel_ratio, to_reference,
                               wavelength_in_medium)

FINE = {'wavelength': 0.532, 'magnification': 0.03, 'n_m': 1.33}
COARSE = {'wavelength': 0.532, 'magnification': 0.10, 'n_m': 1.0}
CONFIG = {'shape': [201, 201],
          'instrument': REFERENCE,
          'particle': {'z_p': [50, 600],
                       'a_p': [0.2, 5.0],
                       'n_p': [1.38, 2.5]}}


def make_estimator(**kwargs) -> Estimator:
    with patch.object(Estimator, '_load_model'), \
         patch.object(Estimator, '_load_config', return_value=CONFIG):
        return Estimator(**kwargs)


def render(instrument, a_p, n_p, z_p, size):
    '''Hologram of one particle at the center of a square image'''
    model = LorenzMie()
    model.instrument.properties = dict(instrument)
    model.coordinates = model.meshgrid((size, size))
    sphere = model.single(Sphere)
    sphere.a_p, sphere.n_p = a_p, n_p
    sphere.r_p = [(size - 1) / 2, (size - 1) / 2, z_p]
    return model.hologram().reshape(size, size)


class TestConversion(unittest.TestCase):

    def test_reference_is_the_identity(self) -> None:
        self.assertEqual(pixel_ratio(REFERENCE), 1.)
        self.assertEqual(from_reference(300., 1.5, 1.59, REFERENCE),
                         (300., 1.5, 1.59))

    def test_wavelength_in_medium(self) -> None:
        self.assertAlmostEqual(wavelength_in_medium(REFERENCE),
                               0.447 / 1.34)

    def test_pixel_ratio(self) -> None:
        # pixels of 0.03 um and a wavelength of 0.4 um in the medium
        # against 0.048 um and 0.3336 um
        self.assertAlmostEqual(pixel_ratio(FINE), 0.5212, places=3)
        self.assertGreater(pixel_ratio(COARSE), 1.)

    def test_round_trip(self) -> None:
        for instrument in (FINE, COARSE):
            there = from_reference(300., 1.5, 1.59, instrument)
            back = to_reference(*there, instrument)
            np.testing.assert_allclose(back, (300., 1.5, 1.59))

    def test_relative_index_is_kept(self) -> None:
        _, _, n_p = from_reference(300., 1.5, 1.59, COARSE)
        self.assertAlmostEqual(n_p / COARSE['n_m'], 1.59 / REFERENCE['n_m'])

    def test_lengths_scale_with_the_wavelength(self) -> None:
        other = dict(REFERENCE, wavelength=2 * REFERENCE['wavelength'])
        z_p, a_p, _ = from_reference(300., 1.5, 1.59, other)
        self.assertAlmostEqual(a_p, 3.)
        # the pixel is the same size and every length doubles
        self.assertAlmostEqual(z_p, 600.)


class TestEquivalence(unittest.TestCase):
    '''Holograms of the same particle, as two instruments see it, agree.'''

    def check(self, instrument, a_p, n_p, z_p, tolerance) -> None:
        size = 201
        ratio = pixel_ratio(instrument)
        ref_size = int(round(size * ratio))
        z_i, a_i, n_i = from_reference(z_p, a_p, n_p, instrument)
        reference = render(REFERENCE, a_p, n_p, z_p, ref_size)
        other = render(instrument, a_i, n_i, z_i, size)
        other = cv2.resize(other, (ref_size, ref_size),
                           interpolation=cv2.INTER_CUBIC)
        signal = np.sqrt(np.mean((reference - 1.) ** 2))
        error = np.sqrt(np.mean((reference - other) ** 2))
        self.assertLess(error / signal, tolerance)

    def test_finer_pixels_other_medium(self) -> None:
        self.check(FINE, 1.5, 1.59, 300., 0.03)

    def test_coarser_pixels_in_air(self) -> None:
        self.check(COARSE, 1.5, 1.59, 300., 0.08)

    def test_without_the_conversion_they_disagree(self) -> None:
        '''The test is able to fail: the particle itself is not the same.'''
        size = 201
        reference = render(REFERENCE, 1.5, 1.59, 300., size)
        other = render(FINE, 1.5, 1.59, 300., size)
        signal = np.sqrt(np.mean((reference - 1.) ** 2))
        error = np.sqrt(np.mean((reference - other) ** 2))
        self.assertGreater(error / signal, 0.3)


class TestEstimatorInstrument(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.crop = np.full((100, 100), 100, dtype=np.uint8)
        self.outputs = torch.tensor([[0.5, 0.5, 0.5]])

    def estimate(self, estimator):
        with patch.object(torch.nn.Module, '__call__',
                          return_value=self.outputs) as call:
            result = estimator.estimate(self.crop)
        return result.iloc[0], call

    def test_default_changes_nothing(self) -> None:
        plain, _ = self.estimate(make_estimator())
        same, _ = self.estimate(make_estimator(instrument=REFERENCE))
        self.assertEqual(plain.to_dict(), same.to_dict())

    def test_scale_and_estimates_follow_the_instrument(self) -> None:
        plain, call = self.estimate(make_estimator())
        other, call_other = self.estimate(make_estimator(instrument=COARSE))
        ratio = pixel_ratio(COARSE)
        scale = call.call_args.kwargs['scale'].item()
        scale_other = call_other.call_args.kwargs['scale'].item()
        self.assertAlmostEqual(scale_other, scale * ratio, places=5)
        expected = from_reference(plain.z_p, plain.a_p, plain.n_p, COARSE)
        np.testing.assert_allclose(
            (other.z_p, other.a_p, other.n_p), expected)

    def test_a_config_without_an_instrument_is_left_alone(self) -> None:
        config = {k: v for k, v in CONFIG.items() if k != 'instrument'}
        with patch.object(Estimator, '_load_model'), \
             patch.object(Estimator, '_load_config', return_value=config):
            estimator = Estimator(instrument=COARSE)
        self.assertEqual(estimator.pixel_ratio, 1.)


class TestCompare(unittest.TestCase):
    '''The comparison of the errors on two instruments'''

    def test_a_perfect_estimator_has_no_error(self) -> None:
        import pandas as pd

        from CATCH.evaluation.cases import extent, make_cases
        from CATCH.evaluation.equivalence import compare, summarize

        cases = make_cases(12, seed=1)
        for instrument in (REFERENCE, FINE):
            truth = iter([from_reference(c.z_p, c.a_p, c.n_p, instrument)
                          for c in cases])

            def estimate(crop, truth=truth) -> pd.DataFrame:
                z_p, a_p, n_p = next(truth)
                return pd.DataFrame([{'z_p': z_p, 'a_p': a_p, 'n_p': n_p}])

            errors = compare(estimate, instrument, cases)
            self.assertEqual(len(errors), len(cases))
            self.assertLess(errors[['da', 'dn', 'dz']].abs().max().max(),
                            1e-9)
            summary = summarize(errors)
            self.assertEqual(summary['a within 5%'], 1.)
        ratio = pixel_ratio(FINE)
        self.assertAlmostEqual(
            errors['size'].iloc[0], round(2 * extent(cases[0]) / ratio),
            delta=1)

    def test_the_crop_is_not_empty(self) -> None:
        from CATCH.evaluation.equivalence import simulate

        crop = simulate(FINE, 1.5, 1.59, 300., 64, seed=0)
        self.assertEqual(crop.shape, (64, 64))
        self.assertEqual(crop.dtype, np.uint8)
        self.assertGreater(crop.std(), 1.)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
