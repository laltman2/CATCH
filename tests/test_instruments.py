'''Converting between instruments, and the physics behind it'''
import unittest
from importlib.util import find_spec

import numpy as np

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import cv2  # noqa: E402
from pylorenzmie.theory import LorenzMie, Sphere  # noqa: E402

from CATCH.lib.instruments import (REFERENCE, from_reference,  # noqa: E402
                               pixel_ratio, to_reference,
                               wavelength_in_medium)

FINE = {'wavelength': 0.532, 'magnification': 0.03, 'n_m': 1.33}
COARSE = {'wavelength': 0.532, 'magnification': 0.10, 'n_m': 1.0}
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
