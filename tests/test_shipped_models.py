'''The shipped weights, loaded and run on the example hologram

Every other test uses random weights (``random_models``).  This one loads
the models that CATCH ships, whose hashes ``Localizer`` and ``Estimator``
verify, and checks that they find and size the particle of the example
hologram, which a bad or swapped weights file would not.  The numbers are
those of the shipped models, with tolerances for the platform.
'''
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from CATCH.CATCHapp import CATCHapp

IMAGE = (Path(__file__).resolve().parent.parent / 'docs' / 'images' /
         'test_image_large.png')


@unittest.skipUnless(IMAGE.is_file(), 'the example image is not installed')
class TestShippedModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        image = np.asarray(Image.open(IMAGE).convert('L'))
        app = CATCHapp(device='cpu', compile=False)
        cls.found = app(image)

    def test_the_brightest_particle_is_found_and_sized(self) -> None:
        found = self.found
        self.assertGreaterEqual(len(found), 1)
        best = found.loc[found['confidence'].idxmax()]
        self.assertGreater(best['confidence'], 0.8)
        self.assertAlmostEqual(best['x_p'], 941.1, delta=3.)
        self.assertAlmostEqual(best['y_p'], 705.4, delta=3.)
        # about 1 um (a colloidal sphere of index near 1.5), well above the
        # focal plane of this instrument
        self.assertAlmostEqual(best['a_p'], 1.02, delta=0.15)
        self.assertAlmostEqual(best['n_p'], 1.53, delta=0.1)
        self.assertTrue(100. < best['z_p'] < 250.)
        self.assertGreater(best['snr'], 100.)
        self.assertTrue(np.isfinite(best['misfit']))
        self.assertFalse(best['edge'])

    def test_every_estimate_is_finite_and_in_range(self) -> None:
        for column in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'snr'):
            self.assertTrue(np.isfinite(self.found[column]).all(), column)
        self.assertTrue((self.found['a_p'] > 0.).all())
        self.assertTrue(self.found['n_p'].between(1.3, 2.6).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
