'''CATCHapp: the columns of the results, the boxes, and the optional fit'''
import importlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import random_models
from CATCH.lib.instruments import REFERENCE

A = importlib.import_module('CATCH.CATCHapp')


def features(*centers):
    return pd.DataFrame([dict(label='hologram', confidence=0.9, x_p=x,
                              y_p=y) for x, y in centers])


def estimates(rows):
    return pd.DataFrame([dict(z_p=z, a_p=a, n_p=n, misfit=0.1, snr=500.,
                              starts=[(z, a, n), (z * 1.1, a, n)])
                         for z, a, n in rows])


class TestCATCHapp(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.mkdtemp()
        lpath, epath = random_models.write_models(cls.directory)
        cls.app = A.CATCHapp(lpath, epath, device='cpu', compile=False)

    @classmethod
    def tearDownClass(cls) -> None:
        random_models.remove(cls.directory)

    def run_with(self, found, rows, images, **kwargs):
        with patch.object(self.app, 'localizer', return_value=found), \
             patch.object(self.app, 'estimator') as estimator:
            estimator.return_value = estimates(rows)
            estimator.reference = dict(REFERENCE)
            return self.app(images, **kwargs)

    def test_the_saturated_fraction_of_the_box(self) -> None:
        image = np.ones((400, 500))
        image[150:250, 200:300] = 2.55          # 100 x 100 saturated
        out = self.run_with(features((250., 200.), (10., 20.)),
                            [(2000., 1., 1.5), (20., 1., 1.5)], image,
                            saturation=2.55)
        # the large box covers the image, of which 100 x 100 is saturated
        self.assertAlmostEqual(out.saturated[0], 100 * 100 / (400 * 500))
        self.assertEqual(out.saturated[1], 0.)     # a small box in a corner
        none = self.run_with(features((250., 200.)), [(2000., 1., 1.5)],
                             image)
        self.assertEqual(none.saturated[0], 0.)    # no limit for a float

    def test_quality_says_why_not_to_trust_a_particle(self) -> None:
        self.assertEqual(A.quality_of(500., 0.), '')
        self.assertEqual(A.quality_of(A.LOW_SNR - 1, 0.), 'low snr')
        self.assertEqual(A.quality_of(500., A.MAX_SATURATED + .01),
                         'saturated')
        self.assertEqual(A.quality_of(10., 0.3), 'saturated, low snr')
        self.assertEqual(A.quality_of(float('nan'), 0.), 'no estimate')
        self.assertEqual(A.quality_of(A.LOW_SNR, A.MAX_SATURATED), '')

    def test_the_quality_column(self) -> None:
        image = np.ones((400, 500))
        with patch.object(self.app, 'localizer',
                          return_value=features((250., 200.), (100., 100.))), \
                patch.object(self.app, 'estimator') as estimator:
            rows = estimates([(2000., 1., 1.5), (20., 1., 1.5)])
            rows['snr'] = [500., 30.]
            estimator.return_value = rows
            estimator.reference = dict(REFERENCE)
            out = self.app(image)
        self.assertEqual(list(out.quality), ['', 'low snr'])

    def test_the_limit_of_the_camera(self) -> None:
        raw8 = np.full((400, 500), 100, np.uint8)
        self.assertEqual(A.saturation_of(raw8, None), 2.55)
        self.assertIsNone(A.saturation_of(raw8.astype(float), None))
        self.assertEqual(A.saturation_of(raw8, 1.8), 1.8)
        self.assertIsNone(A.saturation_of(raw8, np.inf))   # no limit

    def test_the_estimator_gets_the_limit(self) -> None:
        raw8 = np.full((400, 500), 100, np.uint8)
        with patch.object(self.app, 'localizer',
                          return_value=features((250., 200.))), \
                patch.object(self.app, 'estimator') as estimator:
            estimator.return_value = estimates([(2000., 1., 1.5)])
            estimator.reference = dict(REFERENCE)
            self.app(raw8)
            self.assertEqual(estimator.call_args.args[6], 2.55)
            self.app(raw8.astype(float))
            self.assertIsNone(estimator.call_args.args[6])
            self.app(raw8, saturation=np.inf)
            self.assertIsNone(estimator.call_args.args[6])

    def test_nothing(self) -> None:
        self.assertTrue(self.app.analyze(None).empty)
        self.assertTrue(self.run_with(features(), [],
                                      np.ones((50, 50))).empty)

    def test_columns_boxes_and_frames(self) -> None:
        image = np.ones((400, 500))
        out = self.run_with(features((250., 200.), (10., 20.)),
                            [(2000., 1., 1.5), (np.nan, np.nan, np.nan)],
                            [image, image])
        self.assertEqual(list(out.columns[:13]),
                         ['label', 'confidence', 'x_p', 'y_p', 'bbox',
                          'edge', 'saturated', 'snr', 'quality', 'z_p', 'a_p',
                          'n_p', 'framenum'])
        self.assertEqual(list(out.framenum), [0, 0, 1, 1])
        (x0, y0), w, h = out.bbox[0]
        self.assertEqual(w, h)
        self.assertGreater(w, 101)
        self.assertAlmostEqual(x0 + w / 2, 250., delta=1.)
        self.assertAlmostEqual(y0 + h / 2, 200., delta=1.)
        self.assertEqual(out.bbox[1][1], 101)      # no estimate
        self.assertTrue(out.edge[1])
        self.assertIn('starts', out.columns)

    def test_rings_set_the_size_of_the_box(self) -> None:
        image = np.ones((400, 500))
        sizes = {}
        for rings in (5, 10, 20):
            app = A.CATCHapp.__new__(A.CATCHapp)
            app.instrument, app.estimator = dict(REFERENCE), self.app.estimator
            app.rings = rings
            sizes[rings] = app.sides(pd.DataFrame(
                dict(z_p=[2000.], a_p=[1.], n_p=[1.5])))[0]
        self.assertLess(sizes[5], sizes[10])
        self.assertLess(sizes[10], sizes[20])
        self.assertEqual(self.app.rings, 10)       # the default
        self.assertEqual(self.app.npixels, 4000)
        del image

    def test_scale_follows_the_instrument(self) -> None:
        self.assertEqual(self.app.scale, 1.)
        app = A.CATCHapp.__new__(A.CATCHapp)
        app.instrument = dict(REFERENCE, magnification=0.12)
        app.estimator = self.app.estimator
        self.assertAlmostEqual(app.scale, 2.5)

    def test_fit(self) -> None:
        image = np.ones((400, 500))
        fit = dict(x_p=250., y_p=200., z_p=2000., a_p=1., n_p=1.5,
                   redchi=1.)

        class Fitter:
            def __init__(self, instrument, nprocs=1, npixels=4000):
                self.instrument, self.nprocs = dict(instrument), nprocs
                self.npixels = npixels

            def __call__(self, image, noise, particles, saturation=None):
                self.noise, self.saturation = noise, saturation
                return [[dict(fit), dict(fit, a_p=1.2, redchi=1.002)]
                        for _ in particles]

            def close(self):
                pass

        with patch('CATCH.lib.fitter.Fitter', Fitter):
            out = self.run_with(features((250., 200.)),
                                [(2000., 1., 1.5)], image, fit=True)
            self.assertEqual(out.a_fit[0], 1.)
            self.assertIsNone(self.app._fitter.saturation)
            self.run_with(features((250., 200.)), [(2000., 1., 1.5)],
                          image, fit=True, saturation=2.55)
            self.assertEqual(self.app._fitter.saturation, 2.55)
            self.assertEqual(out.nsolutions[0], 2)
            self.assertAlmostEqual(out.alt_a[0], 1.2)
            fitter = self.app._fitter
            self.run_with(features((250., 200.)), [(2000., 1., 1.5)],
                          image, fit=True, noise=0.02)
            self.assertIs(self.app._fitter, fitter)    # kept
            self.assertEqual(fitter.noise, 0.02)
            self.app.npixels = 8000                    # also a new one
            self.run_with(features((250., 200.)), [(2000., 1., 1.5)],
                          image, fit=True)
            self.assertEqual(self.app._fitter.npixels, 8000)
            self.app.npixels = 4000
            self.app.nprocs = 3                        # a new one
            self.run_with(features((250., 200.)), [(2000., 1., 1.5)],
                          image, fit=True)
            self.assertIsNot(self.app._fitter, fitter)
            self.app.nprocs = 1

    def test_whole_app_on_an_image(self) -> None:
        out = self.app(np.ones((128, 128)) * 100.)
        self.assertIsInstance(out, pd.DataFrame)


class TestNoise(unittest.TestCase):

    def test_noise_of(self) -> None:
        rng = np.random.default_rng(0)
        image = 1. + 0.05 * rng.normal(size=(300, 300))
        self.assertAlmostEqual(A.noise_of(image), 0.05, delta=0.003)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
