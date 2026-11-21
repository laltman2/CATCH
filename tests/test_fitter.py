'''Fits from the starts of the Estimator'''
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np

from CATCH import fitter

HAVE_PYLORENZMIE = find_spec('pylorenzmie') is not None
COARSE = dict(wavelength=0.447, magnification=0.12, n_m=1.34)


def render(shape=(201, 201), x=100.3, y=99.6, z=600., a=0.75, n=1.5,
           instrument=COARSE, noise=0.01, seed=0):
    from pylorenzmie.theory import LorenzMie, Sphere
    model = LorenzMie()
    model.instrument.properties = dict(instrument)
    model.coordinates = model.meshgrid(shape)
    model.particle = Sphere(x_p=x, y_p=y, z_p=z, a_p=a, n_p=n)
    image = np.asarray(model.hologram()).reshape(shape)
    return image + noise * np.random.default_rng(seed).normal(size=shape)


class SerialPool:
    '''Stands in for a process pool'''

    def __init__(self, nprocs):
        self.nprocs = nprocs

    def map(self, f, tasks):
        return map(f, tasks)

    def shutdown(self):
        pass


class TestCrop(unittest.TestCase):

    def test_crop_square(self) -> None:
        image = np.arange(100 * 200).reshape(100, 200)
        crop, x0, y0 = fitter.crop_square(image, 100., 50., 20)
        self.assertEqual(crop.shape, (20, 20))
        self.assertEqual((x0, y0), (90, 40))
        crop, x0, y0 = fitter.crop_square(image, 3., 97., 20)
        self.assertEqual((x0, y0), (0, 80))          # shifted inward
        crop, x0, y0 = fitter.crop_square(image, 100., 50., 500)
        self.assertEqual(crop.shape, (100, 100))     # no larger than image

    def test_unknown_loss(self) -> None:
        with self.assertRaises(ValueError):
            fitter.Fitter(COARSE, loss='square')


@unittest.skipUnless(HAVE_PYLORENZMIE, 'pylorenzmie is not installed')
class TestFits(unittest.TestCase):

    def test_fit_particle_with_an_instrument(self) -> None:
        image = render()
        start = dict(x_p=101., y_p=99., z_p=620., a_p=0.78, n_p=1.48)
        out = fitter.fit_particle(image, 0.01, start, 'both',
                                  instrument=COARSE)
        self.assertAlmostEqual(out['a_p'], 0.75, delta=0.005)
        self.assertAlmostEqual(out['x_p'], 100.3, delta=0.1)
        self.assertLess(out['redchi'], out['redchi0'])

    def test_fitter_on_a_frame(self) -> None:
        image = render((260, 300), x=150.3)
        starts = [(620., 0.78, 1.48), (560., 0.70, 1.52)]
        f = fitter.Fitter(COARSE, nprocs=2)
        with patch.object(fitter, 'ProcessPoolExecutor', SerialPool):
            fits = f(image, 0.01, [(150.3, 99.6, 201, starts),
                                   (150.3, 99.6, 151, starts[:1])])
        self.assertEqual([len(x) for x in fits], [2, 1])
        self.assertAlmostEqual(fits[0][0]['a_p'], 0.75, delta=0.005)
        self.assertAlmostEqual(fits[0][0]['x_p'], 150.3, delta=0.1)
        self.assertIsNotNone(f._pool)             # kept for the next frame
        f.close()
        self.assertIsNone(f._pool)

    def test_pixel_budget(self) -> None:
        image = render()
        start = dict(x_p=101., y_p=99., z_p=620., a_p=0.78, n_p=1.48)
        seen = []
        from pylorenzmie.analysis import Mask

        def spy(*args, **kwargs):
            seen.append(kwargs['fraction'])
            return Mask(*args, **kwargs)

        with patch('pylorenzmie.analysis.Mask', spy):
            fitter.fit_particle(image, 0.01, start, 'linear',
                                instrument=COARSE, npixels=2000)
            fitter.fit_particle(image, 0.01, start, 'linear',
                                instrument=COARSE)
        self.assertAlmostEqual(seen[0], 2000 / image.size, places=4)
        self.assertAlmostEqual(seen[-1], fitter.NPIXELS / image.size,
                               places=4)

    def test_the_fitter_passes_its_settings(self) -> None:
        f = fitter.Fitter(COARSE, npixels=1234)
        self.assertEqual((f.npixels, f.side), (1234, None))
        self.assertEqual(fitter.Fitter(COARSE, side=300).side, 300)
        image = render((260, 300), x=150.3)
        seen = []
        def fake(*args, **kwargs):
            seen.append((args[0].shape, kwargs['npixels']))
            return dict(x_p=0., y_p=0.)

        with patch.object(fitter, 'fit_particle', side_effect=fake):
            f(image, 0.01, [(150.3, 99.6, 2000, [(620., 0.78, 1.48)])])
        self.assertEqual(seen, [((260, 260), 1234)])   # no cap but the frame

    def test_a_failed_fit_is_nan(self) -> None:
        with patch.object(fitter, 'fit', side_effect=RuntimeError):
            out = fitter.fit_particle(render(), 0.01, dict(
                x_p=100., y_p=100., z_p=600., a_p=0.75, n_p=1.5))
        self.assertTrue(np.isnan(out['a_p']))

    def test_unknown_loss_of_a_fit(self) -> None:
        with self.assertRaises(ValueError):
            fitter.fit(None, None, loss='square')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
