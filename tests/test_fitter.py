'''Fits from the starts of the Estimator'''
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np

from CATCH.lib import fitter

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

    def test_crop_square_is_shifted_inside_at_the_edges(self) -> None:
        image = np.arange(100 * 200, dtype=float).reshape(100, 200)
        for x, y, x0, y0 in ((2, 50, 0, 40), (199, 50, 180, 40),
                             (100, 1, 90, 0), (100, 99, 90, 80),
                             (-30, -30, 0, 0)):
            crop, cx, cy = fitter.crop_square(image, x, y, 20)
            self.assertEqual(crop.shape, (20, 20))
            self.assertEqual((cx, cy), (x0, y0))
        crop, x0, y0 = fitter.crop_square(image, 100, 50, 20)
        self.assertEqual(crop[0, 0], image[40, 90])

    def test_unknown_loss(self) -> None:
        with self.assertRaises(ValueError):
            fitter.Fitter(COARSE, loss='square')

    def test_a_fitter_is_a_context_manager(self) -> None:
        image = render((260, 300))
        particles = [(100., 99., 101, [(620., 0.78, 1.48)])] * 2
        with patch.object(fitter, 'ProcessPoolExecutor', SerialPool), \
                patch.object(fitter, 'fit_particle',
                             return_value=dict(x_p=0., y_p=0.)), \
                fitter.Fitter(COARSE, nprocs=2) as f:
            f(image, 0.01, particles)
            self.assertIsNotNone(f._pool)          # kept for the next frame
        self.assertIsNone(f._pool)
        f.close()                                     # twice is fine
        broken = fitter.Fitter.__new__(fitter.Fitter)  # __init__ did not run
        broken.close()
        del broken


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

    def test_a_corrupted_patch_is_left_out_by_a_mask_or_a_level(self) -> None:
        image = render()
        bad = np.zeros(image.shape, bool)
        bad[90:110, 90:110] = True             # the core, as if saturated
        corrupted = image.copy()
        corrupted[bad] = 5.
        start = dict(x_p=101., y_p=99., z_p=620., a_p=0.78, n_p=1.48)
        plain = fitter.fit_particle(corrupted, 0.01, start, 'both',
                                    instrument=COARSE)
        masked = fitter.fit_particle(corrupted, 0.01, start, 'both',
                                     instrument=COARSE, saturation=bad)
        by_level = fitter.fit_particle(corrupted, 0.01, start, 'both',
                                       instrument=COARSE, saturation=5.)
        self.assertGreater(abs(plain['a_p'] - 0.75), 0.02)
        for fit in (masked, by_level):
            self.assertAlmostEqual(fit['a_p'], 0.75, delta=0.01)
            self.assertAlmostEqual(fit['x_p'], 100.3, delta=0.2)
        # the mask and the level leave out the same pixels
        self.assertAlmostEqual(masked['a_p'], by_level['a_p'], places=6)

    def test_the_fitter_crops_the_mask_with_the_image(self) -> None:
        image = render((260, 300), x=150.3)
        mask = np.zeros(image.shape, bool)
        mask[100:110, 140:160] = True
        seen = []

        def fake(*args, **kwargs):
            seen.append((args[0].shape, kwargs['saturation']))
            return dict(x_p=0., y_p=0.)

        f = fitter.Fitter(COARSE)
        with patch.object(fitter, 'fit_particle', side_effect=fake):
            f(image, 0.01, [(150.3, 99.6, 101, [(620., 0.78, 1.48)]),
                            (20., 99.6, 101, [(620., 0.78, 1.48)])], mask)
            f(image, 0.01, [(150.3, 99.6, 101, [(620., 0.78, 1.48)])], 2.5)
        (shape1, m1), (shape2, m2), (_, level) = seen
        self.assertEqual(shape1, (101, 101))
        self.assertEqual(m1.shape, shape1)
        self.assertEqual(int(m1.sum()), 200)        # all of it is inside
        self.assertEqual(m2.shape, shape2)          # shifted inward
        self.assertEqual(int(m2.sum()), 0)
        self.assertEqual(level, 2.5)               # a level goes as it is

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

    def test_a_failed_fit_is_nan_with_a_warning(self) -> None:
        start = dict(x_p=100., y_p=100., z_p=600., a_p=0.75, n_p=1.5)
        for error in (RuntimeError('too bad'), ZeroDivisionError('zero'),
                      ValueError('x0 is infeasible'), OverflowError('big')):
            with patch.object(fitter, 'fit', side_effect=error), \
                    self.assertWarnsRegex(RuntimeWarning, 'a fit failed'):
                out = fitter.fit_particle(render(), 0.01, start)
            self.assertTrue(np.isnan(out['a_p']))
            self.assertTrue(np.isnan(out['redchi0']))

    def test_a_mistake_of_the_program_is_not_hidden(self) -> None:
        start = dict(x_p=100., y_p=100., z_p=600., a_p=0.75, n_p=1.5)
        for error in (TypeError('wrong type'), KeyError('z_p'),
                      AttributeError('no such thing')):
            with patch.object(fitter, 'fit', side_effect=error), \
                    self.assertRaises(type(error)):
                fitter.fit_particle(render(), 0.01, start)

    def test_unknown_loss_of_a_fit(self) -> None:
        with self.assertRaises(ValueError):
            fitter.fit(None, None, loss='square')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
