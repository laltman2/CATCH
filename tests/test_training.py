'''Training: the frames, and short runs of both training commands'''
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from CATCH.Estimator import Estimator  # noqa: E402
from CATCH.Localizer import Localizer  # noqa: E402
from CATCH.networks import NRADII  # noqa: E402
from CATCH.training import estimator as TE  # noqa: E402
from CATCH.training import localizer as TL  # noqa: E402
from CATCH.training.frames import Frames, Ranges  # noqa: E402

SMALL = Ranges(shape=(128, 160), nparticles=(1, 3), a_p=(0.3, 1.),
               z_um=(10., 40.))


class TestFrames(unittest.TestCase):

    def test_frame(self) -> None:
        frames = Frames(1, SMALL)
        image, particles, scale = frames.frame()
        self.assertEqual(image.shape, SMALL.shape)
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(1. <= scale <= 2.5)
        self.assertTrue((particles.scale == scale).all())
        z_um = particles.z_p * 0.048 * scale
        self.assertTrue(((z_um >= 10.) | (z_um >= 1.2 * particles.a_p)).all())
        self.assertTrue(particles.x_p.between(0, 159).all())

    def test_fixed_scale_and_perfect_normalization(self) -> None:
        frames = Frames(2, Ranges(**{**SMALL.__dict__, 'imperfect': False,
                                     'nparticles': (0, 0)}))
        image, particles, scale = frames.frame(2.)
        self.assertEqual(scale, 2.)
        self.assertEqual(len(particles), 0)
        self.assertAlmostEqual(float(image.mean()), 1., delta=0.01)

    def test_imperfect(self) -> None:
        frames = Frames(3, SMALL)
        out = frames.imperfect(np.ones(SMALL.shape, np.float32))
        self.assertEqual(out.shape, SMALL.shape)

    def test_evaluation_seeds_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            Frames(2026, SMALL)


class TestLocalizerTraining(unittest.TestCase):

    def test_targets(self) -> None:
        heat, offset, mask = TL.targets([np.array([[10., 6.]])], (16, 24),
                                        torch.device('cpu'))
        self.assertEqual(tuple(heat.shape), (1, 4, 6))
        self.assertEqual(float(heat[0, 1, 2]), 1.)
        self.assertEqual(float(mask.sum()), 1.)
        self.assertAlmostEqual(float(offset[0, 0, 1, 2]), 0.5)
        self.assertAlmostEqual(float(offset[0, 1, 1, 2]), 0.5)

    def test_focal(self) -> None:
        target = torch.zeros(1, 4, 4)
        target[0, 1, 1] = 1.
        good = torch.full((1, 4, 4), -8.)
        good[0, 1, 1] = 8.
        self.assertLess(float(TL.focal(good, target)),
                        float(TL.focal(-good, target)))

    def test_short_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ranges = Ranges(**{**SMALL.__dict__, 'nparticles': (1, 2)})
            TL.train(d, 2, batch=2, every=2, ranges=ranges, nval=2,
                     device='cpu')
            for name in ('weights.pt', 'configuration.json', 'state.pt',
                         'eval.jsonl'):
                self.assertTrue((Path(d) / name).is_file(), name)
            log = json.loads((Path(d) / 'eval.jsonl').read_text())
            self.assertIn('scale 1', log)
            Localizer(d, device='cpu')          # a model that loads
            TL.train(d, 3, batch=2, every=10, ranges=ranges, nval=2,
                     device='cpu', init='default')  # resumes, not init
            self.assertEqual(torch.load(Path(d) / 'state.pt',
                                        weights_only=False)['step'], 3)

    def test_fine_tune_from_the_shipped_weights(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ranges = Ranges(**{**SMALL.__dict__, 'nparticles': (1, 1)})
            TL.train(d, 1, init='default', batch=1, every=5, ranges=ranges,
                     nval=1, device='cpu')
            config = json.loads((Path(d) / 'configuration.json').read_text())
            self.assertIn('from default', config['description'])


class TestEstimatorTraining(unittest.TestCase):

    def test_short_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            TE.train(d, 2, batch=8, every=2, library_frames=3,
                     validation_frames=2, shape=(128, 160), device='cpu')
            for name in ('weights.pt', 'configuration.json', 'state.pt',
                         'eval.jsonl', 'library.npz'):
                self.assertTrue((Path(d) / name).is_file(), name)
            library = np.load(Path(d) / 'library.npz')
            self.assertEqual(library['res'].shape[1], NRADII)
            Estimator(d, device='cpu', compile=False)
            TE.train(d, 3, batch=8, every=10, shape=(128, 160),
                     device='cpu', init='default')
            self.assertEqual(torch.load(Path(d) / 'state.pt',
                                        weights_only=False)['step'], 3)

    def test_fine_tune(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            TE.train(d, 1, init='default', batch=4, every=5,
                     library_frames=2, validation_frames=1,
                     shape=(128, 160), device='cpu')
            self.assertTrue((Path(d) / 'weights.pt').is_file())

    def test_loss_and_evaluate(self) -> None:
        p = torch.zeros(2, 4, 3)
        p[0, 1] = 5.
        t = torch.full((2, 3), 5.)
        reg, ce = TE.loss_of(p, torch.zeros(2, 4), t)
        self.assertGreater(float(reg), 0)
        val = dict(b=np.ones((2, NRADII), np.float32),
                   m=np.ones((2, NRADII), np.float32),
                   z=np.array([500., 3000.]), a=np.ones(2),
                   n=np.full(2, 1.5), scale=np.ones(2),
                   edge=np.full(2, 200.), neighbor=np.full(2, 300.))
        from CATCH.networks import ProfileNet
        out = TE.evaluate(ProfileNet(), val, torch.device('cpu'))
        self.assertIn('scale 1, 10-31 um', out)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
