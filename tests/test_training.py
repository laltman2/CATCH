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
from CATCH.lib.networks import NRADII  # noqa: E402
from CATCH.customization.training import estimator as TE  # noqa: E402
from CATCH.customization.training import localizer as TL  # noqa: E402
from CATCH.customization.training.frames import Frames, Ranges  # noqa: E402

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

    def test_a_run_records_the_hash_of_its_weights(self) -> None:
        shipped = json.loads((Path(__file__).resolve().parent.parent
                              / 'models/estimator/default/configuration.json'
                              ).read_text())['weights_sha256']
        with tempfile.TemporaryDirectory() as d:
            TE.train(d, 1, init='default', batch=4, every=5,
                     library_frames=2, validation_frames=1,
                     shape=(128, 160), device='cpu')
            config = json.loads((Path(d) / 'configuration.json').read_text())
            from CATCH.Localizer import weights_digest
            self.assertEqual(config['weights_sha256'],
                             weights_digest(Path(d) / 'weights.pt'))
            # not the hash of the shipped model that it started from
            self.assertNotEqual(config['weights_sha256'], shipped)

    def test_the_hash_of_the_shipped_model_is_not_inherited(self) -> None:
        from CATCH.customization.training import common
        with tempfile.TemporaryDirectory() as d:
            common.configure(Path(d), 'estimator', 5, None)
            config = json.loads((Path(d) / 'configuration.json').read_text())
        self.assertNotIn('weights_sha256', config)
        self.assertIn('5 steps', config['description'])

    def test_a_camera_does_not_measure_the_saturated_rings(self) -> None:
        profile = torch.tensor([[1., 2., 2.55, 3., 1.2]])
        mask = torch.tensor([[1., 1., 1., 1., 0.]])
        out = TE.saturate(profile, mask)
        self.assertEqual(out.tolist(), [[1., 1., 0., 0., 0.]])
        self.assertEqual(TE.saturate(profile, mask, 2.).tolist(),
                         [[1., 0., 0., 0., 0.]])

    def test_batches_mark_saturated_rings_as_not_measured(self) -> None:
        from CATCH.lib.instruments import REFERENCE
        from CATCH.lib.lorenzmie import RadialModel
        from CATCH.lib.networks import RADII
        library = dict(res=np.zeros((4, NRADII), np.float32),
                       m=np.ones((4, NRADII), np.float32))
        model = RadialModel(RADII, REFERENCE['wavelength'],
                            REFERENCE['magnification'], REFERENCE['n_m'],
                            'cpu', compile=False)
        plain, _ = TE.Batches(library, model, 3, saturated=0.)(32)
        clipped, _ = TE.Batches(library, model, 3, saturated=1.,
                                level=1.1)(32)
        self.assertEqual(plain.shape, clipped.shape)
        # the same particles: the clipped examples have fewer measured rings
        self.assertTrue((clipped[:, 3] <= plain[:, 3]).all())
        self.assertLess(float(clipped[:, 3].sum()), float(plain[:, 3].sum()))
        # and the rings that are not measured carry nothing
        off = clipped[:, 3] == 0
        self.assertTrue((clipped[:, :3][off[:, None, :].expand(-1, 3, -1)]
                         == 0).all())

    def test_each_example_has_its_own_camera_limit(self) -> None:
        from CATCH.lib.instruments import REFERENCE
        from CATCH.lib.lorenzmie import RadialModel
        from CATCH.lib.networks import RADII
        library = dict(res=np.zeros((4, NRADII), np.float32),
                       m=np.ones((4, NRADII), np.float32))
        model = RadialModel(RADII, REFERENCE['wavelength'],
                            REFERENCE['magnification'], REFERENCE['n_m'],
                            'cpu', compile=False)
        fixed, _ = TE.Batches(library, model, 5, saturated=1.,
                              level=(1.3, 1.3))(64)
        same, _ = TE.Batches(library, model, 5, saturated=1.,
                             level=1.3)(64)
        varied, _ = TE.Batches(library, model, 5, saturated=1.,
                               level=(1.1, 4.))(64)
        # two equal limits are one limit
        self.assertTrue(torch.equal(fixed, same))
        # a range gives examples with more and with fewer rings measured
        rings = varied[:, 3].sum(1)
        self.assertGreater(float(rings.max() - rings.min()), 5.)
        self.assertLess(float(varied[:, 3].sum()), float(
            TE.Batches(library, model, 5, saturated=0.)(64)[0][:, 3].sum()))
        # the per-profile limit of saturate
        profile = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        mask = torch.ones(2, 3)
        out = TE.saturate(profile, mask, torch.tensor([[1.5], [2.5]]))
        self.assertEqual(out.tolist(), [[1., 0., 0.], [1., 1., 0.]])

    def test_the_validation_reports_the_saturated_profiles(self) -> None:
        from CATCH.lib.networks import ProfileNet
        b = np.ones((2, NRADII), np.float32)
        b[:, :20] = 3.
        val = dict(b=b, m=np.ones((2, NRADII), np.float32),
                   z=np.array([500., 3000.]), a=np.ones(2),
                   n=np.full(2, 1.5), scale=np.ones(2),
                   edge=np.full(2, 200.), neighbor=np.full(2, 300.))
        out = TE.evaluate(ProfileNet(), val, torch.device('cpu'))
        self.assertEqual(out['saturated']['n'], 2)
        self.assertIn('within3_unmasked', out['saturated'])

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
        from CATCH.lib.networks import ProfileNet
        out = TE.evaluate(ProfileNet(), val, torch.device('cpu'))
        self.assertIn('scale 1, 10-31 um', out)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
