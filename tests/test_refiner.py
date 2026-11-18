'''The Refiner: network, crops, renderer, examples and training'''
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from CATCH import Refiner as R  # noqa: E402
from CATCH.training import refiner as T  # noqa: E402
from CATCH.training import synth  # noqa: E402

SMALL = synth.Distribution(shape=(400, 400), nparticles=(2, 3),
                           a_p=(2., 3.), n_p=(1.5, 1.65), z_p=(150., 300.))
TINY = T.Settings(examples=32, batch_size=16, guesses=2, workers=0,
                  device='cpu', first=16, per_decade=2, nvalidation=8,
                  distribution=SMALL)


def frame(seed: int = 3) -> synth.Frame:
    return synth.make_frame(np.random.default_rng(seed), SMALL,
                            synth.Renderer(SMALL.shape), synth.Extent())


class TestCrop(unittest.TestCase):

    def test_pad_keeps_center_and_marks_outside(self) -> None:
        image = np.full((100, 100), 2.)
        data, mask, ox, oy = R.crop(image, 10., 50., 41, pad=True)
        self.assertEqual(data.shape, (41, 41))
        self.assertAlmostEqual(ox, 0.5 * 0 + (10 - round(10 - 20.5))
                               - 20.)
        self.assertEqual(mask[:, :10].sum(), 0)       # left of the image
        self.assertEqual(mask[:, 10:].min(), 1)
        self.assertTrue((data[mask == 1] == 2).all())
        self.assertTrue((data[mask == 0] == 1).all())

    def test_shift_stays_inside(self) -> None:
        image = np.arange(100 * 100, dtype=float).reshape(100, 100)
        data, mask, ox, oy = R.crop(image, 5., 50., 41, pad=False)
        self.assertEqual(mask.min(), 1)
        self.assertEqual(data[0, 0], image[30, 0])
        self.assertAlmostEqual(ox, 5 - 20.)           # particle off center
        self.assertAlmostEqual(oy, 50 - 30 - 20.)

    def test_shift_larger_than_image(self) -> None:
        data, mask, _, _ = R.crop(np.ones((30, 40)), 15, 15, 100, pad=False)
        self.assertEqual(data.shape, (30, 30))


class TestNetwork(unittest.TestCase):

    def test_shapes(self) -> None:
        net = R.RefinerNet()
        out = net(torch.ones(2, 201, 201), torch.ones(2, 201, 201),
                  torch.ones(2, 201, 201), torch.tensor([1., 2.]))
        self.assertEqual(tuple(out.shape), (2, 5))

    def test_apply(self) -> None:
        state = np.array([200., 3., 0.25, 1., -2.])
        out = R.apply(state, np.array([10., -10., 0., 3., 3.]))
        np.testing.assert_allclose(out[0], 200 * np.exp(0.1))
        np.testing.assert_allclose(out[1], 3 * np.exp(-0.1))
        np.testing.assert_allclose(out[3:], [10., 7.])   # 3 * 3 px
        np.testing.assert_allclose(state, [200., 3., 0.25, 1., -2.])


class TestRenderer(unittest.TestCase):

    def test_finite_and_moves_out_of_sphere(self) -> None:
        r = R.Renderer()
        image = r(300, [10., 3., 0.25, 0., 0.])    # z inside the sphere
        self.assertTrue(np.isfinite(image).all())
        self.assertEqual(image.shape, (201, 201))


class TestExamples(unittest.TestCase):

    def test_target_takes_estimate_to_truth(self) -> None:
        rng = np.random.default_rng(1)
        e = T.make_examples(frame(), rng, R.Renderer(),
                            synth.CropPolicy(mode='mix', repeats=2),
                            guesses=3)
        assert e is not None
        n = len(e.scale)
        self.assertEqual(tuple(e.data.shape), (n, 201, 201))
        self.assertEqual(tuple(e.target.shape), (n, 5))
        for i in range(n):
            again = R.apply(e.states[i], e.target[i].numpy())
            np.testing.assert_allclose(again, e.truth[i], rtol=1e-4,
                                       atol=1e-3)
        self.assertTrue((e.mask >= 0).all() and (e.mask <= 1).all())

    def test_model_at_truth_matches_data(self) -> None:
        # one particle, so that the difference is the noise alone (less
        # after the resize); at an offset of 100 pixels or more as well
        one = synth.Distribution(shape=(400, 400), nparticles=(1, 1),
                                 a_p=(2., 3.), n_p=(1.5, 1.65),
                                 z_p=(150., 300.))
        f = synth.make_frame(np.random.default_rng(0), one,
                             synth.Renderer(one.shape), synth.Extent())
        for mode in ('pad', 'shift'):
            e = T.make_examples(
                f, np.random.default_rng(2), R.Renderer(),
                synth.CropPolicy(mode=mode, jitter=0., tail=0.),
                guesses=1, fixed=1e-9)
            assert e is not None
            diff = (e.data[0] - e.guess[0])[e.mask[0] > 0.99]
            self.assertLess(float(diff.std()), 0.05, mode)
            self.assertGreater(float((e.data[0] - 1).std()), 0.1)

    def test_no_particle_inside(self) -> None:
        f = frame()
        empty = synth.Frame(f.image, f.particles.iloc[:0], f.noise)
        self.assertIsNone(T.make_examples(empty, np.random.default_rng(0),
                                          R.Renderer()))

    def test_join_take(self) -> None:
        rng = np.random.default_rng(1)
        e = T.make_examples(frame(), rng, R.Renderer(), guesses=2)
        assert e is not None
        both = T.join([e, e])
        self.assertEqual(len(both.scale), 2 * len(e.scale))
        part = T.take(both, np.array([0, 1]))
        self.assertEqual(len(part.scale), 2)
        self.assertEqual(len(part.meta), 2)

    def test_perturbation(self) -> None:
        d = np.array([T.perturbation(np.random.default_rng(i), 0.05)
                      for i in range(400)])
        self.assertEqual(d.shape, (400, 3))
        self.assertGreater(d[:, 2].std(), d[:, 0].std())


class TestScore(unittest.TestCase):

    def test_metrics(self) -> None:
        e = T.validation_set(TINY)
        m = T.score(R.RefinerNet(), e, R.Renderer(), torch.device('cpu'))
        for key in ('z0_med', 'z_med', 'a_med', 'c_med', 'good', 'good1'):
            self.assertTrue(np.isfinite(m[key]), key)
        # a network that predicts nothing useful leaves the errors near 4%
        self.assertGreater(m['z0_med'], 0.005)

    def test_state_errors(self) -> None:
        a = np.array([[110., 3., 0.2, 3., 4.]])
        b = np.array([[100., 3., 0.2, 0., 0.]])
        np.testing.assert_allclose(T.state_errors(a, b)[0],
                                   [0.1, 0., 0., 5.])


class TestStream(unittest.TestCase):

    def test_batches_and_reproducible(self) -> None:
        a = T.ExampleStream(7, 8, SMALL, guesses=2)
        b = T.ExampleStream(7, 8, SMALL, guesses=2)
        x, y = next(a.batches()), next(b.batches())
        self.assertEqual(len(x.scale), 8)
        self.assertTrue(torch.equal(x.target, y.target))


class TestTrain(unittest.TestCase):

    def test_run_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = T.train(TINY, directory)
            frame_ = log.frame()
            self.assertEqual(list(frame_.examples), [16, 32])
            self.assertTrue(np.isfinite(frame_.train_loss).all())
            out = Path(directory)
            config = json.loads((out / 'configuration.json').read_text())
            self.assertEqual(config['settings']['guesses'], 2)
            weights = out / 'examples32.pt'
            refiner = R.Refiner(weights, steps=1)
            data = np.ones((120, 120))
            states = refiner([(data, np.ones_like(data))],
                             np.array([[200., 3., 0.25, 0., 0.]]))
            self.assertEqual(states.shape, (1, 5))
            self.assertTrue(np.isfinite(states).all())
            self.assertEqual(len(refiner([], np.zeros((0, 5)))), 0)

    def test_init(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            T.train(TINY, Path(directory) / 'a')
            settings = T.Settings(**{**TINY.__dict__, 'lr': 0.,
                                     'init': str(Path(directory) / 'a'
                                                 / 'examples32.pt')})
            T.train(settings, Path(directory) / 'b')
            before = torch.load(Path(directory) / 'a' / 'examples32.pt')
            after = torch.load(Path(directory) / 'b' / 'examples32.pt')
            for key in before['state_dict']:
                self.assertTrue(torch.equal(before['state_dict'][key],
                                            after['raw'][key]), key)


class TestResume(unittest.TestCase):

    def test_continues_on_new_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = T.Settings(**{**TINY.__dict__, 'resume': True})
            T.train(first, directory)
            self.assertTrue((Path(directory) / 'state.pt').exists())
            saved = torch.load(Path(directory) / 'state.pt',
                               weights_only=False)
            self.assertEqual(saved['examples'], 32)
            self.assertEqual(saved['resumes'], 0)
            longer = T.Settings(**{**TINY.__dict__, 'resume': True,
                                   'examples': 64})
            log = T.train(longer, directory)
            rows = log.frame()
            self.assertEqual(rows.examples.iloc[0], 16)   # the first run
            self.assertEqual(rows.examples.iloc[-1], 64)
            self.assertTrue(rows.seconds.is_monotonic_increasing)
            again = torch.load(Path(directory) / 'state.pt',
                               weights_only=False)
            self.assertEqual(again['resumes'], 1)


class TestPipeline(unittest.TestCase):

    def test_refined_pipeline(self) -> None:
        from CATCH.evaluation.pipelines import truth_pipeline
        from CATCH.evaluation.scenes import make_scenes
        scene = make_scenes(1, (1, 2), (200, 200), 0.05, 5, margin=0.)[0]
        with tempfile.TemporaryDirectory() as directory:
            T.train(TINY, directory)
            refiner = R.Refiner(Path(directory) / 'examples32.pt', steps=1)
            run = R.refined_pipeline(truth_pipeline(), refiner,
                                     minimum_radius=0.)
            out = run(scene)
        self.assertIn('z_p_0', out.columns)
        self.assertEqual(len(out), int(scene.truth.inside.sum()))
        self.assertTrue(np.isfinite(out[['z_p', 'a_p', 'n_p']]
                                    .to_numpy()).all())
        pd.testing.assert_series_equal(out.a_p_0, scene.truth[
            scene.truth.inside].a_p.reset_index(drop=True),
            check_names=False, check_dtype=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
