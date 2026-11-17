'''The training loop of the Estimator'''
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
from CATCH.training import loop, synth  # noqa: E402

SMALL = synth.Distribution(shape=(320, 320), nparticles=(2, 4),
                           a_p=(0.5, 2.), z_p=(150., 400.))
TINY = loop.Settings(examples=96, batch_size=16, workers=0, device='cpu',
                     first=32, per_decade=2, nvalidation=24)


class TestSchedule(unittest.TestCase):

    def test_spacing(self) -> None:
        marks = loop.schedule(1_000_000, 1000, 4)
        self.assertEqual(marks[0], 1000)
        self.assertEqual(marks[-1], 1_000_000)
        self.assertEqual(len(marks), 13)
        ratios = np.diff(np.log10(marks))
        np.testing.assert_allclose(ratios, 0.25, atol=1e-3)

    def test_short_run(self) -> None:
        self.assertEqual(loop.schedule(500, 1000), [500])


class TestRate(unittest.TestCase):

    def test_warmup_and_cosine(self) -> None:
        rate = loop.learning_rate_factor
        self.assertAlmostEqual(rate(0, 1000, 10), 0.1)
        self.assertAlmostEqual(rate(9, 1000, 10), 1.)
        self.assertAlmostEqual(rate(10, 1000, 10), 1.)
        self.assertAlmostEqual(rate(505, 1000, 10), 0.5, places=2)
        self.assertAlmostEqual(rate(1000, 1000, 10), 0., places=6)
        self.assertEqual(rate(900, 1000, 10, decay=False), 1.)
        self.assertAlmostEqual(rate(5, 1000, 0), 1., places=3)


class TestScaler(unittest.TestCase):

    def test_round_trip(self) -> None:
        config = loop.network_config(synth.Distribution())
        scaler = loop.Scaler(config)
        values = torch.tensor([[104., 0.2, 1.38], [600., 5., 2.5]])
        scaled = scaler.normalize(values)
        np.testing.assert_allclose(scaled.numpy(), [[0, 0, 0], [1, 1, 1]],
                                   atol=1e-6)
        np.testing.assert_allclose(scaler.unnormalize(scaled).numpy(),
                                   values.numpy(), rtol=1e-6)


    def test_log_scaling(self) -> None:
        config = loop.network_config(synth.Distribution(), 'log')
        scaler = loop.Scaler(config)
        values = torch.tensor([[104., 0.2, 1.38], [600., 5., 2.5],
                               [300., 1., 1.5]])
        scaled = scaler.normalize(values)
        np.testing.assert_allclose(scaled[:2].numpy(),
                                   [[0, 0, 0], [1, 1, 1]], atol=1e-5)
        np.testing.assert_allclose(scaler.unnormalize(scaled).numpy(),
                                   values.numpy(), rtol=1e-4)
        self.assertAlmostEqual(scaler.n_m, 1.34)

    def test_bad_scaling(self) -> None:
        with self.assertRaises(ValueError):
            loop.network_config(synth.Distribution(), 'cubic')


def weight(net: torch.nn.Module) -> torch.Tensor:
    return net.state_dict()['weight'].clone()


class TestAverage(unittest.TestCase):

    def test_follows_and_lags(self) -> None:
        net = torch.nn.Linear(2, 1)
        average = loop.Average(net, 0.9)
        start = weight(average.module)
        with torch.no_grad():
            net.weight.add_(1.)
        for _ in range(50):
            average.update(net)
        gap = (weight(net) - weight(average.module)).abs().max().item()
        self.assertLess(gap, 0.05)                  # it has caught up
        self.assertFalse(torch.equal(start, weight(average.module)))
        self.assertIsNot(average.module, net)

    def test_early_steps_follow_closely(self) -> None:
        net = torch.nn.Linear(2, 1)
        average = loop.Average(net, 0.999)
        with torch.no_grad():
            net.weight.add_(1.)
        average.update(net)
        gap = (weight(net) - weight(average.module)).abs().max().item()
        self.assertLess(gap, 0.85)                  # not at 0.999

    def test_off(self) -> None:
        net = torch.nn.Linear(2, 1)
        average = loop.Average(net, 0.)
        self.assertIs(average.module, net)


class TestScore(unittest.TestCase):

    def test_metrics(self) -> None:
        samples = loop.validation_set(SMALL, synth.CropPolicy(), 24)
        config = loop.network_config(SMALL)
        metrics = loop.score(loop.TorchEstimator(), samples,
                             loop.Scaler(config), torch.device('cpu'),
                             batch_size=10)
        for key in ('val_loss', 'z_med', 'a_med', 'n_med', 'c_med',
                    'good'):
            self.assertIn(key, metrics)
        self.assertIn('good_r1', metrics)
        for key in ('val_loss', 'z_med', 'a_med', 'n_med', 'c_med',
                    'good'):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertTrue(0 <= metrics['good'] <= 1)
        near = (samples.meta['edge'] < loop.EDGE).any()
        self.assertEqual(np.isfinite(metrics['good_edge']), near)


class TestTrain(unittest.TestCase):

    def test_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = loop.train(TINY, directory, SMALL)
            frame = log.frame()
            self.assertEqual(list(frame.examples), [32, 96])
            self.assertTrue(frame.seconds.is_monotonic_increasing)
            self.assertTrue(np.isfinite(frame.train_loss).all())
            out = Path(directory)
            config = json.loads((out / 'configuration.json').read_text())
            self.assertEqual(config['particle']['z_p'], [150., 400.])
            self.assertTrue((out / 'log.csv').exists())
            weights = out / 'examples96.pt'
            self.assertTrue(weights.exists())
            estimate = Estimator(weights=weights, config=config)
            result = estimate([np.ones((60, 60))])
            self.assertEqual(list(result.columns), ['z_p', 'a_p', 'n_p'])

    def test_log_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = loop.Settings(**{**TINY.__dict__, 'scaling': 'log',
                                        'examples': 32})
            loop.train(settings, directory, SMALL)
            config = json.loads(
                (Path(directory) / 'configuration.json').read_text())
            self.assertEqual(config['scaling']['a_p'], 'log')
            result = Estimator(weights=Path(directory) / 'examples32.pt',
                               config=config)([np.ones((60, 60))])
            self.assertTrue(np.isfinite(result.to_numpy()).all())
            self.assertGreater(result.n_p[0], 1.34)

    def test_network_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = loop.Settings(**{
                **TINY.__dict__, 'examples': 32,
                'network': dict(scale_nodes=8, film=True)})
            loop.train(settings, directory, SMALL)
            config = json.loads(
                (Path(directory) / 'configuration.json').read_text())
            self.assertEqual(config['network']['scale_nodes'], 8)
            estimate = Estimator(weights=Path(directory) / 'examples32.pt',
                                 config=config)
            self.assertEqual(estimate.dense1.in_features, 408)
            self.assertTrue(np.isfinite(
                estimate([np.ones((60, 60))]).to_numpy()).all())

    def test_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first'
            loop.train(loop.Settings(**{**TINY.__dict__, 'examples': 32}),
                       first, SMALL)
            weights = first / 'examples32.pt'
            second = Path(directory) / 'second'
            settings = loop.Settings(**{**TINY.__dict__, 'examples': 32,
                                        'init': str(weights), 'lr': 0.})
            loop.train(settings, second, SMALL)
            # with no learning the weights are those that it started from
            before = torch.load(weights)['state_dict']
            after = torch.load(second / 'examples32.pt')['state_dict']
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_resume_other_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first'
            loop.train(loop.Settings(**{**TINY.__dict__, 'examples': 32}),
                       first, SMALL)
            settings = loop.Settings(**{
                **TINY.__dict__, 'examples': 32, 'scaling': 'log',
                'init': str(first / 'examples32.pt')})
            with self.assertRaises(ValueError):
                loop.train(settings, Path(directory) / 'second', SMALL)

    def test_losses(self) -> None:
        for name in loop.LOSSES:
            with tempfile.TemporaryDirectory() as directory:
                settings = loop.Settings(**{**TINY.__dict__, 'loss': name,
                                            'examples': 32})
                log = loop.train(settings, directory, SMALL)
                self.assertTrue(np.isfinite(log.frame().train_loss).all())
        with self.assertRaises(ValueError), \
                tempfile.TemporaryDirectory() as directory:
            loop.train(loop.Settings(**{**TINY.__dict__, 'loss': 'huber'}),
                       directory, SMALL)

    def test_checkpoint_has_both(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop.train(loop.Settings(**{**TINY.__dict__, 'examples': 32}),
                       directory, SMALL)
            saved = torch.load(Path(directory) / 'examples32.pt')
            self.assertEqual(set(saved), {'state_dict', 'raw'})

    def test_no_decay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = loop.Settings(**{**TINY.__dict__, 'decay': False,
                                        'examples': 32})
            log = loop.train(settings, directory, SMALL)
            self.assertEqual(len(log.frame()), 1)

    def test_workers_differ(self) -> None:
        a = loop.CropStream(1, 8, SMALL, synth.CropPolicy())
        first = next(iter(a))
        again = next(iter(a))
        self.assertTrue(torch.equal(first[2], again[2]))   # reproducible

    def test_lock_on_option(self) -> None:
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            settings = loop.Settings(**{**TINY.__dict__, 'lock_cases': 3,
                                        'examples': 32})
            with patch.object(loop, 'validate',
                              return_value={'lock_on': 0.5}) as called:
                log = loop.train(settings, directory, SMALL)
            called.assert_called_once()
            self.assertEqual(log.frame().lock_on.iloc[0], 0.5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
