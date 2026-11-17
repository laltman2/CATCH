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


class TestScore(unittest.TestCase):

    def test_metrics(self) -> None:
        samples = loop.validation_set(SMALL, synth.CropPolicy(), 24)
        config = loop.network_config(SMALL)
        metrics = loop.score(loop.TorchEstimator(), samples,
                             loop.Scaler(config), torch.device('cpu'),
                             batch_size=10)
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
