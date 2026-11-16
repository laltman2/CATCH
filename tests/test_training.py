'''The training of the Estimator: a small run on the CPU, with its cost'''
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from CATCH.evaluation.cost import TrainingLog  # noqa: E402
from CATCH.training.EstimatorTraining import EstimatorTraining  # noqa: E402

PARTICLE = {'a_p': [0.5, 1.0], 'n_p': [1.4, 1.6], 'k_p': [0, 0],
            'x_p': [90, 110], 'y_p': [90, 110], 'z_p': [100, 300]}


def make_config(directory: Path, **training) -> dict:
    options = dict(epochs=2, batchsize=2, num_workers=0, checkpoint_every=1,
                   savefile=str(directory / 'model'), patience=1)
    options['continue'] = False
    options.update(training)
    return {'instrument': {'wavelength': 0.447, 'magnification': 0.048,
                           'n_m': 1.34},
            'particle': PARTICLE, 'training': options,
            'directory': str(directory / 'data') + '/', 'imgtype': 'png',
            'shape': [201, 201], 'noise': 0.05, 'ext_noise': 0.05,
            'train': {'nframes': 4}, 'test': {'nframes': 2},
            'eval': {'nframes': 2}, 'scale_integer': False,
            'max_overlaps': 1, 'overwrite': True,
            'delete_files_after_training': False}


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)

    def check_outputs(self, savefile: Path) -> pd.DataFrame:
        checkpoints = Path(str(savefile) + '_checkpoints')
        for name in ('best.pt', 'last.pt', 'epoch1.pt'):
            self.assertTrue((checkpoints / name).is_file(), name)
        self.assertTrue(Path(str(savefile) + '_train_info.csv').is_file())
        record, info = TrainingLog.load(Path(str(savefile) + '_cost.csv'))
        self.assertGreater(info['settings']['data_seconds'], 0)
        self.assertIn('hardware', info)
        return record

    def test_manual_training(self) -> None:
        trainer = EstimatorTraining(make_config(self.directory, epochs=2))
        trainer.train()
        record = self.check_outputs(self.directory / 'model')
        self.assertEqual(list(record.epoch), [1, 2])
        self.assertEqual(list(record.examples), [4, 8])
        self.assertTrue(record.seconds.is_monotonic_increasing)
        self.assertTrue((record.seconds > 0).all())
        for column in ('train_loss', 'test_loss', 'z_loss', 'n_testloss'):
            self.assertIn(column, record.columns)
        # the test loss is computed on the CPU too (it was not once)
        self.assertTrue((record.test_loss > 0).all())

    def test_automatic_training_stops_at_the_budget(self) -> None:
        # a patience that is never lost, as it takes hundreds of epochs
        # for the three outputs to stop improving
        config = make_config(self.directory, epochs=0, patience=10**6,
                             max_epochs=3)
        trainer = EstimatorTraining(config)
        trainer.train()
        record = self.check_outputs(self.directory / 'model')
        self.assertEqual(list(record.epoch), [1, 2, 3])
        assert trainer.stoppers is not None
        self.assertFalse(any(s.early_stop
                             for s in trainer.stoppers.values()))

    def test_automatic_training_stops_when_the_outputs_stop(self) -> None:
        config = make_config(self.directory, epochs=0, patience=1)
        trainer = EstimatorTraining(config)
        # the outputs stop improving at once: the loss is the same every
        # time, and a tie is not an improvement
        with patch.object(trainer, 'loss_fn',
                          lambda out, labels: tuple(
                              torch.tensor(1., requires_grad=True) * 0 + 1
                              for _ in range(3))):
            trainer.train()
        assert trainer.stoppers is not None
        self.assertTrue(all(s.early_stop
                            for s in trainer.stoppers.values()))
        self.assertLessEqual(trainer.epoch, 4)

    def test_training_changes_the_weights(self) -> None:
        trainer = EstimatorTraining(make_config(self.directory, epochs=1))
        assert trainer.model is not None
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.train()
        after = list(trainer.model.parameters())
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(before, after, strict=True)))

    def test_a_checkpoint_can_be_evaluated(self) -> None:
        from CATCH.Estimator import Estimator
        config = make_config(self.directory, epochs=1)
        EstimatorTraining(config).train()
        weights = Path(config['training']['savefile'] + '_checkpoints') \
            / 'epoch1.pt'
        estimator = Estimator(weights=weights, config=config)
        crop = np.full((100, 100), 1.)
        self.assertEqual(list(estimator([crop]).columns),
                         ['z_p', 'a_p', 'n_p'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
