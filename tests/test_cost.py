'''Training cost: the log, the cost to reach a target, the checkpoints'''
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from CATCH.evaluation import cost


class TestHardware(unittest.TestCase):

    def test_describe_hardware(self) -> None:
        info = cost.describe_hardware()
        for key in ('platform', 'machine', 'cpus', 'python', 'torch',
                    'cuda', 'devices'):
            self.assertIn(key, info)
        self.assertGreaterEqual(info['cpus'], 1)


class TestTrainingLog(unittest.TestCase):

    def test_log_rows(self) -> None:
        log = cost.TrainingLog(name='run', epochs=3)
        for epoch in (1, 2, 3):
            log.log(epoch, epoch * 100, test_loss=1. / epoch)
        frame = log.frame()
        self.assertEqual(list(frame.epoch), [1, 2, 3])
        self.assertEqual(list(frame.examples), [100, 200, 300])
        self.assertTrue(frame.seconds.is_monotonic_increasing)
        self.assertEqual(list(frame.columns),
                         ['seconds', 'epoch', 'examples', 'test_loss'])

    def test_a_pause_is_not_counted(self) -> None:
        log = cost.TrainingLog()
        with log.pause():
            time.sleep(0.2)
        self.assertLess(log.seconds, 0.1)

    def test_a_pause_that_fails_is_still_not_counted(self) -> None:
        log = cost.TrainingLog()
        with self.assertRaises(RuntimeError), log.pause():
            time.sleep(0.2)
            raise RuntimeError
        self.assertLess(log.seconds, 0.1)

    def test_save_and_load(self) -> None:
        log = cost.TrainingLog(name='run', config={'batch': 64})
        log.log(1, 100, lock_on=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cost.csv'
            log.save(path)
            frame, info = cost.TrainingLog.load(path)
        self.assertEqual(list(frame.examples), [100])
        self.assertEqual(info['settings']['config'], {'batch': 64})
        self.assertIn('torch', info['hardware'])


class TestCostToReach(unittest.TestCase):

    def setUp(self) -> None:
        self.record = pd.DataFrame(dict(
            seconds=[10., 20., 30., 40.], epoch=[1, 2, 3, 4],
            examples=[100, 200, 300, 400],
            lock_on=[float('nan'), 0.2, 0.5, 0.6],
            loss=[1., 0.5, 0.2, 0.1]))

    def test_first_time_that_the_target_is_reached(self) -> None:
        self.assertEqual(cost.cost_to_reach(self.record, 'lock_on', 0.5),
                         dict(seconds=30., epoch=3, examples=300))

    def test_a_loss_is_to_go_below(self) -> None:
        self.assertEqual(cost.cost_to_reach(self.record, 'loss', 0.3,
                                            lower_is_better=True),
                         dict(seconds=30., epoch=3, examples=300))

    def test_a_target_that_is_never_reached(self) -> None:
        self.assertIsNone(cost.cost_to_reach(self.record, 'lock_on', 0.9))

    def test_throughput(self) -> None:
        self.assertEqual(cost.throughput(self.record), 10.)


class TestCheckpoints(unittest.TestCase):

    def test_epoch_of(self) -> None:
        self.assertEqual(cost.epoch_of('a/b/epoch300.pt'), 300)
        with self.assertRaises(ValueError):
            cost.epoch_of('best.pt')

    def test_validate(self) -> None:
        results = pd.DataFrame(dict(ok=[True, False, False, False],
                                    all5=[True, True, False, False],
                                    z_rel=[.1, .2, .3, .4],
                                    a_rel=[.01] * 4, n_rel=[.02] * 4))
        with patch('CATCH.evaluation.run.evaluate', return_value=results):
            metrics = cost.validate(lambda crops: pd.DataFrame(), ncases=4)
        self.assertEqual(metrics['lock_on'], 0.25)
        self.assertEqual(metrics['within5'], 0.5)
        self.assertAlmostEqual(metrics['z_med'], 0.25)

    def test_evaluate_checkpoints_in_order_of_the_epoch(self) -> None:
        seen = []

        def make(path):
            seen.append(path.name)
            return lambda crops: pd.DataFrame()

        def fake_validate(estimate, ncases, nprocs):
            return dict(lock_on=len(seen) / 10.)

        paths = ['x/epoch20.pt', 'x/epoch3.pt', 'x/epoch100.pt']
        with patch.object(cost, 'validate', fake_validate):
            record = cost.evaluate_checkpoints(paths, make, 1000,
                                               extra=dict(run='a'))
        self.assertEqual(seen, ['epoch3.pt', 'epoch20.pt', 'epoch100.pt'])
        self.assertEqual(list(record.epoch), [3, 20, 100])
        self.assertEqual(list(record.examples), [3000, 20000, 100000])
        self.assertEqual(list(record.run), ['a'] * 3)
        self.assertEqual(list(record.lock_on), [0.1, 0.2, 0.3])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
