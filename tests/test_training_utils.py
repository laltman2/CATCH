import unittest
from types import SimpleNamespace

from CATCH.training.Classify import classify
from CATCH.training.EarlyStopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):

    def test_improvement_resets_counter(self) -> None:
        stop = EarlyStopping(patience=3)
        stop(1.0)
        stop(1.1)
        stop(1.2)
        self.assertEqual(stop.counter, 2)
        stop(0.5)
        self.assertEqual(stop.counter, 0)
        self.assertEqual(stop.best_loss, 0.5)
        self.assertFalse(stop.early_stop)

    def test_stops_after_patience(self) -> None:
        stop = EarlyStopping(patience=2)
        for loss in (1.0, 1.1, 1.2):
            stop(loss)
        self.assertTrue(stop.early_stop)

    def test_delta_requires_a_real_improvement(self) -> None:
        stop = EarlyStopping(patience=1, delta=0.1)
        stop(1.0)
        stop(0.95)
        self.assertTrue(stop.early_stop)


class TestClassify(unittest.TestCase):

    def test_single_class(self) -> None:
        config = {'particle': {'names': ['hologram']}}
        self.assertEqual(classify(None, config), 0)

    def test_sign_of_index_contrast(self) -> None:
        config = {'particle': {'names': ['-n_p', '+n_p']},
                  'instrument': {'n_m': 1.34}}
        low = SimpleNamespace(n_p=1.2)
        high = SimpleNamespace(n_p=1.5)
        self.assertEqual(classify(low, config), 0)
        self.assertEqual(classify(high, config), 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
