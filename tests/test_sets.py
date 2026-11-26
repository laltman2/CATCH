'''The development and test sets of scenes'''
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import pandas as pd

from CATCH.customization.evaluation import sets

TINY = sets.SceneSet('tiny', nscenes=4, seed=7, nparticles=(1, 3),
                     shape=(96, 128))


class TestDefinitions(unittest.TestCase):

    def test_the_sets(self) -> None:
        self.assertEqual(list(sets.SETS), ['dev', 'test'])
        dev, test = sets.SETS['dev'], sets.SETS['test']
        self.assertNotEqual(dev.seed, test.seed)
        self.assertGreater(test.nscenes, dev.nscenes)

    def test_dev_is_the_set_of_the_baseline(self) -> None:
        '''The baseline of the original CATCH: seed 2026, 200 scenes of 1
        to 6 particles in 1024 x 1280 pixels with noise of 0.05'''
        dev = sets.SETS['dev']
        self.assertEqual((dev.seed, dev.nscenes, dev.nparticles, dev.shape,
                          dev.noise, dev.margin, dev.min_separation),
                         (2026, 200, (1, 6), (1024, 1280), 0.05, 0., 20.))

    def test_get(self) -> None:
        self.assertEqual(sets.get('test'), sets.SETS['test'])
        with self.assertRaises(ValueError):
            sets.get('other')


class TestMaking(unittest.TestCase):

    def test_the_first_scenes_do_not_depend_on_how_many(self) -> None:
        few = TINY.make(2)
        many = TINY.make()
        self.assertEqual(len(few), 2)
        self.assertEqual(len(many), 4)
        for a, b in zip(few, many, strict=False):
            self.assertTrue(a.truth.equals(b.truth))

    def test_too_many_or_none(self) -> None:
        for count in (0, 5, -1):
            with self.assertRaises(ValueError):
                TINY.make(count)

    def test_truth_table(self) -> None:
        table = sets.truth_table(TINY.make(3))
        self.assertEqual(sorted(table.scene.unique()), [0, 1, 2])
        self.assertIn('x_p', table.columns)


class TestStoredTruth(unittest.TestCase):

    def test_stored_sizes(self) -> None:
        for name, (particles, scenes) in dict(dev=(717, 200),
                                              test=(1381, 400)).items():
            table = pd.read_csv(sets.SETS[name].path())
            self.assertEqual((len(table), table.scene.nunique()),
                             (particles, scenes), name)
            self.assertEqual(table.scene.max(), scenes - 1)

    def test_the_sets_have_no_particle_in_common(self) -> None:
        positions = [set(map(tuple, pd.read_csv(sets.SETS[name].path())[
            ['x_p', 'y_p']].round(3).to_numpy())) for name in sets.SETS]
        self.assertEqual(positions[0] & positions[1], set())

    def test_the_code_still_makes_the_stored_scenes(self) -> None:
        for name in sets.SETS:
            sets.verify(name, nscenes=2)

    def test_write_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sets.SETS, {'tiny': TINY}), \
             patch.object(sets, 'DIRECTORY', Path(tmp) / 'sets'):
            path = sets.write_truth('tiny')
            self.assertEqual(path.name, 'tiny_truth.csv')
            sets.verify('tiny', nscenes=4)

            stored = pd.read_csv(path)
            stored.loc[0, 'x_p'] += 1.
            stored.to_csv(path, index=False)
            with self.assertRaises(ValueError) as context:
                sets.verify('tiny', nscenes=2)
            self.assertIn('changed', str(context.exception))

            stored.iloc[1:].to_csv(path, index=False)
            with self.assertRaises(ValueError) as context:
                sets.verify('tiny', nscenes=2)
            self.assertIn('particles', str(context.exception))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
