'''The diagnosis of an estimator: neighbors, shifts and the particle itself'''
import unittest
from importlib.util import find_spec
from typing import Any

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from CATCH.evaluation import diagnose, pipelines, scenes  # noqa: E402

SHAPE = (256, 320)


def small_scenes(count=2):
    options: dict[str, Any] = dict(shape=SHAPE, radii=(1.,),
                                   indexes={'ps': 1.59},
                                   z_range=(250., 300.))
    return scenes.make_scenes(count, (1, 2), **options)


class TestPadAround(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.arange(100. * 120).reshape(100, 120)

    def test_inside_the_image_it_is_the_crop(self) -> None:
        crop, x0, y0 = pipelines.crop_around(self.image, 60, 50, 40)
        np.testing.assert_array_equal(
            pipelines.pad_around(self.image, 60, 50, 40), crop)

    def test_at_the_corner_it_is_padded_and_centered(self) -> None:
        out = pipelines.pad_around(self.image, 5, 5, 40, value=-1.)
        self.assertEqual(out.shape, (40, 40))
        # the pixel at the particle is at the center of the crop
        self.assertEqual(out[20, 20], self.image[5, 5])
        self.assertTrue((out[:15] == -1.).all())
        self.assertTrue((out[:, :15] == -1.).all())
        np.testing.assert_array_equal(out[15:, 15:], self.image[:25, :25])

    def test_a_crop_larger_than_the_image(self) -> None:
        out = pipelines.pad_around(self.image, 60, 50, 300)
        self.assertEqual(out.shape, (300, 300))
        self.assertEqual(out[150, 150], self.image[50, 60])
        self.assertEqual(out[0, 0], 1.)

    def test_a_crop_outside_the_image_is_background(self) -> None:
        out = pipelines.pad_around(self.image, -500, -500, 30)
        self.assertTrue((out == 1.).all())

    def test_the_pipeline_pads_on_request(self) -> None:
        made = small_scenes(1)
        seen = {}

        def estimator(crops):
            seen['crops'] = crops
            return pd.DataFrame({'z_p': [300.] * len(crops),
                                 'a_p': [1.] * len(crops),
                                 'n_p': [1.5] * len(crops)})

        for pad in (False, True):
            out = pipelines.estimator_on_truth(estimator, pad=pad)(made[0])
            self.assertEqual(len(out), len(seen['crops']))
            self.assertTrue(all(c.shape[0] == c.shape[1]
                                for c in seen['crops']))
        # a padded crop is as large as the extent says, not capped
        truth = made[0].truth[made[0].truth.inside].iloc[0]
        side = int(2 * pipelines.extent(truth.z_p, truth.a_p, truth.n_p))
        self.assertEqual(seen['crops'][0].shape[0], max(side, 31))


class TestDiagnose(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.made = small_scenes(2)

        def estimate(crops):
            return pd.DataFrame({'z_p': [300.] * len(crops),
                                 'a_p': [1.] * len(crops),
                                 'n_p': [1.59] * len(crops)})

        cls.table = diagnose.errors(estimate, cls.made)

    def one(self, scene, row, x, y, seed):
        particle = pd.DataFrame([row[['x_p', 'y_p', 'z_p', 'a_p', 'n_p']]])
        particle['x_p'], particle['y_p'] = x, y
        return scenes.render(scenes.describe(particle, SHAPE), SHAPE,
                             scene.noise, np.random.default_rng(seed))

    def test_the_particle_alone_has_no_neighbor(self) -> None:
        scene = next(s for s in self.made if len(s.truth) > 1)
        row = scene.truth.iloc[0]
        image = diagnose.alone(scene, row, centered=False, seed=3)
        self.assertEqual(image.shape, scene.image.shape)
        np.testing.assert_allclose(
            image, self.one(scene, row, row.x_p, row.y_p, 3))
        self.assertFalse(np.allclose(image, scene.image))

    def test_a_centered_particle_is_at_the_center(self) -> None:
        scene = self.made[0]
        row = scene.truth.iloc[0]
        image = diagnose.alone(scene, row, centered=True, seed=3)
        rows, columns = SHAPE
        np.testing.assert_allclose(
            image, self.one(scene, row, columns / 2., rows / 2., 3))

    def test_the_crops_of_the_conditions(self) -> None:
        for scene in self.made:
            truth, crops = diagnose.crops(scene)
            self.assertEqual(set(crops), set(diagnose.CONDITIONS))
            for condition in diagnose.CONDITIONS:
                self.assertEqual(len(crops[condition]), len(truth))
            self.assertIn('side', truth)

    def test_one_row_for_each_particle_and_condition(self) -> None:
        n = sum(int(s.truth.inside.sum()) for s in self.made)
        self.assertEqual(len(self.table), n * len(diagnose.CONDITIONS))
        for column in ('da', 'dz', 'dn', 'condition', 'side', 'edge',
                       'neighbor'):
            self.assertIn(column, self.table.columns)

    def test_the_errors_are_relative(self) -> None:
        row = self.table.iloc[0]
        self.assertAlmostEqual(row.dz, (300. - row.z_p) / row.z_p)
        self.assertAlmostEqual(row.da, (1. - row.a_p) / row.a_p)

    def test_summary(self) -> None:
        summary = diagnose.summarize(self.table)
        self.assertEqual(list(summary.index), list(diagnose.CONDITIONS))
        self.assertTrue((summary.n == summary.n.iloc[0]).all())
        by = diagnose.summarize(self.table, 'a_p')
        self.assertIn('both', by.columns)
        self.assertEqual(int(by.n.sum()), len(self.table))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
