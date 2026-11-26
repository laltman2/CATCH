'''The measurement of the centers that a localizer finds'''
import unittest
from importlib.util import find_spec

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np
import pandas as pd

from CATCH.customization.evaluation import centers as measure
from CATCH.customization.evaluation.pipelines import extent
from CATCH.customization.evaluation.scenes import Scene, describe

SHAPE = (1024, 1280)


def scene_of(positions, index=0) -> Scene:
    particles = pd.DataFrame(
        [dict(x_p=x, y_p=y, z_p=300., a_p=1., n_p=1.59)
         for x, y in positions])
    return Scene(np.ones(SHAPE), describe(particles, SHAPE), 0.05, index)


def localizer_of(scene: Scene, clip: bool = True, shift: float = 0.):
    '''A localizer that knows the truth: square boxes, cut by the border'''
    def localize(image: np.ndarray) -> pd.DataFrame:
        rows = []
        for _, p in scene.truth.iterrows():
            half = extent(p.z_p, p.a_p, p.n_p)
            x0, y0, x1, y1 = p.x_p - half, p.y_p - half, \
                p.x_p + half, p.y_p + half
            if clip:
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, SHAPE[1]), min(y1, SHAPE[0])
            rows.append({'x_p': p.x_p + shift, 'y_p': p.y_p,
                         'bbox': ((int(x0), int(y0)), int(x1 - x0),
                                  int(y1 - y0)),
                         'edge': bool(clip and (x0 == 0 or y0 == 0
                                                or x1 == SHAPE[1]
                                                or y1 == SHAPE[0])),
                         'confidence': 0.9})
        return pd.DataFrame(rows)
    return localize


class TestCenters(unittest.TestCase):

    def test_a_localizer_that_knows_the_truth(self) -> None:
        scene = scene_of([(640, 512), (300, 600)])
        table = measure.centers(localizer_of(scene, clip=False), [scene])
        self.assertEqual(len(table), 2)
        self.assertTrue(table.found.all())
        self.assertLess(table.err.max(), 1e-9)
        np.testing.assert_allclose(table.ratio, 1., atol=0.02)
        self.assertFalse(table.clipped.any())

    def test_a_box_cut_by_the_border(self) -> None:
        scene = scene_of([(10, 512), (10, 10), (640, 512)])
        table = measure.centers(localizer_of(scene), [scene])
        self.assertEqual(list(table.clipped), [True, True, False])
        self.assertEqual(list(table.corner), [False, True, False])
        # the returned center is right, the middle of the box is not
        self.assertLess(table.err.max(), 1e-9)
        self.assertGreater(table.raw_err.iloc[0], 10.)
        self.assertLess(table.raw_err.iloc[2], 1e-9)
        # a cut box is smaller than the feature
        self.assertLess(table.ratio.iloc[1], table.ratio.iloc[2])

    def test_a_center_that_is_far_off_is_counted_with_its_error(self) -> None:
        scene = scene_of([(640, 512)])
        table = measure.centers(localizer_of(scene, clip=False, shift=40.),
                                [scene])
        self.assertTrue(table.found.iloc[0])
        self.assertAlmostEqual(table.err.iloc[0], 40.)
        self.assertEqual(measure.recall(table, 15.).iloc[0], 0.)
        self.assertEqual(measure.recall(table, 100.).iloc[0], 1.)

    def test_nothing_found(self) -> None:
        scene = scene_of([(640, 512)])
        for empty in (None, pd.DataFrame()):
            def nothing(image: np.ndarray,
                        found=empty) -> pd.DataFrame | None:
                return found

            table = measure.centers(nothing, [scene])
            self.assertFalse(table.found.iloc[0])
            self.assertFalse(table.raw_found.iloc[0])
            self.assertEqual(measure.recall(table).iloc[0], 0.)

    def test_recall_by_distance_to_the_edge(self) -> None:
        scene = scene_of([(10, 512), (640, 512)])
        table = measure.centers(localizer_of(scene), [scene])
        r = measure.recall(table)
        self.assertEqual(r.iloc[0], 1.)
        self.assertEqual(measure.recall(table, raw=True).iloc[0], 0.)

    def test_particles_outside_the_image_are_left_out(self) -> None:
        scene = scene_of([(640, 512), (-30, 50)])
        table = measure.centers(localizer_of(scene, clip=False), [scene])
        self.assertEqual(len(table), 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
