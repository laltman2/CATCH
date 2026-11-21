'''Scenes with several particles: simulation, matching and evaluation'''
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from typing import Any

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cv2  # noqa: E402

from CATCH.customization.evaluation import (  # noqa: E402
    detect_images, pipelines, scene_run, scenes)
from CATCH.customization.evaluation.cases import RADII  # noqa: E402
from CATCH.customization.evaluation.matching import match  # noqa: E402

SHAPE = (256, 320)


def small_scenes(count=2, nparticles=(1, 1), **kwargs):
    options: dict[str, Any] = dict(shape=SHAPE, radii=(1.,),
                                   indexes={'ps': 1.59},
                                   z_range=(250., 300.))
    options.update(kwargs)
    return scenes.make_scenes(count, nparticles, **options)


class TestDescribe(unittest.TestCase):

    def test_edge_neighbor_and_inside(self) -> None:
        particles = pd.DataFrame(dict(
            x_p=[10., 100., 400.], y_p=[50., 50., 300.], z_p=[200.] * 3,
            a_p=[1.] * 3, n_p=[1.5] * 3))
        out = scenes.describe(particles, (256, 320))
        self.assertEqual(list(out.nparticles), [3, 3, 3])
        # the third is 81 pixels past the right edge and 45 past the bottom
        self.assertEqual(list(out.edge), [10., 50., -81.])
        self.assertEqual(list(out.inside), [True, True, False])
        self.assertEqual(out.neighbor[0], 90.)
        self.assertEqual(out.neighbor[1], 90.)
        self.assertGreater(out.neighbor[2], 300.)

    def test_a_lone_particle_has_no_neighbor(self) -> None:
        particles = pd.DataFrame(dict(x_p=[5.], y_p=[5.], z_p=[200.],
                                      a_p=[1.], n_p=[1.5]))
        self.assertEqual(scenes.describe(particles, SHAPE).neighbor[0],
                         np.inf)


class TestMakeScenes(unittest.TestCase):

    def test_scenes(self) -> None:
        made = scenes.make_scenes(4, (1, 3), shape=SHAPE)
        self.assertEqual(len(made), 4)
        for scene in made:
            self.assertEqual(scene.image.shape, SHAPE)
            # normalized, to within what a large sphere does in a small image
            self.assertAlmostEqual(scene.image.mean(), 1., delta=0.3)
            self.assertTrue(1 <= len(scene.truth) <= 3)
            self.assertEqual(scene.noise, 0.05)
            for column in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'nparticles',
                           'edge', 'neighbor', 'inside'):
                self.assertIn(column, scene.truth.columns)
            self.assertTrue(scene.truth.a_p.isin(RADII).all())
            self.assertTrue(scene.truth.inside.all())

    def test_reproducible(self) -> None:
        a = small_scenes(2)
        b = small_scenes(2)
        self.assertTrue(np.array_equal(a[0].image, b[0].image))
        self.assertTrue(a[1].truth.equals(b[1].truth))

    def test_separation(self) -> None:
        for scene in scenes.make_scenes(6, (4, 4), shape=SHAPE,
                                        min_separation=60.):
            self.assertGreaterEqual(scene.truth.neighbor.min(), 60.)

    def test_margin_puts_centers_outside_the_image(self) -> None:
        made = small_scenes(8, (3, 3), margin=300.)
        self.assertTrue(any(not scene.truth.inside.all() for scene in made))

    def test_crowding_that_cannot_be_satisfied(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            scenes.make_scenes(1, (3, 3), shape=(20, 20),
                               min_separation=100., max_attempts=20)
        self.assertIn('20 attempts', str(context.exception))

    def test_fields_add(self) -> None:
        '''Two particles are not the sum of the two holograms'''
        rng = np.random.default_rng(0)
        one = pd.DataFrame(dict(x_p=[100.], y_p=[100.], z_p=[250.],
                                a_p=[1.], n_p=[1.59]))
        two = pd.concat([one, one.assign(x_p=[180.], z_p=[300.])],
                        ignore_index=True)
        h1 = scenes.render(one, SHAPE, 0., rng)
        h2 = scenes.render(one.assign(x_p=[180.], z_p=[300.]), SHAPE, 0.,
                           rng)
        both = scenes.render(two, SHAPE, 0., rng)
        self.assertFalse(np.allclose(both, h1 + h2 - 1., atol=1e-3))
        self.assertAlmostEqual(both.mean(), 1., delta=0.1)


class TestImages(unittest.TestCase):

    def test_to_uint8(self) -> None:
        image = np.array([[0., 1., 2.54, 3., -1.]])
        out = scenes.to_uint8(image)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(list(out[0]), [0, 100, 254, 255, 0])

    def test_scenes_know_their_number(self) -> None:
        self.assertEqual([s.index for s in small_scenes(3)], [0, 1, 2])

    def test_write_images(self) -> None:
        made = small_scenes(2)
        with tempfile.TemporaryDirectory() as tmp:
            paths = scenes.write_images(made, Path(tmp) / 'new' / 'images')
            self.assertEqual([p.name for p in paths],
                             ['scene_0000.png', 'scene_0001.png'])
            image = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        self.assertTrue(np.array_equal(image, scenes.to_uint8(made[1].image)))

    def test_an_image_that_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / 'file'
            blocked.write_text('')
            with self.assertRaises(OSError):
                scenes.write_images(small_scenes(1), blocked)


class TestRecorded(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'found.csv'
        pd.DataFrame(dict(scene=[0, 0, 2], x_p=[1., 2., 3.], y_p=[4., 5., 6.],
                          z_p=[200.] * 3, a_p=[1.] * 3, n_p=[1.5] * 3,
                          confidence=[.9] * 3)).to_csv(self.path,
                                                       index=False)

    def test_returns_the_rows_of_the_scene(self) -> None:
        pipeline = pipelines.recorded_pipeline(self.path)
        made = small_scenes(3)
        self.assertEqual(list(pipeline(made[0]).x_p), [1., 2.])
        self.assertEqual(len(pipeline(made[1])), 0)
        self.assertEqual(list(pipeline(made[2]).x_p), [3.])
        self.assertEqual(list(pipeline(made[0]).columns),
                         list(pipelines.COLUMNS))

    def test_a_file_that_lacks_columns(self) -> None:
        pd.DataFrame(dict(scene=[0], x_p=[1.])).to_csv(self.path,
                                                       index=False)
        with self.assertRaises(ValueError):
            pipelines.recorded_pipeline(self.path)

    def test_scoring_what_was_recorded(self) -> None:
        made = small_scenes(2)
        record = pd.concat([detections(s).assign(scene=s.index)
                            for s in made])
        record.to_csv(self.path, index=False)
        results = scene_run.evaluate_scenes(
            pipelines.recorded_pipeline(self.path), made)
        self.assertTrue(results.particles.ok.all())


class TestDetectImages(unittest.TestCase):

    def test_detect(self) -> None:
        seen = []

        def app(image):
            seen.append(image)
            if len(seen) == 2:
                return pd.DataFrame()
            return pd.DataFrame(dict(
                label=['holo'], confidence=[.8], x_p=[1.], y_p=[2.],
                z_p=[200.], a_p=[1.], n_p=[1.5], framenum=[0]))

        with tempfile.TemporaryDirectory() as tmp:
            scenes.write_images(small_scenes(3), tmp)
            found, times = detect_images.detect(tmp, app)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0].ndim, 2)
        self.assertEqual(list(found.scene), [0, 2])
        self.assertNotIn('framenum', found.columns)
        self.assertIn('confidence', found.columns)
        self.assertEqual(list(times.scene), [0, 1, 2])

    def test_nothing_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scenes.write_images(small_scenes(1), tmp)
            found, _ = detect_images.detect(tmp, lambda image: [])
        self.assertEqual(len(found), 0)
        self.assertIn('scene', found.columns)

    def test_no_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             self.assertRaises(FileNotFoundError):
            detect_images.detect(tmp, lambda image: None)


class TestMatch(unittest.TestCase):

    def test_matches_within_the_gate(self) -> None:
        truth = [[10., 10.], [100., 100.], [200., 50.]]
        found = [[101., 99.], [11., 12.], [300., 300.]]
        m = match(truth, found, gate=15.)
        self.assertEqual(list(m.truth), [0, 1])
        self.assertEqual(list(m.detection), [1, 0])
        self.assertTrue(np.allclose(m.distance, [np.hypot(1, 2),
                                                 np.hypot(1, 1)]))
        self.assertEqual(list(m.missed), [2])
        self.assertEqual(list(m.spurious), [2])

    def test_a_detection_matches_only_one_particle(self) -> None:
        m = match([[0., 0.], [6., 0.]], [[5., 0.]], gate=15.)
        self.assertEqual(list(m.truth), [1])
        self.assertEqual(list(m.missed), [0])
        self.assertEqual(len(m.spurious), 0)

    def test_the_closest_pairs_come_first(self) -> None:
        m = match([[0., 0.]], [[10., 0.], [3., 0.]], gate=15.)
        self.assertEqual(list(m.detection), [1])
        self.assertEqual(list(m.spurious), [0])

    def test_nothing_found_or_nothing_there(self) -> None:
        m = match([[1., 1.]], np.empty((0, 2)))
        self.assertEqual((len(m.truth), list(m.missed)), (0, [0]))
        m = match(np.empty((0, 2)), [[1., 1.]])
        self.assertEqual((len(m.truth), list(m.spurious)), (0, [0]))


class TestCrop(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.arange(100 * 200, dtype=float).reshape(100, 200)

    def test_interior(self) -> None:
        crop, x0, y0 = pipelines.crop_around(self.image, 100, 50, 20)
        self.assertEqual((crop.shape, x0, y0), ((20, 20), 90, 40))
        self.assertEqual(crop[0, 0], self.image[40, 90])

    def test_shifted_inside_at_the_edges(self) -> None:
        for x, y, x0, y0 in ((2, 50, 0, 40), (199, 50, 180, 40),
                             (100, 1, 90, 0), (100, 99, 90, 80),
                             (-30, -30, 0, 0)):
            crop, cx, cy = pipelines.crop_around(self.image, x, y, 20)
            self.assertEqual(crop.shape, (20, 20))
            self.assertEqual((cx, cy), (x0, y0))

    def test_the_side_is_at_most_the_shorter_side(self) -> None:
        crop, x0, y0 = pipelines.crop_around(self.image, 100, 50, 500)
        self.assertEqual(crop.shape, (100, 100))


class TestPipelines(unittest.TestCase):

    def setUp(self) -> None:
        self.made = small_scenes(3, (2, 2), margin=200.)

    def test_truth_pipeline_returns_the_particles_in_the_image(self) -> None:
        for scene in self.made:
            found = pipelines.truth_pipeline()(scene)
            self.assertEqual(list(found.columns), list(pipelines.COLUMNS))
            self.assertEqual(len(found), int(scene.truth.inside.sum()))

    def test_estimator_on_truth(self) -> None:
        seen = []

        def estimator(image, x, y, instrument=None):
            seen.append((image, x, y, instrument))
            return pd.DataFrame(dict(z_p=[111.] * len(x), a_p=[2.] * len(x),
                                     n_p=[1.5] * len(x)))

        scene = small_scenes(1, (2, 2))[0]
        found = pipelines.estimator_on_truth(estimator)(scene)
        self.assertEqual(list(found.z_p), [111., 111.])
        self.assertTrue(np.array_equal(found.x_p, scene.truth.x_p))
        self.assertIs(seen[0][0], scene.image)
        self.assertTrue(np.array_equal(seen[0][1], scene.truth.x_p))
        self.assertEqual(seen[0][3], scene.instrument)

    def test_crops_on_truth(self) -> None:
        seen = []

        def estimator(crops):
            seen.extend(crops)
            return pd.DataFrame(dict(z_p=[111.] * len(crops),
                                     a_p=[2.] * len(crops),
                                     n_p=[1.5] * len(crops)))

        pipeline = pipelines.crops_on_truth(estimator, minimum=41)
        scene = small_scenes(1, (2, 2))[0]
        found = pipeline(scene)
        self.assertEqual(len(found), 2)
        self.assertEqual(list(found.z_p), [111., 111.])
        self.assertTrue(np.array_equal(found.x_p, scene.truth.x_p))
        self.assertEqual(len(seen), 2)
        for crop in seen:
            self.assertEqual(crop.shape[0], crop.shape[1])
            self.assertGreaterEqual(crop.shape[0], 41)

    def test_estimator_on_truth_without_particles_in_the_image(self) -> None:
        scene = scenes.Scene(np.ones(SHAPE), pd.DataFrame(
            dict(x_p=[-50.], y_p=[10.], z_p=[200.], a_p=[1.], n_p=[1.5],
                 inside=[False])), 0.05)
        found = pipelines.estimator_on_truth(
            lambda *args: pd.DataFrame())(scene)
        self.assertEqual(len(found), 0)
        self.assertEqual(list(found.columns), list(pipelines.COLUMNS))

    def test_catch_pipeline(self) -> None:
        seen = []

        def app(image):
            seen.append(image)
            return pd.DataFrame(dict(
                label=['holo'], confidence=[.9], x_p=[5.], y_p=[6.],
                z_p=[200.], a_p=[1.], n_p=[1.5], framenum=[0]),
                index=[7])

        scene = small_scenes(1)[0]
        found = pipelines.catch_pipeline(app)(scene)
        self.assertEqual(list(found.columns), list(pipelines.COLUMNS))
        self.assertEqual(list(found.index), [0])
        self.assertEqual(seen[0].dtype, np.uint8)
        self.assertAlmostEqual(seen[0].mean(), 100., delta=10.)

    def test_catch_pipeline_finds_nothing(self) -> None:
        found = pipelines.catch_pipeline(lambda image: pd.DataFrame())(
            small_scenes(1)[0])
        self.assertEqual(len(found), 0)
        self.assertEqual(list(found.columns), list(pipelines.COLUMNS))


def detections(scene, dx=0., dy=0., extra=None,
               **estimates) -> pd.DataFrame:
    '''The truth, displaced, with an estimate replaced, and extra rows'''
    out = pd.DataFrame(scene.truth[list(pipelines.COLUMNS)]).copy()
    out['x_p'] += dx
    out['y_p'] += dy
    for key, value in estimates.items():
        out[key] = value
    if extra is not None:
        out = pd.concat([out, pd.DataFrame([extra])], ignore_index=True)
    return out


class TestEvaluateScenes(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.made = small_scenes(2)

    def test_the_truth_locks_on(self) -> None:
        results = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            self.made)
        p = results.particles
        self.assertEqual(len(p), 2)
        self.assertTrue(p.found.all())
        self.assertTrue((p.center_err == 0).all())
        self.assertTrue((p.z_rel == 0).all())
        self.assertTrue(p.ok.all())
        self.assertEqual(list(results.scenes.tp), [1, 1])
        self.assertEqual(list(results.scenes.fp), [0, 0])
        for column in ('x_fit', 'a_fit', 'redchi', 'edge', 'neighbor'):
            self.assertIn(column, p.columns)

    def test_in_chunks_gives_the_same_results(self) -> None:
        made = small_scenes(3)
        whole = scene_run.evaluate_scenes(pipelines.truth_pipeline(), made)
        parts = scene_run.evaluate_scenes(pipelines.truth_pipeline(), made,
                                          chunk=2)
        self.assertEqual(list(parts.particles.scene), [0, 1, 2])
        self.assertEqual(list(parts.scenes.scene), [0, 1, 2])
        for name in ('a_fit', 'center_err', 'ok'):
            self.assertEqual(list(parts.particles[name]),
                             list(whole.particles[name]))

    def test_scenes_are_numbered_by_their_own_index(self) -> None:
        made = small_scenes(3)[1:]
        results = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            made)
        self.assertEqual(list(results.particles.scene), [1, 2])

    def test_in_several_processes(self) -> None:
        results = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            self.made, nprocs=2)
        self.assertTrue(results.particles.ok.all())

    def test_a_pipeline_that_finds_nothing(self) -> None:
        empty = pd.DataFrame(columns=list(pipelines.COLUMNS))
        results = scene_run.evaluate_scenes(lambda scene: empty, self.made)
        p = results.particles
        self.assertFalse(p.found.any() or p.ok.any())
        self.assertTrue(p.a_fit.isna().all())
        self.assertEqual(list(results.scenes.fn), [1, 1])
        self.assertEqual(list(results.scenes.ndetected), [0, 0])

    def test_a_spurious_detection(self) -> None:
        extra = dict(x_p=-1000., y_p=-1000., z_p=200., a_p=1., n_p=1.5)
        results = scene_run.evaluate_scenes(
            lambda scene: detections(scene, extra=extra), self.made)
        self.assertEqual(list(results.scenes.fp), [1, 1])
        self.assertTrue(results.particles.found.all())

    def test_the_center_is_off(self) -> None:
        results = scene_run.evaluate_scenes(
            lambda scene: detections(scene, dx=3., dy=4.), self.made)
        p = results.particles
        self.assertTrue(np.allclose(p.center_err, 5.))
        self.assertTrue(np.allclose(p.x_det - p.x_p, 3.))

    def test_a_detection_outside_the_gate_is_a_miss(self) -> None:
        results = scene_run.evaluate_scenes(
            lambda scene: detections(scene, dx=30.), self.made)
        self.assertFalse(results.particles.found.any())
        self.assertEqual(list(results.scenes.fp), [1, 1])
        self.assertEqual(list(results.scenes.fn), [1, 1])

    def test_estimates_that_cannot_start_a_fit(self) -> None:
        results = scene_run.evaluate_scenes(
            lambda scene: detections(scene, z_p=np.nan), self.made)
        p = results.particles
        self.assertTrue(p.found.all())
        self.assertTrue(p.a_fit.isna().all())
        self.assertFalse(p.ok.any())

    def test_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            scene_run.evaluate_scenes(
                lambda scene: pd.DataFrame(scene.truth[['x_p', 'y_p']]),
                self.made)


class TestSummaries(unittest.TestCase):

    def setUp(self) -> None:
        p = pd.DataFrame(dict(
            scene=[0, 0, 0, 1], index=[0, 1, 2, 0],
            nparticles=[3, 3, 3, 1], edge=[300., 10., -5., 40.],
            neighbor=[100., 100., 500., np.inf],
            inside=[True, True, False, True],
            found=[True, False, True, True],
            center_err=[2., np.nan, 4., 1.],
            z_p=[200.] * 4, a_p=[1.] * 4, n_p=[1.5] * 4,
            z_est=[210., np.nan, 200., 200.], a_est=[1., np.nan, 1., 1.],
            n_est=[1.5, np.nan, 1.5, 1.5],
            a_fit=[1.001, np.nan, 1.1, 1.002]))
        s = pd.DataFrame(dict(scene=[0, 1], nparticles=[3, 1],
                              ndetected=[3, 2], tp=[2, 1], fp=[1, 1],
                              fn=[1, 0], seconds=[1., 3.]))
        self.results = scene_run.SceneResults(scene_run.score(p), s)

    def test_score(self) -> None:
        p = self.results.particles
        self.assertEqual(list(p.ok), [True, False, False, True])
        self.assertAlmostEqual(p.z_rel[0], 0.05)

    def test_summary_counts_the_particles_in_the_image(self) -> None:
        summary = scene_run.summarize_scenes(self.results)
        self.assertEqual(summary.particles, 3)
        self.assertEqual(summary.scenes, 2)
        self.assertAlmostEqual(summary.recall, 2 / 3, places=3)
        self.assertAlmostEqual(summary.precision, 2 / 4, places=3)
        self.assertAlmostEqual(summary.lock_on, 2 / 3, places=3)
        self.assertEqual(summary.lock_on_found, 1.0)
        self.assertEqual(summary.center_med, 1.5)
        self.assertEqual(summary.seconds, 2.0)

    def test_summary_with_the_particles_outside(self) -> None:
        summary = scene_run.summarize_scenes(self.results,
                                             inside_only=False)
        self.assertEqual(summary.particles, 4)
        self.assertAlmostEqual(summary.recall, 3 / 4, places=3)

    def test_binned(self) -> None:
        table = scene_run.binned(self.results.particles, 'edge')
        self.assertEqual(table.n.sum(), 3)
        self.assertEqual(table.loc['[0.0, 25.0)', 'recall'], 0.0)
        table = scene_run.binned(self.results.particles, 'neighbor')
        self.assertEqual(table.loc['[400.0, inf)', 'n'], 1)   # the lone one
        table = scene_run.binned(self.results.particles, 'nparticles',
                                 bins=(1, 2, 4), inside_only=False)
        self.assertEqual(list(table.n), [1, 3])

    def test_report(self) -> None:
        text = scene_run.report(self.results)
        for words in ('precision', 'by nparticles', 'by edge',
                      'by neighbor'):
            self.assertIn(words, text)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
