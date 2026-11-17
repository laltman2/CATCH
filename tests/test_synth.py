'''Synthetic training data: frames, crops, and the stream of them'''
import unittest
from importlib.util import find_spec
from unittest.mock import patch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from CATCH.Estimator import Estimator  # noqa: E402
from CATCH.evaluation import scenes  # noqa: E402
from CATCH.evaluation.cases import INSTRUMENT  # noqa: E402
from CATCH.evaluation.pipelines import extent  # noqa: E402
from CATCH.evaluation.sets import SETS  # noqa: E402
from CATCH.instruments import REFERENCE  # noqa: E402
from CATCH.training import synth  # noqa: E402

SMALL = synth.Distribution(shape=(320, 320), nparticles=(2, 4),
                           a_p=(0.5, 2.), z_p=(150., 400.))


class TestMix(unittest.TestCase):

    def test_both_modes(self) -> None:
        frame = synth.make_frame(np.random.default_rng(3), SMALL,
                                 synth.Renderer(SMALL.shape),
                                 synth.Extent())
        with patch.object(synth, 'pad_around',
                          wraps=synth.pad_around) as padded, \
                patch.object(synth, 'crop_around',
                             wraps=synth.crop_around) as shifted:
            crops = synth.make_crops(
                frame, np.random.default_rng(0),
                synth.CropPolicy(mode='mix', repeats=20))
        self.assertEqual(padded.call_count + shifted.call_count,
                         len(crops.targets))
        self.assertGreater(padded.call_count, 0)
        self.assertGreater(shifted.call_count, 0)


class TestRepeats(unittest.TestCase):

    def test_repeats(self) -> None:
        frame = synth.make_frame(np.random.default_rng(3), SMALL,
                                 synth.Renderer(SMALL.shape),
                                 synth.Extent())
        one = synth.make_crops(frame, np.random.default_rng(0))
        three = synth.make_crops(frame, np.random.default_rng(0),
                                 synth.CropPolicy(repeats=3))
        self.assertEqual(len(three.targets), 3 * len(one.targets))
        # the same particle, with independent errors of the center
        self.assertTrue(torch.equal(three.targets[0], three.targets[1]))
        self.assertFalse(torch.equal(three.images[0], three.images[1]))


class TestParticles(unittest.TestCase):

    def test_ranges(self) -> None:
        rng = np.random.default_rng(0)
        dist = synth.Distribution(nparticles=(5, 5))
        frames = pd.concat([synth.sample_particles(rng, dist)
                            for _ in range(200)])
        self.assertEqual(len(frames), 1000)
        self.assertTrue(frames.a_p.between(*dist.a_p).all())
        self.assertTrue(frames.n_p.between(*dist.n_p).all())
        self.assertTrue(frames.z_p.between(*dist.z_p).all())
        # uniform in the logarithm of the radius
        self.assertAlmostEqual(np.median(np.log(frames.a_p)),
                               np.log(dist.a_p).mean(), delta=0.15)

    def test_number_of_particles(self) -> None:
        rng = np.random.default_rng(1)
        dist = synth.Distribution(nparticles=(0, 10))
        counts = {len(synth.sample_particles(rng, dist))
                  for _ in range(300)}
        self.assertEqual(counts, set(range(11)))

    def test_the_plane_of_the_image_is_outside_the_sphere(self) -> None:
        rng = np.random.default_rng(2)
        dist = synth.Distribution(nparticles=(10, 10))
        p = pd.concat([synth.sample_particles(rng, dist)
                       for _ in range(100)])
        clearance = p.z_p * dist.magnification / p.a_p
        self.assertGreaterEqual(clearance.min(), dist.clearance - 1e-9)
        # the evaluation sets are within what is trained
        self.assertLess(dist.clearance, 150. * 0.048 / 5.)

    def test_a_range_of_z_that_cannot_hold_the_radii(self) -> None:
        dist = synth.Distribution(a_p=(4., 5.), z_p=(50., 100.),
                                  nparticles=(5, 5))
        with self.assertRaises(ValueError):
            synth.sample_particles(np.random.default_rng(0), dist)

    def test_separation_and_margin(self) -> None:
        rng = np.random.default_rng(3)
        dist = synth.Distribution(shape=(400, 400), nparticles=(8, 8),
                                  min_separation=40., margin=50.)
        for _ in range(20):
            p = synth.sample_particles(rng, dist)
            d = np.hypot(p.x_p.to_numpy()[:, None] - p.x_p.to_numpy(),
                         p.y_p.to_numpy()[:, None] - p.y_p.to_numpy())
            np.fill_diagonal(d, np.inf)
            self.assertGreaterEqual(d.min(), 40.)
            self.assertTrue(p.x_p.between(-50, 450).all())

    def test_particles_are_put_at_the_edges_and_corners(self) -> None:
        rng = np.random.default_rng(4)
        dist = synth.Distribution(shape=(1000, 1000), nparticles=(6, 6),
                                  margin=0., edge_bias=0., corner_bias=1.,
                                  band=100.)
        p = pd.concat([synth.sample_particles(rng, dist)
                       for _ in range(30)])
        x = np.minimum(p.x_p, 1000 - p.x_p)
        y = np.minimum(p.y_p, 1000 - p.y_p)
        self.assertTrue(((x <= 100) & (y <= 100)).all())
        dist = synth.Distribution(shape=(1000, 1000), nparticles=(6, 6),
                                  margin=0., edge_bias=1., corner_bias=0.,
                                  band=100.)
        p = pd.concat([synth.sample_particles(rng, dist)
                       for _ in range(30)])
        edge = np.minimum.reduce([p.x_p, 1000 - p.x_p, p.y_p, 1000 - p.y_p])
        self.assertTrue((edge <= 100).all())

    def test_the_band_stays_in_the_frame(self) -> None:
        rng = np.random.default_rng(0)
        dist = synth.Distribution(shape=(30, 30), nparticles=(4, 4),
                                  margin=0., min_separation=1.,
                                  corner_bias=1., band=150.)
        p = pd.concat([synth.sample_particles(rng, dist)
                       for _ in range(30)])
        self.assertTrue(p.x_p.between(0, 30).all())
        self.assertTrue(p.y_p.between(0, 30).all())

    def test_no_room(self) -> None:
        dist = synth.Distribution(shape=(30, 30), nparticles=(5, 5),
                                  min_separation=100., max_attempts=20,
                                  margin=0.)
        with self.assertRaises(RuntimeError):
            synth.sample_particles(np.random.default_rng(0), dist)


class TestRendering(unittest.TestCase):

    def test_agrees_with_the_scenes_of_the_harness(self) -> None:
        rng = np.random.default_rng(5)
        particles = synth.sample_particles(rng, SMALL)
        ours = synth.Renderer(SMALL.shape)(particles, 0., rng)
        theirs = scenes.render(particles, SMALL.shape, 0.,
                               np.random.default_rng(0))
        np.testing.assert_allclose(ours, theirs, atol=3e-3)
        self.assertEqual(ours.dtype, np.float32)

    def test_no_particles_is_a_noisy_background(self) -> None:
        image = synth.Renderer((64, 64))(
            pd.DataFrame(columns=['x_p', 'y_p', 'z_p', 'a_p', 'n_p']),
            0.05, np.random.default_rng(0))
        self.assertAlmostEqual(float(image.mean()), 1., delta=0.01)
        self.assertAlmostEqual(float(image.std()), 0.05, delta=0.01)

    def test_extent_is_the_one_of_the_harness(self) -> None:
        measure = synth.Extent()
        for z, a, n in ((300., 1., 1.59), (150., 5., 1.45),
                        (550., 0.3, 2.0)):
            self.assertEqual(measure(z, a, n), extent(z, a, n))

    def test_frame(self) -> None:
        rng = np.random.default_rng(6)
        frame = synth.make_frame(rng, SMALL, synth.Renderer(SMALL.shape))
        self.assertEqual(frame.image.shape, SMALL.shape)
        for column in ('x_p', 'a_p', 'extent', 'edge', 'neighbor',
                       'inside'):
            self.assertIn(column, frame.particles.columns)
        self.assertTrue(SMALL.noise[0] <= frame.noise <= SMALL.noise[1])

    def test_a_hologram_that_is_not_finite_is_refused(self) -> None:
        def broken(particles, noise, rng):
            return np.full(SMALL.shape, np.nan, dtype=np.float32)
        with self.assertRaises(FloatingPointError):
            synth.make_frame(np.random.default_rng(0), SMALL, broken)  # type: ignore[arg-type]


class TestCrops(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dist = synth.Distribution(
            shape=(512, 512), nparticles=(3, 3), a_p=(0.5, 1.5),
            z_p=(150., 300.), margin=0., edge_bias=0.5, corner_bias=0.2)
        rng = np.random.default_rng(7)
        cls.frame = synth.make_frame(rng, cls.dist,
                                     synth.Renderer(cls.dist.shape))

    def test_prepare_is_what_the_estimator_does(self) -> None:
        config = {'shape': [201, 201], 'instrument': REFERENCE,
                  'particle': {'z_p': [50, 600], 'a_p': [0.2, 5.0],
                               'n_p': [1.38, 2.5]}}
        with patch.object(Estimator, '_load_model'), \
             patch.object(Estimator, '_load_config', return_value=config):
            estimator = Estimator()
        rng = np.random.default_rng(8)
        for side in (60, 201, 333):
            crop = 1. + 0.2 * rng.normal(size=(side, side))
            expected = estimator._load_image(100. * crop)[0]
            np.testing.assert_allclose(synth.prepare(100. * crop),
                                       expected, atol=1e-6)

    def test_samples(self) -> None:
        rng = np.random.default_rng(9)
        samples = synth.make_crops(self.frame, rng)
        inside = self.frame.particles[self.frame.particles.inside]
        n = len(inside)
        self.assertEqual(samples.images.shape, (n, 1, 201, 201))
        self.assertEqual(samples.scale.shape, (n, 1))
        self.assertEqual(samples.targets.shape, (n, 3))
        np.testing.assert_allclose(
            samples.targets.numpy(),
            np.asarray(inside[['z_p', 'a_p', 'n_p']], dtype=np.float32),
            rtol=1e-6)
        np.testing.assert_allclose(
            samples.scale[:, 0].numpy(),
            np.asarray(samples.meta.side) / 201., rtol=1e-6)
        self.assertGreaterEqual(float(samples.images.min()), 0.)
        self.assertLessEqual(float(samples.images.max()), 1.)

    def test_padding_keeps_the_particle_at_the_center(self) -> None:
        rng = np.random.default_rng(10)
        pad = synth.make_crops(
            self.frame, rng, synth.CropPolicy('pad', jitter=0., tail=0.,
                                              side_noise=0.))
        shift = synth.make_crops(
            self.frame, rng, synth.CropPolicy('shift', jitter=0., tail=0.,
                                              side_noise=0.))
        self.assertLess(pad.meta.offset.max(), 2.)
        near = shift.meta.edge < shift.meta.side / 2
        if near.any():
            self.assertGreater(shift.meta.offset[near].max(), 10.)

    def test_a_frame_without_particles_inside(self) -> None:
        empty = synth.Frame(
            np.ones((64, 64), dtype=np.float32),
            pd.DataFrame(columns=['x_p', 'y_p', 'z_p', 'a_p', 'n_p',
                                  'extent', 'edge', 'neighbor', 'inside']),
            0.05)
        samples = synth.make_crops(empty, np.random.default_rng(0))
        self.assertEqual(samples.images.shape, (0, 1, 201, 201))
        self.assertEqual(len(samples.meta), 0)


class TestStream(unittest.TestCase):

    def test_the_seed_names_the_data(self) -> None:
        a = next(synth.Stream(11, SMALL).frames())
        b = next(synth.Stream(11, SMALL).frames())
        c = next(synth.Stream(12, SMALL).frames())
        np.testing.assert_array_equal(a.image, b.image)
        self.assertFalse(np.array_equal(a.image, c.image))

    def test_the_seeds_of_the_evaluation_sets_are_refused(self) -> None:
        for seed in {s.seed for s in SETS.values()} | {2026}:
            with self.assertRaises(ValueError):
                synth.Stream(seed, SMALL)

    def test_batches_of_crops(self) -> None:
        stream = synth.Stream(13, SMALL)
        batches = stream.crops(8)
        for _ in range(3):
            batch = next(batches)
            self.assertEqual(batch.images.shape, (8, 1, 201, 201))
            self.assertEqual(batch.targets.shape, (8, 3))
            self.assertEqual(len(batch.meta), 8)
            self.assertIsInstance(batch.scale, torch.Tensor)

    def test_instrument_constants(self) -> None:
        self.assertEqual(synth.Distribution().magnification,
                         INSTRUMENT['magnification'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
