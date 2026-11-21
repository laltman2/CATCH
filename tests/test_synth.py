'''Synthetic training frames'''
import unittest
from importlib.util import find_spec

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from CATCH.evaluation import scenes  # noqa: E402
from CATCH.evaluation.cases import INSTRUMENT  # noqa: E402
from CATCH.evaluation.pipelines import extent  # noqa: E402
from CATCH.instruments import REFERENCE  # noqa: E402
from CATCH.training import synth  # noqa: E402

SMALL = synth.Distribution(shape=(320, 320), nparticles=(2, 4),
                           a_p=(0.5, 2.), z_p=(150., 400.))


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


class TestInstrument(unittest.TestCase):

    def test_instrument_constants(self) -> None:
        self.assertEqual(synth.Distribution().magnification,
                         INSTRUMENT['magnification'])
        self.assertEqual(dict(INSTRUMENT), dict(REFERENCE))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
