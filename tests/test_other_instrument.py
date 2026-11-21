'''The evaluation harness for an instrument other than the reference'''
import argparse
import unittest
from dataclasses import replace
from importlib.util import find_spec
from typing import Any

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pylorenzmie.theory import LorenzMie, Sphere  # noqa: E402

from CATCH.customization.evaluation import (  # noqa: E402
    fitting, pipelines, scan, scene_run, scenes)
from CATCH.customization.evaluation.cases import (  # noqa: E402
    INSTRUMENT, Z_RANGE)
from CATCH.customization.evaluation.sets import SETS  # noqa: E402
from CATCH.lib.instruments import pixel_ratio  # noqa: E402

#: a microscope with a pixel of 1.74 reference pixels (a scale in the
#: range that the shipped networks serve)
OTHER = dict(wavelength=0.532, magnification=0.1, n_m=1.33)
SHAPE = (256, 320)


def make(instrument=None, count=1, seed=2026, **kwargs):
    options: dict[str, Any] = dict(shape=SHAPE, radii=(1.,),
                                   indexes={'ps': 1.59}, seed=seed,
                                   instrument=instrument)
    options.update(kwargs)
    return scenes.make_scenes(count, (1, 1), **options)


class TestScenesOfAnInstrument(unittest.TestCase):

    def test_the_scene_knows_its_instrument(self) -> None:
        self.assertEqual(make()[0].instrument, INSTRUMENT)
        self.assertEqual(make(OTHER)[0].instrument, OTHER)

    def test_the_same_particles_as_another_instrument_sees_them(self) -> None:
        ref, other = make(count=3), make(OTHER, count=3)
        ratio = pixel_ratio(OTHER, INSTRUMENT)
        self.assertGreater(ratio, 1.5)
        for a, b in zip(ref, other, strict=True):
            for key in ('x_p', 'y_p', 'a_p', 'n_p'):
                self.assertTrue(np.array_equal(a.truth[key], b.truth[key]))
            # the heights are the same in wavelengths of the medium
            np.testing.assert_allclose(b.truth.z_p, a.truth.z_p / ratio)
            self.assertFalse(np.array_equal(a.image, b.image))

    def test_the_heights_stay_in_the_converted_range(self) -> None:
        ratio = pixel_ratio(OTHER, INSTRUMENT)
        for scene in make(OTHER, count=5, z_range=None):
            self.assertTrue(
                Z_RANGE[0] / ratio <= scene.truth.z_p[0] <= Z_RANGE[1] / ratio)

    def test_an_explicit_range_is_in_the_pixels_of_the_instrument(self
                                                                  ) -> None:
        scene = make(OTHER, z_range=(100., 110.))[0]
        self.assertTrue(100. <= scene.truth.z_p[0] <= 110.)

    def test_the_hologram_is_that_of_the_instrument(self) -> None:
        particles = pd.DataFrame(dict(x_p=[160.], y_p=[128.], z_p=[150.],
                                      a_p=[1.], n_p=[1.59]))
        image = scenes.render(particles, SHAPE, 0., np.random.default_rng(0),
                              OTHER)
        model = LorenzMie()
        model.instrument.properties = dict(OTHER)
        model.coordinates = model.meshgrid(SHAPE)
        model.particle = Sphere(x_p=160., y_p=128., z_p=150., a_p=1.,
                                n_p=1.59)
        np.testing.assert_allclose(image, model.hologram().reshape(SHAPE))
        reference = scenes.render(particles, SHAPE, 0.,
                                  np.random.default_rng(0))
        self.assertGreater(np.abs(image - reference).max(), 0.05)

    def test_a_set_makes_its_scenes_for_an_instrument(self) -> None:
        same = SETS['dev'].make(1)[0]
        other = SETS['dev'].make(1, OTHER)[0]
        self.assertEqual(same.instrument, INSTRUMENT)
        self.assertEqual(other.instrument, OTHER)
        self.assertTrue(np.array_equal(same.truth.x_p, other.truth.x_p))
        self.assertFalse(np.array_equal(same.truth.z_p, other.truth.z_p))


class TestFitsOfAnInstrument(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = make(OTHER)[0]
        cls.truth = cls.scene.truth.iloc[0]
        cls.job = scene_run.make_job(0, cls.scene, cls.truth)
        assert cls.job is not None

    def test_the_extent_is_in_pixels_of_the_instrument(self) -> None:
        t = self.truth
        other = pipelines.extent(t.z_p, t.a_p, t.n_p, OTHER)
        same = pipelines.extent(t.z_p, t.a_p, t.n_p)
        self.assertNotEqual(other, same)
        self.assertEqual(pipelines.extent(t.z_p, t.a_p, t.n_p, INSTRUMENT),
                         same)

    def test_the_job_carries_the_instrument(self) -> None:
        self.assertEqual(self.job.instrument, OTHER)

    def test_the_bounds_of_z_follow_the_pixel(self) -> None:
        self.assertIsNone(fitting.fit_limits(INSTRUMENT))
        limits = fitting.fit_limits(OTHER)
        assert limits is not None
        ratio = pixel_ratio(OTHER, INSTRUMENT)
        self.assertAlmostEqual(limits['z_p'][0], 20. / ratio)
        self.assertAlmostEqual(limits['z_p'][1], 3000. / ratio)
        self.assertEqual(limits['a_p'], (0.1, 10.))

    def test_a_fit_from_the_truth_locks_on(self) -> None:
        result = scene_run.fit_job(self.job)
        self.assertTrue(fitting.locks_on(result['a_p'], self.truth.a_p),
                        result)

    def test_the_fit_needs_the_instrument(self) -> None:
        # the same job, fitted as if the reference instrument made it
        wrong = replace(self.job, instrument=dict(INSTRUMENT))
        result = scene_run.fit_job(wrong)
        self.assertFalse(fitting.locks_on(result['a_p'], self.truth.a_p))

    def test_the_scan_uses_the_instrument(self) -> None:
        small = scan.Scan(z_span=0.1, a_span=0.1, c_span=0.1, spacing=0.05,
                          keep=3, npixels=500)
        starts, chi = scan.scan(self.job.data, self.scene.noise,
                                self.job.start, small, instrument=OTHER)
        self.assertEqual(np.unravel_index(np.argmin(chi), chi.shape),
                         (2, 2, 2))
        self.assertLess(chi[2, 2, 2], 3.)
        _, wrong = scan.scan(self.job.data, self.scene.noise, self.job.start,
                             small)
        self.assertGreater(wrong[2, 2, 2], chi[2, 2, 2])

    def test_evaluate_scenes_with_the_truth(self) -> None:
        results = scene_run.evaluate_scenes(pipelines.truth_pipeline(),
                                            [self.scene])
        self.assertTrue(results.particles.ok.all())


class TestPipelinesOfAnInstrument(unittest.TestCase):

    def test_the_references_are_for_the_reference_instrument(self) -> None:
        pipeline = pipelines.crops_on_truth(lambda crops: pd.DataFrame())
        with self.assertRaisesRegex(ValueError, 'reference instrument'):
            pipeline(make(OTHER)[0])

    def test_the_estimator_gets_the_instrument_of_the_scene(self) -> None:
        seen = []

        def estimator(image, x, y, instrument=None):
            seen.append(instrument)
            return pd.DataFrame(dict(z_p=[100.] * len(x), a_p=[1.] * len(x),
                                     n_p=[1.5] * len(x)))

        pipelines.estimator_on_truth(estimator)(make(OTHER)[0])
        self.assertEqual(seen, [OTHER])


class TestCommandLine(unittest.TestCase):

    def test_no_option_is_the_reference_instrument(self) -> None:
        parser = argparse.ArgumentParser()
        scenes.add_instrument_option(parser)
        args = parser.parse_args([])
        self.assertIsNone(scenes.instrument_from_option(args.instrument))

    def test_the_option_names_the_properties(self) -> None:
        parser = argparse.ArgumentParser()
        scenes.add_instrument_option(parser)
        args = parser.parse_args(['--instrument', '0.532', '0.1', '1.33'])
        self.assertEqual(scenes.instrument_from_option(args.instrument),
                         OTHER)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
