'''The extent of a hologram, against the installed pylorenzmie'''
import unittest
from importlib.util import find_spec

import numpy as np

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

from pylorenzmie.theory import Sphere  # noqa: E402

from CATCH.utilities import mtd  # noqa: E402

INSTRUMENT = {'wavelength': 0.447, 'magnification': 0.048, 'n_m': 1.34}
PARTICLE = {'a_p': [0.5, 1.0], 'n_p': [1.4, 1.6], 'k_p': [0, 0],
            'x_p': [90, 110], 'y_p': [90, 110], 'z_p': [100, 300],
            'nspheres': [1, 2], 'names': ['hologram']}


def make_config(directory: str, **kwargs) -> dict:
    config = {'instrument': dict(INSTRUMENT),
              'particle': {k: (list(v) if isinstance(v, list) else v)
                           for k, v in PARTICLE.items()},
              'shape': [201, 201], 'noise': 0.05, 'ext_noise': 0.01,
              'directory': directory, 'imgtype': 'png', 'nframes': 2,
              'overwrite': True, 'max_overlaps': 1, 'scale_integer': False,
              'training': {'batchsize': 2},
              'train': {'nframes': 2}, 'test': {'nframes': 2},
              'eval': {'nframes': 2}}
    config.update(kwargs)
    return config


class TestModel(unittest.TestCase):

    def test_make_model(self) -> None:
        model = mtd.make_model((201, 201), INSTRUMENT)
        self.assertEqual(model.coordinates.shape, (3, 201*201))
        self.assertEqual(model.instrument.wavelength, 0.447)
        self.assertEqual(model.instrument.n_m, 1.34)
        model.particle = Sphere(a_p=1., n_p=1.59, x_p=100., y_p=100.,
                                z_p=200.)
        hologram = model.hologram()
        self.assertEqual(hologram.shape, (201*201,))
        # a normalized hologram has a mean of about 1
        self.assertAlmostEqual(hologram.mean(), 1., delta=0.05)

    def test_image_shape_is_rows_by_columns(self) -> None:
        model = mtd.make_model((40, 60), INSTRUMENT)
        x, y, z = model.coordinates
        self.assertEqual((x.max(), y.max(), z.max()), (59., 39., 0.))

    def test_particle_properties_are_plain_floats(self) -> None:
        sphere = Sphere(a_p=np.float64(1.), z_p=np.float64(200.))
        properties = mtd.particle_properties(sphere)
        self.assertEqual(sorted(properties),
                         ['a_p', 'k_p', 'n_p', 'x_p', 'y_p', 'z_p'])
        self.assertTrue(all(type(v) is float for v in properties.values()))

    def test_feature_extent(self) -> None:
        config = make_config('unused')
        sphere = Sphere(a_p=1., n_p=1.59, z_p=200.)
        extent = mtd.feature_extent(sphere, config)
        self.assertGreater(extent, 0)
        self.assertLessEqual(extent, 300)
        self.assertEqual(extent, mtd.feature_extent(sphere, config))
        # too few fringes in the field of view: the range is returned
        self.assertEqual(mtd.feature_extent(sphere, config, nfringes=10**6),
                         300)
        # a hologram far away has too few fringes in the range: it is
        # never longer than the range (a grid once ran into the next row)
        far = Sphere(a_p=2.24, n_p=2.07, z_p=597.)
        self.assertEqual(mtd.feature_extent(far, config), 300)
        for z in (50., 200., 416., 598.):
            sphere = Sphere(a_p=3., n_p=2., z_p=z)
            self.assertLessEqual(mtd.feature_extent(sphere, config), 300)
        # the fringes of a hologram spread out as the sphere moves away
        near = mtd.feature_extent(Sphere(a_p=1., n_p=1.59, z_p=100.),
                                  config, nfringes=5)
        far = mtd.feature_extent(Sphere(a_p=1., n_p=1.59, z_p=400.),
                                 config, nfringes=5)
        self.assertGreater(far, near)
