'''Data generation, against the installed pylorenzmie (the ``train`` extra)'''
import ast
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

from pylorenzmie.theory import Sphere  # noqa: E402

from CATCH.training import Torch_DataLoader, YOLO_data_generator  # noqa: E402
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
        # the fringes of a hologram spread out as the sphere moves away
        near = mtd.feature_extent(Sphere(a_p=1., n_p=1.59, z_p=100.),
                                  config, nfringes=5)
        far = mtd.feature_extent(Sphere(a_p=1., n_p=1.59, z_p=400.),
                                 config, nfringes=5)
        self.assertGreater(far, near)


class TestSampling(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)
        self.config = make_config('unused')

    def test_make_value(self) -> None:
        self.assertEqual(mtd.make_value(2.5), 2.5)
        self.assertEqual(mtd.make_value([3, 3]), 3)
        for _ in range(20):
            self.assertTrue(1. <= mtd.make_value([1., 2.]) <= 2.)

    def test_make_value_of_several_ranges(self) -> None:
        for _ in range(20):
            value = YOLO_data_generator.make_value([[1., 2.], [5., 6.]])
            self.assertTrue(1. <= value <= 2. or 5. <= value <= 6.)

    def test_make_sample(self) -> None:
        # the field is wide enough that three spheres can fit
        self.config['particle'].update(nspheres=[3, 3], x_p=[0, 2000],
                                       y_p=[0, 2000])
        particle = self.config['particle']
        sample = mtd.make_sample(self.config)
        self.assertEqual(len(sample), 3)
        for sphere in sample:
            self.assertTrue(0.5 <= sphere.a_p <= 1.0)
            self.assertTrue(100 <= sphere.z_p <= 300)
            self.assertTrue(particle['x_p'][0] <= sphere.x_p
                            <= particle['x_p'][1])

    def test_spheres_do_not_overlap(self) -> None:
        self.config['particle'].update(nspheres=[3, 3], x_p=[0, 2000],
                                       y_p=[0, 2000])
        mpp = INSTRUMENT['magnification']
        sample = mtd.make_sample(self.config)
        for i, a in enumerate(sample):
            for b in sample[:i]:
                distance = np.linalg.norm([a.x_p - b.x_p, a.y_p - b.y_p,
                                           a.z_p - b.z_p])
                self.assertGreaterEqual(distance, (a.a_p + b.a_p)/mpp)

    def test_impossible_separation_is_reported(self) -> None:
        '''Spheres that cannot be kept apart raise an error, not a hang'''
        # equal z_p and a_p, in a field that is narrower than the separation
        # of 2 a_p / magnification = 42 pixels
        self.config['particle'].update(nspheres=[2, 2], z_p=[200, 200],
                                       a_p=[1., 1.])
        for module in (mtd, YOLO_data_generator):
            with self.subTest(module.__name__):
                with self.assertRaises(RuntimeError) as context:
                    module.make_sample(self.config, max_attempts=50)
                self.assertIn('50 attempts', str(context.exception))

    def test_yolo_sample(self) -> None:
        self.config['particle'].update(nspheres=[2, 2], x_p=[0, 2000],
                                       y_p=[0, 2000])
        sample = YOLO_data_generator.make_sample(self.config)
        self.assertEqual(len(sample), 2)

    def test_add_overlaps_lie_in_an_annulus(self) -> None:
        ext = 50.
        spheres = mtd.add_overlaps(ext, 4, self.config)
        self.assertEqual(len(spheres), 4)
        for sphere in spheres:
            r = np.hypot(sphere.x_p - ext, sphere.y_p - ext)
            self.assertTrue(1.5*ext <= r <= 2*ext)


class TestAnnotations(unittest.TestCase):

    def setUp(self) -> None:
        self.config = make_config('unused')
        self.sample = [Sphere(a_p=1., n_p=1.59, x_p=100., y_p=50.,
                              z_p=200.)]

    def test_format_json(self) -> None:
        for module in (mtd, YOLO_data_generator):
            with self.subTest(module.__name__):
                annotation = json.loads(
                    module.format_json(self.sample, self.config))
                params = ast.literal_eval(annotation[0])
                self.assertEqual(params['z_p'], 200.)
                self.assertEqual(params['a_p'], 1.)
                self.assertEqual(params['n_p'], 1.59)

    def test_format_json_with_scale(self) -> None:
        text = Torch_DataLoader.format_json(self.sample, self.config,
                                            scale=1.5, num_overlaps=2)
        params = ast.literal_eval(json.loads(text)[0])
        self.assertEqual((params['scale'], params['num_overlaps']), (1.5, 2))
        self.assertEqual(params['z_p'], 200.)

    def test_format_yolo(self) -> None:
        for module in (mtd, YOLO_data_generator):
            with self.subTest(module.__name__):
                line = module.format_yolo(self.sample, self.config).split()
                self.assertEqual(line[0], '0')
                x, y, w, h = (float(v) for v in line[1:])
                self.assertAlmostEqual(x, 100/201, places=5)
                self.assertAlmostEqual(y, 50/201, places=5)
                self.assertGreater(w, 0)
                self.assertEqual(w, h)


class TestGenerators(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(1)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)

    def test_mtd(self) -> None:
        out = self.directory / 'mtd'
        config = make_config(str(out), shape=[64, 64])
        configfile = self.directory / 'mtd.json'
        configfile.write_text(json.dumps(config))
        mtd.mtd(str(configfile))
        images = sorted((out / 'images_labels').glob('*.png'))
        self.assertEqual(len(images), 2)
        image = cv2.imread(str(images[0]), 0)
        assert image is not None
        self.assertEqual(image.shape, (64, 64))
        self.assertEqual(len(list((out / 'params').glob('*.json'))), 2)
        names = (out / 'filenames.txt').read_text().split()
        self.assertEqual(len(names), 2)

    def test_yolo_makedata(self) -> None:
        out = self.directory / 'yolo'
        config = make_config(str(out), shape=[64, 64])
        YOLO_data_generator.makedata(config)
        self.assertEqual(len(list((out / 'images').glob('*.png'))), 2)
        self.assertEqual(len(list((out / 'labels').glob('*.txt'))), 2)
        label = (out / 'labels' / 'image00000.txt').read_text().split()
        self.assertEqual(len(label) % 5, 0)
        # without overwrite, nothing is made again when the files exist
        config['overwrite'] = False
        before = (out / 'images' / 'image00000.png').stat().st_mtime_ns
        YOLO_data_generator.makedata(config)
        after = (out / 'images' / 'image00000.png').stat().st_mtime_ns
        self.assertEqual(before, after)

    def check_estimator_data(self, scale_integer: bool) -> None:
        out = self.directory / ('int' if scale_integer else 'float')
        config = make_config(str(out) + '/', scale_integer=scale_integer)
        Torch_DataLoader.makedata(config)
        for settype in ('train', 'test', 'eval'):
            self.assertEqual(
                len(list((out / settype / 'images').glob('*.png'))), 2)
        dataset = Torch_DataLoader.EstimatorDataset(config)
        self.assertEqual(len(dataset), 2)
        image, scale, outputs = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 201, 201))
        self.assertEqual(tuple(scale.shape), (1,))
        self.assertGreater(scale.item(), 0)
        self.assertEqual(tuple(outputs.shape), (3,))

    def test_missing_image_is_reported(self) -> None:
        out = self.directory / 'missing'
        config = make_config(str(out) + '/')
        Torch_DataLoader.makedata(config)
        (out / 'train' / 'images' / 'image00000.png').unlink()
        dataset = Torch_DataLoader.EstimatorDataset(config)
        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_estimator_data_with_float_scale(self) -> None:
        self.check_estimator_data(scale_integer=False)

    def test_estimator_data_with_integer_scale(self) -> None:
        self.check_estimator_data(scale_integer=True)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
