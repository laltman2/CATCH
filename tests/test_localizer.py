'''The Localizer: detections, the polished centers and their guard'''
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import random_models

# the module, not the class of the same name that CATCH exports
L = importlib.import_module('CATCH.Localizer')


def rings(shape, x, y, scale=1.):
    '''A hologram-like pattern centered on (x, y)'''
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r = np.hypot(xx - x, yy - y) * scale
    return (1. + 0.3 * np.cos(r ** 2 / 400.) * np.exp(-r / 150.)
            ).astype(np.float32)


class TestHelpers(unittest.TestCase):

    def test_normalize(self) -> None:
        image = np.full((4, 4), 100, np.uint8)
        self.assertTrue(np.allclose(L.normalize(image), 1.))
        self.assertTrue(np.allclose(L.normalize(np.ones((4, 4))), 1.))
        color = np.ones((4, 4, 3))
        self.assertEqual(L.normalize(color).shape, (4, 4))

    def test_model_directory(self) -> None:
        self.assertTrue(
            (L.model_directory('localizer', None) / 'weights.pt').is_file())
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(L.model_directory('localizer', d), Path(d))
        with self.assertRaises(FileNotFoundError):
            L.model_directory('localizer', 'no such model')

    def test_default_device(self) -> None:
        self.assertIn(L.default_device().type, ('cpu', 'cuda', 'mps'))

    def test_symmetry_center(self) -> None:
        image = rings((300, 320), 160.4, 140.7)
        x, y = L.symmetry_center(image, 163., 138.)
        self.assertAlmostEqual(x, 160.4, delta=0.2)
        self.assertAlmostEqual(y, 140.7, delta=0.2)

    def test_symmetry_center_near_an_edge(self) -> None:
        image = rings((300, 320), 30.3, 150.2)
        x, y = L.symmetry_center(image, 32., 151.)
        self.assertAlmostEqual(x, 30.3, delta=0.5)
        self.assertAlmostEqual(y, 150.2, delta=0.5)


class TestLocalizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.mkdtemp()
        lpath, _ = random_models.write_models(cls.directory)
        cls.localizer = L.Localizer(lpath, device='cpu')

    @classmethod
    def tearDownClass(cls) -> None:
        random_models.remove(cls.directory)

    def detections(self, x, y, s):
        return (np.array([x], float), np.array([y], float),
                np.array([s], float))

    def test_configuration(self) -> None:
        self.assertEqual(self.localizer.scales, (1.0, 2.5))
        self.assertGreater(self.localizer.threshold, 0)

    def test_output(self) -> None:
        image = rings((256, 256), 128., 128.)
        found = self.localizer(image)
        self.assertEqual(list(found.columns), list(L.COLUMNS))
        found = self.localizer([image, image])
        self.assertEqual(len(found), 2)

    def test_threshold_and_guard(self) -> None:
        image = rings((300, 320), 160.4, 140.7)
        peaks = self.detections([160., 250., 40.], [141., 100., 40.],
                                [0.9, 0.8, 0.1])
        peaks = tuple(p[None, 0] for p in peaks)
        with patch.object(L, 'peaks', return_value=peaks):
            found = self.localizer(image)
        self.assertEqual(len(found), 2)           # the weak one is dropped
        # near the true center, the polished center is used
        self.assertAlmostEqual(found.x_p[0], 160.4, delta=0.2)
        # far from any symmetric pattern, the detector's center is kept
        self.assertEqual((found.x_p[1], found.y_p[1]), (250., 100.))
        self.assertEqual(found.label[0], 'hologram')

    def test_frames_outside_the_trained_scales_are_resized(self) -> None:
        image = np.ones((200, 300), np.float32)
        seen = []

        def fake(net, images, count):
            seen.append(tuple(images.shape))
            return self.detections([[10.]], [[20.]], [[0.9]])

        with patch.object(L, 'peaks', side_effect=fake):
            found = self.localizer(image, scale=5.)
        self.assertEqual(seen[0], (1, 400, 600))   # enlarged to scale 2.5
        self.assertAlmostEqual(found.x_p[0], (10 + .5) / 2 - .5)
        with patch.object(L, 'peaks', side_effect=fake):
            self.localizer(image, scale=0.5)
        self.assertEqual(seen[1], (1, 100, 150))   # reduced to scale 1


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
