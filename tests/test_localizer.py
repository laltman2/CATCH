import importlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from conftest import make_fake_yolov5


def load_localizer_module():
    '''(Re)imports CATCH.Localizer against the stub yolov5.'''
    sys.modules.pop('CATCH.Localizer', None)
    return importlib.import_module('CATCH.Localizer')


def detections(*boxes) -> pd.DataFrame:
    '''A frame like yolov5's: (xmin, ymin, xmax, ymax, confidence)'''
    rows = [dict(xmin=x0, ymin=y0, xmax=x1, ymax=y1, confidence=c,
                 **{'class': 0, 'name': 'hologram'})
            for x0, y0, x1, y1, c in boxes]
    return pd.DataFrame(rows)


class TestImport(unittest.TestCase):

    def test_new_yolov5_is_refused(self) -> None:
        too_new = make_fake_yolov5('7.0.0')
        with patch.dict(sys.modules, {'yolov5': too_new}), \
             self.assertRaises(ImportError):
            load_localizer_module()
        load_localizer_module()

    def test_lazy_attributes_are_classes(self) -> None:
        import CATCH
        for name in ('Localizer', 'CATCHapp'):
            sys.modules.pop(f'CATCH.{name}', None)
        self.assertEqual(type(CATCH.Localizer), type)
        self.assertEqual(type(CATCH.CATCHapp), type)
        self.assertFalse(hasattr(CATCH, 'NoSuchThing'))

    def test_importing_a_submodule_does_not_hide_the_class(self) -> None:
        '''Importing CATCH.CATCHapp binds the module, not the class.'''
        import CATCH
        for name in ('Localizer', 'CATCHapp'):
            sys.modules.pop(f'CATCH.{name}', None)
        module = importlib.import_module('CATCH.CATCHapp')
        self.assertEqual(type(module), type(sys))
        self.assertEqual(type(CATCH.CATCHapp), type)
        self.assertEqual(type(CATCH.Localizer), type)
        from CATCH import Localizer
        self.assertEqual(Localizer.__name__, 'Localizer')

    def test_star_import_needs_no_yolov5(self) -> None:
        import CATCH
        self.assertEqual(CATCH.__all__, ['Estimator'])


class TestLocalizer(unittest.TestCase):

    def setUp(self) -> None:
        self.module = load_localizer_module()
        self.localizer = self.module.Localizer()

    def test_default_model_path(self) -> None:
        path = self.localizer.model_path
        self.assertEqual(path.parts[-3:],
                         ('cfg_yolov5', 'small_noisy', 'weights.pt'))

    def test_named_model_and_device(self) -> None:
        localizer = self.module.Localizer(model='other', device='cpu')
        self.assertEqual(localizer.model_path.parts[-2], 'other')
        self.assertEqual(localizer.device, 'cpu')

    def test_the_checkpoint_may_hold_a_whole_model(self) -> None:
        '''torch.load unpickles the model that the weights file holds,
        for the weights of CATCH only'''
        import functools

        import torch
        original = torch.load
        load = self.localizer.torch_load
        self.assertIsInstance(load, functools.partial)
        self.assertEqual(load.keywords, {'weights_only': False})
        self.assertIs(torch.load, original)

    def test_the_checkpoint_loader_is_restored_after_an_error(self) -> None:
        import torch
        original = torch.load
        with self.assertRaises(RuntimeError), \
             self.module.trusted_checkpoints():
            raise RuntimeError
        self.assertIs(torch.load, original)

    def test_detect_is_localize(self) -> None:
        self.assertEqual(self.localizer.detect, self.localizer.localize)

    def test_find_center_interior(self) -> None:
        p = pd.Series(dict(xmin=40, xmax=60, ymin=30, ymax=70))
        x_p, y_p, edge = self.localizer.find_center(p, (100, 200))
        self.assertEqual((x_p, y_p, edge), (50., 50., False))

    def test_find_center_edges(self) -> None:
        # box 20 wide, 40 tall: ext = 40, in an image 100 high, 200 wide
        shape = (100, 200, 3)
        cases = {
            'left': dict(xmin=-10, xmax=10, ymin=30, ymax=70),
            'right': dict(xmin=190, xmax=210, ymin=30, ymax=70),
            'bottom': dict(xmin=40, xmax=60, ymin=-10, ymax=30),
            'top': dict(xmin=40, xmax=60, ymin=80, ymax=120),
        }
        for name, box in cases.items():
            with self.subTest(name):
                *_, edge = self.localizer.find_center(pd.Series(box), shape)
                self.assertTrue(edge)

    def test_find_center_shift_formula(self) -> None:
        '''The center of a truncated feature may lie outside the image.

        A box clipped at the left edge still has the extent of the feature,
        so the center is ``xmax - ext/2``, here outside the field of view.
        '''
        p = pd.Series(dict(xmin=-10, xmax=10, ymin=30, ymax=70))
        x_p, y_p, _ = self.localizer.find_center(p, (100, 200))
        self.assertEqual((x_p, y_p), (-10., 50.))

    def test_localize_single_image(self) -> None:
        self.localizer.frames = [detections((40, 30, 60, 70, 0.9))]
        image = np.full((100, 200), 150, dtype=np.uint8)
        result = self.localizer(image)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns),
                         ['label', 'confidence', 'x_p', 'y_p', 'bbox',
                          'edge'])
        row = result.iloc[0]
        self.assertEqual(row.label, 'hologram')
        self.assertEqual((row.x_p, row.y_p), (50., 50.))
        self.assertEqual(row.bbox, ((40, 30), 20, 40))
        self.assertFalse(row.edge)

    def test_localize_list_and_threshold(self) -> None:
        self.localizer.frames = [
            detections((40, 30, 60, 70, 0.9), (10, 10, 30, 30, 0.2)),
            detections()]
        image = np.full((100, 200), 150, dtype=np.uint8)
        results = self.localizer([image, image])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(len(results[1]), 0)

    def test_dark_images_are_scaled(self) -> None:
        '''Normalized holograms with maximum below 100 are times 100.'''
        self.localizer.frames = [detections()]
        dark = np.full((50, 80), 1.0)
        self.localizer(dark)
        images, size = self.localizer.calls[0]
        self.assertEqual(images[0].max(), 100.)
        self.assertEqual(size, 80)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
