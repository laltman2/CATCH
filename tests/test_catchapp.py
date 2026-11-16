import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from CATCH.CATCHapp import CATCHapp


def feature(x_p, y_p, w, h, edge=False) -> dict:
    return {'label': 'hologram', 'confidence': 0.9, 'x_p': x_p, 'y_p': y_p,
            'bbox': ((0, 0), w, h), 'edge': edge}


def crop_of(image, x_p, y_p, w, h, edge=False) -> np.ndarray:
    features = pd.DataFrame([feature(x_p, y_p, w, h, edge)])
    (crop,) = CATCHapp.crop(image, features)
    return crop


class TestCrop(unittest.TestCase):

    def setUp(self) -> None:
        # the value of each pixel is its own flat index
        self.image = np.arange(100*200).reshape(100, 200)

    def test_even_size(self) -> None:
        crop = crop_of(self.image, 100, 50, 20, 20)
        self.assertEqual(crop.shape, (20, 20))
        self.assertEqual(crop[0, 0], self.image[40, 90])

    def test_odd_size_and_square_from_larger_side(self) -> None:
        crop = crop_of(self.image, 100, 50, 11, 21)
        self.assertEqual(crop.shape, (21, 21))

    def test_edge_crops_are_shifted_inward(self) -> None:
        # (x_p, y_p) of a 20 x 20 box near each edge of a 100 x 200 image,
        # and the (row, column) of the image where its crop must begin
        cases = {
            'left': (3, 50, 40, 0),
            'top': (100, 3, 0, 90),
            'right': (197, 50, 40, 180),
            'bottom': (100, 97, 80, 90),
        }
        for name, (x_p, y_p, row, col) in cases.items():
            with self.subTest(name):
                crop = crop_of(self.image, x_p, y_p, 20, 20, edge=True)
                self.assertEqual(crop.shape, (20, 20))
                self.assertEqual(crop[0, 0], self.image[row, col])

    def test_one_crop_per_feature(self) -> None:
        features = pd.DataFrame([feature(50, 50, 20, 20),
                                 feature(120, 60, 30, 30)])
        crops = CATCHapp.crop(self.image, features)
        self.assertEqual([c.shape for c in crops], [(20, 20), (30, 30)])


class TestAnalyze(unittest.TestCase):

    def setUp(self) -> None:
        patches = [patch('CATCH.CATCHapp.Localizer'),
                   patch('CATCH.CATCHapp.Estimator')]
        self.localizer_class, self.estimator_class = (
            p.start() for p in patches)
        for p in patches:
            self.addCleanup(p.stop)
        self.app = CATCHapp(lmodel='l', emodel='e')
        self.image = np.full((100, 200), 100, dtype=np.uint8)

    def test_models_are_passed_on(self) -> None:
        self.localizer_class.assert_called_once_with(model='l')
        self.estimator_class.assert_called_once_with(model='e')

    def test_analyze_combines_features_and_estimates(self) -> None:
        self.app.localizer = MagicMock(return_value=[
            pd.DataFrame([feature(50, 50, 20, 20)]),
            pd.DataFrame(),
            pd.DataFrame([feature(60, 40, 20, 20),
                          feature(120, 60, 30, 30)])])
        self.app.estimator = MagicMock(side_effect=lambda crops: pd.DataFrame(
            {'z_p': [100.]*len(crops), 'a_p': [1.]*len(crops),
             'n_p': [1.5]*len(crops)}))
        result = self.app([self.image]*3)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.framenum), [0, 2, 2])
        for column in ('x_p', 'y_p', 'bbox', 'z_p', 'a_p', 'n_p'):
            self.assertIn(column, result.columns)
        sizes = [c.shape for c in self.app.estimator.call_args_list[1].args[0]]
        self.assertEqual(sizes, [(20, 20), (30, 30)])

    def test_single_image_is_wrapped(self) -> None:
        self.app.localizer = MagicMock(
            return_value=[pd.DataFrame([feature(50, 50, 20, 20)])])
        self.app.estimator = MagicMock(return_value=pd.DataFrame(
            {'z_p': [1.], 'a_p': [1.], 'n_p': [1.5]}))
        result = self.app.analyze(self.image)
        self.assertEqual(len(result), 1)
        self.app.localizer.assert_called_once()
        self.assertEqual(len(self.app.localizer.call_args.args[0]), 1)

    def test_nothing_found_gives_empty_frame(self) -> None:
        self.app.localizer = MagicMock(return_value=[pd.DataFrame()])
        result = self.app.analyze(self.image)
        self.assertTrue(result.empty)
        self.app.estimator.assert_not_called()

    def test_none_gives_empty_frame(self) -> None:
        self.assertTrue(self.app.analyze(None).empty)
        self.app.localizer.assert_not_called()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
