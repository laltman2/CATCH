import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from CATCH.Estimator import Estimator

CONFIG = {'shape': [201, 201],
          'particle': {'z_p': [50, 600],
                       'a_p': [0.2, 5.0],
                       'n_p': [1.38, 2.5]}}


def make_estimator(**kwargs) -> Estimator:
    '''An Estimator with random weights: no trained model is loaded.'''
    with patch.object(Estimator, '_load_model'), \
         patch.object(Estimator, '_load_config', return_value=CONFIG):
        return Estimator(**kwargs)


class TestEstimator(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.estimator = make_estimator()
        self.crop = np.full((100, 100), 100, dtype=np.uint8)

    def test_configuration(self) -> None:
        self.assertEqual(self.estimator.shape, (201, 201))
        self.assertEqual(self.estimator.model, 'default')
        self.assertEqual(self.estimator.device, torch.device('cpu'))
        self.assertFalse(self.estimator.training)
        self.assertEqual(self.estimator.predict, self.estimator.estimate)

    def test_estimate_one_image(self) -> None:
        result = self.estimator(self.crop)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['z_p', 'a_p', 'n_p'])
        self.assertEqual(len(result), 1)

    def test_estimate_list(self) -> None:
        crops = [self.crop, np.full((50, 50), 80, dtype=np.uint8)]
        result = self.estimator.estimate(crops)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_estimate_uses_the_unnormalized_outputs(self) -> None:
        '''Estimates are the network outputs in physical units.'''
        outputs = torch.tensor([[0., 0.5, 1.]])
        with patch.object(torch.nn.Module, '__call__',
                          return_value=outputs):
            result = self.estimator.estimate(self.crop)
        row = result.iloc[0]
        self.assertAlmostEqual(row.z_p, 50.)
        self.assertAlmostEqual(row.a_p, 2.6, places=5)
        self.assertAlmostEqual(row.n_p, 2.5, places=5)

    def test_scale_is_crop_size_over_network_size(self) -> None:
        seen = {}

        def forward(self_, image, scale):
            seen['scale'] = scale.clone()
            seen['shape'] = tuple(image.shape)
            return torch.zeros(len(image), 3)

        with patch.object(Estimator.__mro__[1], 'forward', forward):
            self.estimator.estimate([self.crop])
        self.assertAlmostEqual(seen['scale'].item(), 100/201, places=6)
        self.assertEqual(seen['shape'], (1, 1, 201, 201))

    def test_load_image_normalizes_to_mean_100(self) -> None:
        image = np.full((40, 40), 1.0)
        tensor = self.estimator._load_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 1, 201, 201))
        # uint8 image of value 100, as a fraction of 255
        self.assertAlmostEqual(tensor.mean().item(), 100/255, places=3)

    def test_load_image_accepts_color(self) -> None:
        image = np.full((40, 40, 3), 100, dtype=np.uint8)
        tensor = self.estimator._load_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 1, 201, 201))

    def test_nonsquare_crop_warns(self) -> None:
        image = np.full((40, 50), 100, dtype=np.uint8)
        with self.assertLogs('CATCH.Estimator', level='WARNING'):
            self.estimator._load_image(image)

    def test_directory_and_model_path(self) -> None:
        directory = self.estimator.directory()
        self.assertTrue((directory / 'Estimator.py').exists())
        path = self.estimator._model_path()
        self.assertEqual(path.parts[-2:], ('cfg_estimator', 'default'))


class TestLoadingFromFiles(unittest.TestCase):
    '''A checkpoint and a configuration can be given directly'''

    def test_weights_and_config(self) -> None:
        torch.manual_seed(2)
        reference = make_estimator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'epoch5.pt'
            torch.save({'state_dict': reference.state_dict(),
                        'optimizer': {}}, path)
            loaded = Estimator(weights=path, config=CONFIG)
        self.assertEqual(loaded.shape, (201, 201))
        self.assertEqual(loaded.weights, path)
        for a, b in zip(loaded.state_dict().values(),
                        reference.state_dict().values(), strict=True):
            self.assertTrue(torch.equal(a, b))

    def test_the_configuration_is_copied(self) -> None:
        with patch.object(Estimator, '_load_model'):
            estimator = Estimator(config=CONFIG)
        estimator.config['shape'][0] = 5
        self.assertEqual(CONFIG['shape'], [201, 201])


class TestLoading(unittest.TestCase):
    '''Weights and configuration are read from DIRECTORY/cfg_estimator'''

    def test_load_weights_and_configuration(self) -> None:
        torch.manual_seed(1)
        reference = make_estimator()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'cfg_estimator' / 'mine'
            model.mkdir(parents=True)
            torch.save({'state_dict': reference.state_dict()},
                       model / 'weights.pt')
            (model / 'configuration.json').write_text(json.dumps(CONFIG))
            with patch.object(Estimator, 'directory',
                              return_value=Path(tmp)):
                loaded = Estimator(model='mine')
        self.assertEqual(loaded.model, 'mine')
        self.assertEqual(loaded.shape, (201, 201))
        for p in loaded.parameters():
            self.assertFalse(p.requires_grad)
        for a, b in zip(loaded.state_dict().values(),
                        reference.state_dict().values(), strict=True):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
