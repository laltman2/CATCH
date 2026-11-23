'''The command line of the scene harness'''
import contextlib
import io
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

if find_spec('pylorenzmie') is None:  # pragma: no cover
    raise unittest.SkipTest('pylorenzmie is not installed')

import numpy as np
import pandas as pd

from CATCH.customization.evaluation import scene_run
from CATCH.customization.evaluation.scenes import CAMERA_LEVEL

SMALL = ['--nscenes', '2', '--nparticles', '1', '1', '--shape', '200', '240',
         '--nprocs', '1']


def parse(*argv):
    return scene_run._parser().parse_args(list(argv))


class TestParser(unittest.TestCase):

    def test_defaults(self) -> None:
        args = parse()
        self.assertEqual(args.pipeline, 'oracle')
        self.assertEqual(args.estimator, 'catch')
        self.assertIsNone(args.scene_set)
        self.assertIsNone(args.camera)
        self.assertIsNone(args.instrument)
        self.assertFalse(args.mask_saturated)
        self.assertEqual(args.starts, 1)

    def test_options(self) -> None:
        args = parse('--set', 'test', '--camera', '--mask-saturated',
                     '--instrument', '0.532', '0.1', '1.33',
                     '--pipeline', 'catch', '--lmodel', 'a', '--model', 'b')
        self.assertEqual(args.scene_set, 'test')
        self.assertEqual(args.camera, CAMERA_LEVEL)      # the bare option
        self.assertTrue(args.mask_saturated)
        self.assertEqual(args.instrument, [0.532, 0.1, 1.33])
        self.assertEqual((args.lmodel, args.model), ('a', 'b'))
        self.assertEqual(parse('--camera', '1.6').camera, 1.6)

    def test_bad_values_are_refused(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            parse('--pipeline', 'nonsense')


class TestScenesOf(unittest.TestCase):

    def test_a_set(self) -> None:
        scenes = scene_run._scenes_of(parse('--set', 'dev', '--nscenes', '2'),
                                      None)
        self.assertEqual(len(scenes), 2)

    def test_the_frozen_set_says_so(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            scenes = scene_run._scenes_of(
                parse('--set', 'test', '--nscenes', '1'), None)
        self.assertEqual(len(scenes), 1)
        self.assertIn('frozen', err.getvalue())

    def test_scenes_from_the_options(self) -> None:
        args = parse(*SMALL, '--seed', '3', '--noise', '0.02')
        scenes = scene_run._scenes_of(args, None)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0].image.shape, (200, 240))
        self.assertEqual(scenes[0].noise, 0.02)
        self.assertEqual(len(scenes[0].truth), 1)

    def test_an_instrument(self) -> None:
        other = dict(wavelength=0.532, magnification=0.1, n_m=1.33)
        scenes = scene_run._scenes_of(parse(*SMALL), other)
        self.assertEqual(scenes[0].instrument, other)

    def test_a_camera_clips_the_scenes(self) -> None:
        scenes = scene_run._scenes_of(parse(*SMALL, '--camera', '1.2'), None)
        for scene in scenes:
            self.assertEqual(scene.saturation, 1.2)
            self.assertLessEqual(float(scene.image.max()), 1.2)


class TestPipelineOf(unittest.TestCase):

    @staticmethod
    def make(*argv, instrument=None):
        parser = scene_run._parser()
        return scene_run._pipeline_of(parser.parse_args(list(argv)), parser,
                                      instrument)

    def test_the_truth(self) -> None:
        scene = scene_run._scenes_of(parse(*SMALL), None)[0]
        found = self.make('--pipeline', 'truth')(scene)
        self.assertEqual(len(found), len(scene.truth))

    def test_recorded_needs_detections(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            self.make('--pipeline', 'recorded')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'found.csv'
            pd.DataFrame(dict(scene=[0], x_p=[1.], y_p=[2.], z_p=[3.],
                              a_p=[1.], n_p=[1.5])).to_csv(path)
            scene = scene_run._scenes_of(parse(*SMALL), None)[0]
            found = self.make('--pipeline', 'recorded', '--detections',
                              str(path))(scene)
        self.assertEqual(len(found), 1)

    def test_the_oracle_has_two_kinds_of_estimator(self) -> None:
        with patch.object(scene_run, 'catch_estimator',
                          return_value='network') as network, \
                patch.object(scene_run, 'estimator_on_truth',
                             return_value='pipeline') as on_truth:
            self.make('--pipeline', 'oracle', '--model', 'mine',
                      '--mask-saturated')
        network.assert_called_once_with('mine')
        on_truth.assert_called_once_with('network', True)
        with patch.object(scene_run, 'pylorenzmie_estimator',
                          return_value='crops') as references, \
                patch.object(scene_run, 'crops_on_truth',
                             return_value='pipeline') as on_crops:
            self.make('--pipeline', 'oracle', '--estimator', 'analytic',
                      '--pad')
        references.assert_called_once_with('analytic')
        on_crops.assert_called_once_with('crops', pad=True)

    def test_catch_has_the_instrument(self) -> None:
        made = []

        class App:
            def __init__(self, **kwargs):
                made.append(kwargs)

        other = dict(wavelength=0.532, magnification=0.1, n_m=1.33)
        with patch('CATCH.CATCHapp', App), \
                patch.object(scene_run, 'catch_pipeline',
                             return_value='pipeline'):
            self.make('--pipeline', 'catch', '--lmodel', 'l', '--model', 'e',
                      instrument=other)
        self.assertEqual(made, [dict(lmodel='l', emodel='e',
                                     instrument=other)])


class TestMain(unittest.TestCase):

    @staticmethod
    def run_main(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scene_run.main(list(argv))
        return out.getvalue()

    def test_a_run_with_the_truth(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            text = self.run_main('--pipeline', 'truth', *SMALL,
                                 '--output', f'{d}/run')
            particles = pd.read_csv(f'{d}/run_particles.csv')
            scenes = pd.read_csv(f'{d}/run_scenes.csv')
        self.assertIn('lock_on', text)
        self.assertEqual(len(particles), 2)
        self.assertEqual(len(scenes), 2)
        self.assertIn('saturated', particles.columns)

    def test_images_are_written_and_the_run_stops(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            text = self.run_main(*SMALL, '--write-images', d)
            self.assertEqual(len(list(Path(d).glob('scene_*.png'))), 2)
        self.assertIn('wrote 2 images', text)

    def test_the_references_are_for_the_reference_instrument(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            self.run_main('--pipeline', 'oracle', '--estimator', 'analytic',
                          '--instrument', '0.532', '0.1', '1.33', *SMALL)

    def test_the_options_reach_the_evaluation(self) -> None:
        seen = {}

        def fake(pipeline, scenes, *args):
            seen['args'] = args
            seen['scenes'] = scenes
            return scene_run.SceneResults(
                scene_run.score(pd.DataFrame(dict(
                    scene=[0], found=[False], inside=[True], a_p=[1.],
                    z_p=[300.], n_p=[1.5], edge=[50.], neighbor=[np.inf],
                    nparticles=[1]))),
                pd.DataFrame(dict(scene=[0], nparticles=[1], ndetected=[0],
                                  tp=[0], fp=[0], fn=[1], seconds=[0.])))

        with patch.object(scene_run, 'evaluate_scenes', fake):
            self.run_main('--pipeline', 'truth', *SMALL, '--grid',
                          '--camera', '--mask-saturated', '--gate', '9')
        gate, nprocs, chunk, starts = seen['args'][:4]
        self.assertEqual((gate, nprocs, chunk), (9., 1, 50))
        self.assertEqual(starts, scene_run.GRID)
        self.assertTrue(seen['args'][-1])                 # mask_saturated
        self.assertEqual(seen['scenes'][0].saturation, CAMERA_LEVEL)
        self.assertEqual(np.shape(seen['scenes'][0].image), (200, 240))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
