import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / 'src' / 'CATCH'


class TestPackaging(unittest.TestCase):

    def setUp(self) -> None:
        with open(ROOT / 'pyproject.toml', 'rb') as f:
            self.config = tomllib.load(f)

    def test_version_is_written_once(self) -> None:
        '''The version lives in pyproject.toml only.'''
        self.assertNotIn('dynamic', self.config['project'])
        self.assertRegex(self.config['project']['version'],
                         r'^\d+\.\d+\.\d+$')
        self.assertFalse((ROOT / 'version.py').exists())

    def test_the_distribution_is_catch_python(self) -> None:
        '''PyPI: ``catch-python`` installs the package ``CATCH``.'''
        self.assertEqual(self.config['project']['name'], 'catch-python')
        found = self.config['tool']['setuptools']['packages']['find']
        self.assertEqual(found['include'], ['CATCH*'])

    def test_the_installed_version_is_the_one_in_pyproject(self) -> None:
        import CATCH
        if CATCH.__version__ == 'unknown':
            self.skipTest('CATCH is not installed')
        self.assertEqual(CATCH.__version__,
                         self.config['project']['version'])

    def test_a_tag_publishes_after_the_tests_passed(self) -> None:
        '''publish.yml waits for the workflow that test.yml defines.'''
        workflows = ROOT / '.github' / 'workflows'
        tests = (workflows / 'test.yml').read_text()
        publish = (workflows / 'publish.yml').read_text()
        name = re.search(r'^name:\s*(.+)$', tests, re.M)
        assert name is not None
        self.assertIn(f'--workflow {name.group(1).strip()} ', publish)
        self.assertIn("- 'v*'", publish)
        self.assertIn('id-token: write', publish)

    def test_the_package_is_marked_as_typed(self) -> None:
        '''PEP 561: the marker is there and is shipped with the package.'''
        self.assertTrue((PACKAGE / 'py.typed').is_file())
        data = self.config['tool']['setuptools']['package-data']['CATCH']
        self.assertIn('py.typed', data)

    def test_the_citation_is_of_this_version(self) -> None:
        text = (ROOT / 'CITATION.cff').read_text()
        cited = re.search(r'^version:\s*"?([\w.]+)"?\s*$', text, re.M)
        assert cited is not None
        self.assertEqual(cited.group(1), self.config['project']['version'])
        self.assertNotIn('YOLO', text)         # the 2.x localizer is gone

    def test_the_package_says_that_it_is_typed(self) -> None:
        classifiers = self.config['project']['classifiers']
        self.assertIn('Typing :: Typed', classifiers)

    def test_the_model_cards_are_shipped(self) -> None:
        data = self.config['tool']['setuptools']['package-data']['CATCH']
        self.assertIn('models/*/*/MODEL_CARD.md', data)
        for kind in ('localizer', 'estimator'):
            self.assertTrue(
                (PACKAGE / 'models' / kind / 'default' / 'MODEL_CARD.md')
                .is_file())

    def test_the_stored_truth_of_the_sets_is_shipped(self) -> None:
        '''sets.verify needs the files, wherever the harness runs.'''
        data = self.config['tool']['setuptools']['package-data']
        shipped = data['CATCH.customization.evaluation']
        self.assertEqual(shipped, ['sets/*.csv'])
        directory = PACKAGE / 'customization' / 'evaluation' / 'sets'
        self.assertEqual(sorted(p.name for p in directory.glob('*.csv')),
                         ['dev_truth.csv', 'test_truth.csv'])

    def test_packages_are_found_in_src(self) -> None:
        find = self.config['tool']['setuptools']['packages']['find']
        self.assertEqual(find['where'], ['src'])
        self.assertTrue((ROOT / 'src' / 'CATCH' / '__init__.py').is_file())


class TestMinimumVersions(unittest.TestCase):
    '''The CI job of the lowest versions installs the lower bounds.'''

    @staticmethod
    def minor(version: str) -> str:
        parts = version.split('.')[:2]
        return '.'.join(parts + ['0'] * (2 - len(parts)))

    def test_the_pins_are_the_lower_bounds(self) -> None:
        workflow = ROOT / '.github' / 'workflows' / 'test.yml'
        if not workflow.is_file():
            self.skipTest('no .github in this tree')
        text = workflow.read_text()
        with open(ROOT / 'pyproject.toml', 'rb') as f:
            dependencies = tomllib.load(f)['project']['dependencies']
        for requirement in dependencies:
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)>=([\d.]+)', requirement)
            self.assertIsNotNone(match, f'{requirement}: no lower bound')
            assert match is not None
            name, bound = match.groups()
            pin = re.search(rf'"{name}==([\d.]+)\.\*"', text)
            self.assertIsNotNone(pin, f'{name}: not pinned in test.yml')
            assert pin is not None
            self.assertEqual(self.minor(pin.group(1)), self.minor(bound),
                             f'{name}: the pin is not the lower bound')


class TestMedia(unittest.TestCase):

    SUFFIXES = ('.png', '.jpg', '.jpeg', '.gif', '.avi', '.mp4')
    SKIP = ('.git', 'devel', '__pycache__', '.ipynb_checkpoints')

    def test_media_files_are_under_docs(self) -> None:
        stray = []
        for path in ROOT.rglob('*'):
            parts = path.relative_to(ROOT).parts
            if any(part in self.SKIP for part in parts) or parts[0] == 'docs':
                continue
            if path.suffix.lower() in self.SUFFIXES:
                stray.append(str(path.relative_to(ROOT)))
        self.assertEqual(stray, [])

    def test_example_images_exist(self) -> None:
        for name in ('test_image_crop.png', 'test_image_large.png'):
            self.assertTrue((ROOT / 'docs' / 'images' / name).is_file())


class TestLightImport(unittest.TestCase):
    '''Importing CATCH loads neither pylorenzmie nor the training code.'''

    def test_import_is_light(self) -> None:
        code = ('import sys, CATCH; '
                'bad = [m for m in ("pylorenzmie", '
                '"CATCH.customization.training", '
                '"CATCH.customization.evaluation", "CATCH.lib.fitter") '
                'if m in sys.modules]; '
                'print(bad)')
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=tempfile.gettempdir(), capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')


class TestModules(unittest.TestCase):
    '''instruments and selection are in lib, and only there'''

    def test_they_are_in_lib(self) -> None:
        from CATCH.lib import instruments, selection
        self.assertTrue(hasattr(instruments, 'REFERENCE'))
        self.assertTrue(hasattr(selection, 'select'))

    def test_the_old_aliases_are_gone(self) -> None:
        import importlib

        import CATCH
        self.assertFalse(hasattr(CATCH, 'instruments'))
        self.assertFalse(hasattr(CATCH, 'selection'))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('CATCH.instruments')
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('CATCH.selection')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
