import re
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPackaging(unittest.TestCase):

    def setUp(self) -> None:
        with open(ROOT / 'pyproject.toml', 'rb') as f:
            self.config = tomllib.load(f)

    def test_version_is_written_once(self) -> None:
        '''The version lives in version.py only; pyproject reads it.'''
        self.assertIn('version', self.config['project']['dynamic'])
        self.assertNotIn('version', self.config['project'])
        text = (ROOT / 'version.py').read_text()
        self.assertIsNotNone(re.search(r"__version__ = '\d+\.\d+\.\d+'",
                                       text))

    def test_the_package_is_marked_as_typed(self) -> None:
        '''PEP 561: the marker is there and is shipped with the package.'''
        self.assertTrue((ROOT / 'py.typed').is_file())
        data = self.config['tool']['setuptools']['package-data']['CATCH']
        self.assertIn('py.typed', data)

    def test_packages_have_directories(self) -> None:
        mapping = self.config['tool']['setuptools']['package-dir']
        for name, directory in mapping.items():
            self.assertTrue((ROOT / directory).is_dir(), name)


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
                'bad = [m for m in ("pylorenzmie", "CATCH.training", '
                '"CATCH.evaluation", "CATCH.lib.fitter") '
                'if m in sys.modules]; '
                'print(bad)')
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=ROOT.parent, capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')


class TestReexports(unittest.TestCase):
    '''instruments and selection stay importable from the top, for now'''

    def test_aliases(self) -> None:
        import importlib

        import CATCH
        from CATCH.lib import instruments, selection

        # by name, because the aliases exist only at run time
        self.assertIs(CATCH.instruments, instruments)
        self.assertIs(CATCH.selection, selection)
        self.assertIs(importlib.import_module('CATCH.instruments'),
                      instruments)
        self.assertIs(importlib.import_module('CATCH.selection'), selection)
        namespace = {}
        exec('from CATCH.instruments import REFERENCE\n'
             'from CATCH.selection import select', namespace)
        self.assertIs(namespace['REFERENCE'], instruments.REFERENCE)
        self.assertIs(namespace['select'], selection.select)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
