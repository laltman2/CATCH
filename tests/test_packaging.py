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

    def test_packages_have_directories(self) -> None:
        mapping = self.config['tool']['setuptools']['package-dir']
        for name, directory in mapping.items():
            self.assertTrue((ROOT / directory).is_dir(), name)


class TestLightImport(unittest.TestCase):
    '''Importing CATCH needs neither YOLOv5 nor the training code.'''

    def test_import_is_light(self) -> None:
        code = ('import sys, CATCH; '
                'bad = [m for m in ("yolov5", "pylorenzmie", '
                '"CATCH.training", "CATCH.utilities", "CATCH.Localizer") '
                'if m in sys.modules]; '
                'print(bad)')
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=ROOT.parent, capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
