import re
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
