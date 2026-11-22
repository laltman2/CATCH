'''Sphinx configuration for CATCH.'''

import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'src'))

# -- Project information -----------------------------------------------------

project = 'CATCH'
author = 'David G. Grier'
copyright = '2026, David G. Grier'
# pyproject.toml is the one place where the version is written; read it
# directly, because the documentation is built without installing CATCH.
release = tomllib.loads((root / 'pyproject.toml').read_text())[
    'project']['version']

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

# The documentation is built without PyTorch, OpenCV and pylorenzmie
autodoc_mock_imports = ['torch', 'pylorenzmie']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

exclude_patterns = ['_build']

# Check every cross-reference; the names of the mocked libraries are left as
# plain text.
nitpicky = True
nitpick_ignore_regex = [
    (r'py:.*', r'(torch|pylorenzmie)(\..*)?'),
]

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/laltman2/CATCH',
}
