'''Sphinx configuration for CATCH.'''

import importlib.util
import sys
import tomllib
from pathlib import Path

# The repository root is the Python package CATCH, whatever the checkout
# directory is called (Read the Docs names it after the version), so load it
# under that name rather than relying on an installation.
root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    'CATCH', root / '__init__.py', submodule_search_locations=[str(root)])
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules['CATCH'] = package

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
autodoc_mock_imports = ['torch', 'cv2', 'pylorenzmie']

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
    (r'py:.*', r'(torch|cv2|pylorenzmie)(\..*)?'),
]

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/laltman2/CATCH',
}
