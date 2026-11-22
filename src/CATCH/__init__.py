'''CATCH: Characterizing And Tracking Colloids Holographically'''
from importlib.metadata import PackageNotFoundError, version

from CATCH.CATCHapp import CATCHapp
from CATCH.Estimator import Estimator
from CATCH.Localizer import Localizer

try:
    __version__ = version('CATCH')
except PackageNotFoundError:  # not installed (e.g. a bare source checkout)
    __version__ = 'unknown'

__all__ = ['CATCHapp', 'Estimator', 'Localizer', '__version__']
