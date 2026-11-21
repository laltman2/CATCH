'''CATCH: Characterizing And Tracking Colloids Holographically'''
import sys

from CATCH.version import __version__
from CATCH.CATCHapp import CATCHapp
from CATCH.Estimator import Estimator
from CATCH.Localizer import Localizer
from CATCH.lib import instruments, selection

# For now these two stay importable from the top, as CATCH.instruments and
# CATCH.selection; their home is CATCH.lib.
sys.modules[f'{__name__}.instruments'] = instruments
sys.modules[f'{__name__}.selection'] = selection

__all__ = ['CATCHapp', 'Estimator', 'Localizer', '__version__']
