import importlib
import sys

from CATCH.version import __version__
from CATCH.Estimator import Estimator


__all__ = ['Localizer', 'Estimator', 'CATCHapp']

_lazy = ('Localizer', 'CATCHapp')


def __getattr__(name: str):
    # Localizer (and so CATCHapp) needs the optional yolov5 package, so they
    # are imported on first use rather than with the package.  Importing a
    # submodule binds its name in this namespace to the module, which would
    # hide the class of the same name, so bind the classes afterward.
    if name not in _lazy:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module(f'{__name__}.{name}')
    for lazy in _lazy:
        loaded = sys.modules.get(f'{__name__}.{lazy}')
        if loaded is not None:
            globals()[lazy] = getattr(loaded, lazy)
    return getattr(module, name)
