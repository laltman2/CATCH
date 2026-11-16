import importlib
import sys
import types
from typing import TYPE_CHECKING

from CATCH.version import __version__
from CATCH.Estimator import Estimator

if TYPE_CHECKING:  # the classes that load on first use, for type checkers
    from CATCH.CATCHapp import CATCHapp
    from CATCH.Localizer import Localizer


# Localizer and CATCHapp need the optional yolov5 package, so they are not in
# __all__: ``from CATCH import *`` must work without it.
__all__ = ['Estimator']


def _lazy(name: str) -> property:
    # Importing the submodule CATCH.<name> binds the module to the name
    # ``name`` in this namespace, hiding the class of the same name.  A
    # property is a data descriptor, so it wins over that binding (which it
    # ignores) and the name always gives the class.
    def get(self: types.ModuleType) -> type:
        module = importlib.import_module(f'{self.__name__}.{name}')
        return getattr(module, name)

    def ignore(self: types.ModuleType, value: object) -> None:
        pass

    return property(get, ignore)


class _Package(types.ModuleType):
    Localizer = _lazy('Localizer')
    CATCHapp = _lazy('CATCHapp')


sys.modules[__name__].__class__ = _Package
