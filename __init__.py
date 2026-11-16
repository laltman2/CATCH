from CATCH.version import __version__
from CATCH.Estimator import Estimator


__all__ = ['Localizer', 'Estimator', 'CATCHapp']


def __getattr__(name: str):
    # Localizer (and so CATCHapp) needs the optional yolov5 package, so they
    # are imported on first use rather than with the package.
    if name == 'Localizer':
        from CATCH.Localizer import Localizer
        return Localizer
    if name == 'CATCHapp':
        from CATCH.CATCHapp import CATCHapp
        return CATCHapp
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
