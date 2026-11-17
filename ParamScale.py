import math
from collections.abc import Sequence
from typing import Any


def _log(value: Any) -> Any:
    '''Natural logarithm of a number or of a torch tensor'''
    return value.log() if hasattr(value, 'log') else math.log(value)


def _exp(value: Any) -> Any:
    '''Exponential of a number or of a torch tensor'''
    return value.exp() if hasattr(value, 'exp') else math.exp(value)


class ParamScale(object):
    '''Reversibly scales parameters into the range [0, 1]

    Parameters
    ----------
    config : dict
        ``config['particle']`` maps the names ``'z_p'``, ``'a_p'`` and
        ``'n_p'`` to their ``[minimum, maximum]`` ranges.  The optional
        ``config['scaling']`` maps ``'a_p'`` to ``'log'`` (the radius is
        scaled by its logarithm, so that equal relative errors weigh the
        same) and ``'n_p'`` to ``'log_contrast'`` (the logarithm of
        ``n_p - n_m``, with ``n_m`` of ``config['instrument']``).  The
        default is a linear scale for all three.
    '''

    def __init__(self, config: dict) -> None:
        self.z_range = config['particle']['z_p']
        self.a_range = config['particle']['a_p']
        self.n_range = config['particle']['n_p']
        self.ranges = [self.z_range, self.a_range, self.n_range]
        scaling = config.get('scaling', {})
        n_m = config.get('instrument', {}).get('n_m', 0.)
        self._forward: list[Any] = [None, None, None]
        self._inverse: list[Any] = [None, None, None]
        if scaling.get('a_p') == 'log':
            self._forward[1], self._inverse[1] = _log, _exp
        if scaling.get('n_p') == 'log_contrast':
            self._forward[2] = lambda n: _log(n - n_m)
            self._inverse[2] = lambda u: _exp(u) + n_m

    def _limits(self, k: int) -> tuple[Any, Any]:
        low, high = self.ranges[k]
        forward = self._forward[k]
        return (forward(low), forward(high)) if forward else (low, high)

    @staticmethod
    def _check(params: Sequence[Any]) -> None:
        if len(params) != 3:
            raise ValueError('three parameters: z_p, a_p and n_p')

    def normalize(self, params: Sequence[Any]) -> list[Any]:
        '''Scales ``(z_p, a_p, n_p)`` into [0, 1]'''
        self._check(params)
        scaled_params = []
        for k, param in enumerate(params):
            pmin, pmax = self._limits(k)
            forward = self._forward[k]
            value = forward(param) if forward else param
            scaled_params.append((value - pmin)/(pmax - pmin))
        return scaled_params

    def unnormalize(self, params: Sequence[Any]) -> list[Any]:
        '''Scales ``(z_p, a_p, n_p)`` from [0, 1] to the configured ranges'''
        self._check(params)
        unscaled_params = []
        for k, param in enumerate(params):
            pmin, pmax = self._limits(k)
            unscaled = (param * (pmax - pmin)) + pmin
            inverse = self._inverse[k]
            unscaled_params.append(inverse(unscaled) if inverse
                                   else unscaled)
        return unscaled_params
