from collections.abc import Sequence
from typing import Any


class ParamScale(object):
    '''Reversibly scales parameters into the range [0, 1]

    Parameters
    ----------
    config : dict
        ``config['particle']`` maps the names ``'z_p'``, ``'a_p'`` and
        ``'n_p'`` to their ``[minimum, maximum]`` ranges.
    '''

    def __init__(self, config: dict) -> None:
        self.z_range = config['particle']['z_p']
        self.a_range = config['particle']['a_p']
        self.n_range = config['particle']['n_p']
        self.ranges = [self.z_range, self.a_range, self.n_range]

    def normalize(self, params: Sequence[Any]) -> list[Any]:
        '''Scales ``(z_p, a_p, n_p)`` into [0, 1]'''
        scaled_params = []
        for param, minmax in zip(params, self.ranges, strict=True):
            pmin = minmax[0]
            pmax = minmax[1]
            prange = pmax - pmin
            scaled = (param - pmin)/prange
            scaled_params.append(scaled)
        return scaled_params

    def unnormalize(self, params: Sequence[Any]) -> list[Any]:
        '''Scales ``(z_p, a_p, n_p)`` from [0, 1] to the configured ranges'''
        unscaled_params = []
        for param, minmax in zip(params, self.ranges, strict=True):
            pmin = minmax[0]
            pmax = minmax[1]
            prange = pmax - pmin
            unscaled = (param * prange) + pmin
            unscaled_params.append(unscaled)
        return unscaled_params
