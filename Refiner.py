'''Refiner: iterative correction of an estimate of a particle's properties

The Estimator is off by a few percent in ``z_p`` and ``a_p`` and by ten
percent in the contrast of the index.  For large spheres that scatter
strongly that is further than the basin of a fit is wide (0.1-0.5%), and the
fit does not lock on.  The Refiner sees the hologram and the hologram that
the model gives for the current estimate, and predicts the correction of the
estimate; applied a few times it brings the estimate into the basin.

The network works on the same crop as the Estimator (twice the extent of the
feature, scaled to 201 x 201) and on the same scale.  Its inputs are the
data, the model rendered at the estimate on the same grid, and a mask of the
pixels that are inside the image (a particle near an edge has a crop that is
padded or shifted).  Rendering the model needs ``pylorenzmie``, which is
imported when a ``Renderer`` is made, so that the network can be used (and
trained on prepared crops) without it.

A state is five numbers, ``(z_p, a_p, c, x, y)``: ``c`` is ``n_p - n_m``
and ``x`` and ``y`` are the offsets of the particle from the center of the
crop, in pixels of the image.  The correction is predicted in percent for
the first three and in units of 3 pixels for the others.
'''
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from CATCH.evaluation.cases import INSTRUMENT

#: side of the image that the network takes
N = 201
#: size of a pixel [um] and refractive index of the medium
PIXEL = INSTRUMENT['magnification']
N_M = INSTRUMENT['n_m']
#: the sphere must be this many radii from the plane of the image
CLEARANCE = 1.2
#: units of the outputs: percent, and 3 pixels
UNITS = np.array([100., 100., 100., 1. / 3., 1. / 3.])


class RefinerNet(nn.Module):
    '''Predicts the correction of an estimate from the data and the model

    ``forward(data, guess, mask, scale)`` takes tensors of shape
    ``(B, 201, 201)`` (the hologram, the model at the estimate, the mask)
    and ``(B,)`` (the side of the crop over 201), and returns ``(B, 5)``:
    the corrections of ``ln z_p``, ``ln a_p`` and ``ln c`` in percent and of
    ``x`` and ``y`` in units of 3 pixels, such that
    ``true = estimate * exp(correction / 100)`` for the first three.
    '''

    def __init__(self) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Module:
            return nn.Sequential(nn.Conv2d(i, o, 5, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.conv = nn.Sequential(block(4, 32), block(32, 32),
                                  block(32, 48), block(48, 48))
        self.dense = nn.Sequential(nn.Linear(48 * 12 * 12 + 2, 256),
                                   nn.ReLU(), nn.Linear(256, 5))

    def forward(self, data: torch.Tensor, guess: torch.Tensor,
                mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # The data have an amplitude of up to 2 near the center and the
        # difference that carries the signal is 0.05: compress the first
        # two and boost the third.
        x = torch.stack([torch.asinh(data - 1.) * mask,
                         torch.asinh(guess - 1.) * mask,
                         10. * (data - guess) * mask, mask], 1)
        f = self.conv(x).flatten(1)
        s = torch.stack([scale, torch.log(scale)], 1)
        return self.dense(torch.cat([f, s], 1))


def crop(image: np.ndarray, x: float, y: float, side: int,
         pad: bool = True) -> tuple[np.ndarray, np.ndarray, float, float]:
    '''A square crop with a mask of the pixels that are in the image

    Parameters
    ----------
    image : numpy.ndarray
        The normalized hologram.
    x, y : float
        The center of the crop [pixels]: where the Localizer found the
        particle.
    side : int
        Side of the crop [pixels].
    pad : bool
        ``True``: centered on ``(x, y)``, background where the crop leaves
        the image.  ``False``: moved inside the image (the side is at most
        that of the image), as ``CATCHapp.crop`` does.  Default: True

    Returns
    -------
    data, mask : numpy.ndarray
        The crop (1 where there is no image) and 1 where it has pixels of
        the image, 0 elsewhere.
    ox, oy : float
        Offset of ``(x, y)`` from the center of the crop [pixels].
    '''
    rows, columns = image.shape[:2]
    if pad:
        x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
    else:
        side = int(min(side, rows, columns))
        x0 = int(np.clip(round(x - side / 2), 0, columns - side))
        y0 = int(np.clip(round(y - side / 2), 0, rows - side))
    data = np.ones((side, side))
    mask = np.zeros((side, side))
    top, left = max(y0, 0), max(x0, 0)
    bottom, right = min(y0 + side, rows), min(x0 + side, columns)
    if bottom > top and right > left:
        data[top - y0:bottom - y0, left - x0:right - x0] = \
            image[top:bottom, left:right]
        mask[top - y0:bottom - y0, left - x0:right - x0] = 1.
    center = (side - 1) / 2.
    return data, mask, x - x0 - center, y - y0 - center


def resize(array: np.ndarray) -> torch.Tensor:
    '''A crop on the 201 x 201 grid of the network (antialiased)'''
    tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))
    return F.interpolate(tensor[None, None], size=(N, N), mode='bilinear',
                         antialias=True)[0, 0]


def resize_mask(mask: np.ndarray) -> torch.Tensor:
    '''A mask on the grid of the network: values from 0 to 1'''
    return resize(mask).clamp(0., 1.)


class Renderer:
    '''The hologram of one sphere on the grid of a crop

    The grid has ``201 x 201`` points, spaced by ``side / 201`` pixels, so
    that a crop of any side is the same size to the network, as the
    Estimator's crop is.  Needs ``pylorenzmie``.
    '''

    def __init__(self, instrument: dict[str, float] | None = None) -> None:
        from pylorenzmie.theory import LorenzMie
        self.model = LorenzMie()
        self.model.instrument.properties = dict(instrument or INSTRUMENT)
        grid = np.asarray(self.model.meshgrid((N, N)), dtype=float)
        self.grid = grid - grid.mean(axis=1, keepdims=True)

    def __call__(self, side: float,
                 state: Sequence[float] | np.ndarray) -> np.ndarray:
        '''The hologram of ``state`` on the grid of a crop of ``side``

        The axial position is moved out of the sphere if it is inside it.
        '''
        from pylorenzmie.theory import Sphere
        z, a, c, x, y = state
        z = max(z, CLEARANCE * a / PIXEL)
        self.model.coordinates = self.grid * (side / N)
        self.model.particle = Sphere(x_p=x, y_p=y, z_p=z, a_p=a,
                                     n_p=N_M + c)
        return np.asarray(self.model.hologram()).reshape(N, N)


def apply(state: np.ndarray, correction: np.ndarray) -> np.ndarray:
    '''The state after a correction of the network (new array)'''
    out = np.array(state, dtype=float)
    out[:3] *= np.exp(np.asarray(correction[:3], dtype=float) / 100.)
    out[3:] += np.asarray(correction[3:], dtype=float) / UNITS[3:]
    return out


def refine_states(net: nn.Module, data: torch.Tensor, mask: torch.Tensor,
                  sides: np.ndarray, states: np.ndarray, renderer: Renderer,
                  steps: int = 3,
                  device: torch.device | str = 'cpu') -> np.ndarray:
    '''Applies the network ``steps`` times

    Parameters
    ----------
    net : torch.nn.Module
        A ``RefinerNet`` (in eval mode).
    data, mask : torch.Tensor
        Shape ``(B, 201, 201)``.
    sides : numpy.ndarray
        Sides of the crops [pixels], shape ``(B,)``.
    states : numpy.ndarray
        Shape ``(B, 5)``: the estimates.
    renderer : Renderer
    steps : int
        Default: 3
    device : torch.device
        Where the network is.  Default: cpu

    Returns
    -------
    states : numpy.ndarray
        The refined states.  A state whose model is not finite is left as
        it is from there on.
    '''
    states = np.array(states, dtype=float)
    scale = torch.as_tensor(np.asarray(sides, dtype=np.float32) / N)
    done = np.zeros(len(states), dtype=bool)
    for _ in range(steps):
        guesses = np.zeros((len(states), N, N), dtype=np.float32)
        for i, (side, state) in enumerate(zip(sides, states, strict=True)):
            if done[i]:
                continue
            image = renderer(side, state)
            if np.isfinite(image).all():
                guesses[i] = image
            else:
                done[i] = True
        with torch.no_grad():
            out = net(data.to(device), torch.as_tensor(guesses).to(device),
                      mask.to(device), scale.to(device)).cpu().numpy()
        for i in range(len(states)):
            if not done[i] and np.isfinite(out[i]).all():
                states[i] = apply(states[i], out[i])
    return states


class Refiner:
    '''Refines the estimates of the Estimator

    Parameters
    ----------
    weights : str or Path
        A checkpoint of ``training.refiner`` (``examples<N>.pt``).
    device : str
        Default: ``'cpu'``
    steps : int
        Applications of the network to an estimate.  Default: 3

    Notes
    -----
    Needs ``pylorenzmie`` to render the model.  Valid for the instrument of
    ``evaluation.cases.INSTRUMENT`` (100X, 48 nm per pixel) and for spheres
    of the sizes that it was trained on (2-5 um by default).
    '''

    def __init__(self, weights: str | Path, device: str = 'cpu',
                 steps: int = 3, renderer: Renderer | None = None) -> None:
        self.net = RefinerNet()
        saved = torch.load(weights, map_location='cpu')
        self.net.load_state_dict(saved['state_dict'])
        self.net.eval().to(device)
        self.device = torch.device(device)
        self.steps = steps
        self.renderer = renderer or Renderer()

    def __call__(self, crops: Sequence[tuple[np.ndarray, np.ndarray]],
                 states: np.ndarray) -> np.ndarray:
        '''Refines the states of particles in padded crops with masks

        Parameters
        ----------
        crops : sequence of (data, mask)
            From ``crop``.
        states : numpy.ndarray
            Shape ``(len(crops), 5)``: ``(z_p, a_p, n_p - n_m, x, y)``.

        Returns
        -------
        states : numpy.ndarray
        '''
        if not len(crops):
            return np.zeros((0, 5))
        data = torch.stack([resize(d) for d, _ in crops])
        mask = torch.stack([resize_mask(m) for _, m in crops])
        sides = np.array([d.shape[0] for d, _ in crops], dtype=float)
        return refine_states(self.net, data, mask, sides, states,
                             self.renderer, self.steps, self.device)


def refined_pipeline(pipeline: Callable[[Any], Any], refiner: Refiner,
                     minimum_radius: float = 1.5,
                     minimum: int = 31, pad: bool = True,
                     largest_shift: float = 10.) -> Callable[[Any], Any]:
    '''A pipeline whose estimates are refined

    The detections of ``pipeline`` (a function of a ``Scene`` returning a
    table of ``x_p, y_p, z_p, a_p, n_p``) are refined where the estimated
    radius is at least ``minimum_radius`` [um]; the table has the originals
    in columns ``z_p_0`` etc.

    Parameters
    ----------
    pipeline : callable
    refiner : Refiner
    minimum_radius : float
        Smaller spheres are not refined.  Default: 1.5
    minimum : int
        Smallest side of a crop [pixels].  Default: 31
    pad : bool
        Crops padded and centered, in place of shifted inside the image.
        Default: True
    largest_shift : float
        The refined center is kept only within this many pixels of the
        detected one.  Default: 10
    '''
    from CATCH.evaluation.pipelines import extent

    def run(scene: Any) -> Any:
        found = pipeline(scene).copy()
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p'):
            found[key + '_0'] = found[key]
        rows, crops, states, origins = [], [], [], []
        for i in found.index:
            x, y, z, a, n = (float(found.loc[i, k]) for k in
                             ('x_p', 'y_p', 'z_p', 'a_p', 'n_p'))
            if not (np.isfinite([x, y, z, a, n]).all() and z > 0
                    and a >= minimum_radius and n > N_M):
                continue
            side = max(int(2 * extent(z, a, n)), minimum)
            data, mask, ox, oy = crop(scene.image, x, y, side, pad)
            crops.append((data, mask))
            states.append([z, a, n - N_M, ox, oy])
            origins.append((x - ox, y - oy))
            rows.append(i)
        if rows:
            refined = refiner(crops, np.array(states))
            for i, (cx, cy), r, s in zip(rows, origins, refined, states,
                                         strict=True):
                found.loc[i, ['z_p', 'a_p', 'n_p']] = [r[0], r[1],
                                                       N_M + r[2]]
                if np.hypot(r[3] - s[3], r[4] - s[4]) <= largest_shift:
                    found.loc[i, ['x_p', 'y_p']] = [cx + r[3], cy + r[4]]
        return found
    return run
