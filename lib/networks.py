'''The two networks of CATCH

:class:`DetectorNet` finds the centers of the holograms in a whole frame
(the Localizer); :class:`ProfileNet` proposes candidates for the properties
of a particle from the radial profile of its hologram (the Estimator).
Both work in pixels of the camera through a scale factor (see
:mod:`CATCH.lib.instruments`), so that one set of weights serves instruments
whose pixels are 1 to 2.5 times the size of the reference pixel.
'''
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: cells of the heatmap of the detector [pixels]
STRIDE = 4
#: distance from an edge of the frame beyond which the detector ignores it
EDGE_SCALE = 256.
#: number of radii of the profile
NRADII = 640
#: radii of the profile in pixels of the reference (2, 4, ..., 1280)
RADII = 2. * np.arange(1, NRADII + 1)
#: candidates proposed for each particle
NCANDIDATES = 64
#: center of the outputs, the ``log`` of z_p [reference pixels], a_p [um]
#: and the contrast n_p - n_m
CENTER = np.log([800., 1.1, .22])


def detector_inputs(images: torch.Tensor) -> torch.Tensor:
    '''Normalized holograms (B, H, W) with the distances to the edges'''
    b, h, w = images.shape
    d = images.device
    x = torch.minimum(torch.arange(w, device=d, dtype=torch.float32),
                      torch.arange(w - 1, -1, -1, device=d,
                                   dtype=torch.float32))
    y = torch.minimum(torch.arange(h, device=d, dtype=torch.float32),
                      torch.arange(h - 1, -1, -1, device=d,
                                   dtype=torch.float32))
    x = (x / EDGE_SCALE).clamp(max=1.).expand(b, h, w)
    y = (y / EDGE_SCALE).clamp(max=1.)[:, None].expand(b, h, w)
    return torch.stack([3 * (images - 1).clamp(-1, 3), x, y], 1)


class Block(nn.Module):
    '''Residual block of depthwise-separable convolutions (2D)'''

    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation,
                      groups=c, bias=False),
            nn.GroupNorm(8, c), nn.Conv2d(c, c, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.GroupNorm(8, c), nn.Conv2d(c, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


def _down(a: int, b: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(a, b, 3, 2, 1, bias=False),
                         nn.GroupNorm(8, b), nn.GELU())


class DetectorNet(nn.Module):
    '''Heatmap of particle centers with their offsets (CenterNet)

    An encoder to 1/32 of the resolution, with dilated blocks that see the
    whole of a large hologram, and a decoder back to cells of
    :data:`STRIDE` pixels.  For each cell: the logit that a center falls in
    it, and the position of the center inside the cell.
    '''

    def __init__(self, w: tuple[int, ...] = (24, 32, 64, 96, 160, 224)
                 ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 5, 2, 2, bias=False),
                                  nn.GroupNorm(8, w[0]), nn.GELU())
        self.s4 = nn.Sequential(_down(w[0], w[1]), Block(w[1]), Block(w[1]))
        self.s8 = nn.Sequential(_down(w[1], w[2]), Block(w[2]), Block(w[2]))
        self.s16 = nn.Sequential(_down(w[2], w[3]), Block(w[3]),
                                 Block(w[3]), Block(w[3], 2))
        self.s32 = nn.Sequential(_down(w[3], w[4]), Block(w[4]),
                                 Block(w[4], 2), Block(w[4], 4),
                                 Block(w[4], 2), Block(w[4]))
        self.up16 = self._fuse(w[4], w[3])
        self.up8 = self._fuse(w[3], w[2])
        self.up4 = self._fuse(w[2], w[1])
        last = nn.Conv2d(w[1], 1, 1)
        assert last.bias is not None
        nn.init.constant_(last.bias, -4.6)     # few centers at the start
        self.head = nn.Sequential(Block(w[1]), last)
        self.offset = nn.Sequential(Block(w[1]), nn.Conv2d(w[1], 2, 1))

    @staticmethod
    def _fuse(a: int, b: int) -> nn.ModuleDict:
        return nn.ModuleDict(dict(
            top=nn.Conv2d(a, b, 1), side=nn.Conv2d(b, b, 1),
            block=nn.Sequential(Block(b), Block(b))))

    @staticmethod
    def _join(m: nn.ModuleDict, top: torch.Tensor,
              side: torch.Tensor) -> torch.Tensor:
        t = F.interpolate(m['top'](top), size=side.shape[-2:],
                          mode='bilinear', align_corners=False)
        return m['block'](t + m['side'](side))

    def forward(self, images: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        '''Logits (B, h, w) and offsets in [0, 1] (B, 2, h, w)'''
        s4 = self.s4(self.stem(detector_inputs(images)))
        s8 = self.s8(s4)
        s16 = self.s16(s8)
        s32 = self.s32(s16)
        x = self._join(self.up16, s32, s16)
        x = self._join(self.up8, x, s8)
        x = self._join(self.up4, x, s4)
        return self.head(x)[:, 0], torch.sigmoid(self.offset(x))


def peaks(net: DetectorNet, images: torch.Tensor, count: int = 40
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''The strongest local maxima of the heatmap: x, y [pixels], score

    Each of shape (B, count), strongest first.
    '''
    heat, off = net(images)
    p = torch.sigmoid(heat)
    top = F.max_pool2d(p[:, None], 5, 1, 2)[:, 0] == p
    p = p * top
    count = min(count, p[0].numel())
    v, i = p.flatten(1).topk(count, 1)
    w = p.shape[2]
    y, x = torch.div(i, w, rounding_mode='floor'), i % w
    ox = off[:, 0].flatten(1).gather(1, i)
    oy = off[:, 1].flatten(1).gather(1, i)
    return (((x + ox) * STRIDE).cpu().numpy(),
            ((y + oy) * STRIDE).cpu().numpy(), v.cpu().numpy())


def grid_shape(shape: tuple[int, int]) -> tuple[int, int]:
    '''Shape of the heatmap of a frame'''
    return math.ceil(shape[0] / STRIDE), math.ceil(shape[1] / STRIDE)


class Block1d(nn.Module):
    '''Residual block of 1D convolutions'''

    def __init__(self, c: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv1d(c, c, 7, padding=3), nn.GroupNorm(8, c))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


def profile_features(b: torch.Tensor, mask: torch.Tensor,
                     pixel: float = .048) -> torch.Tensor:
    '''Inputs (P, 4, NRADII) of the ProfileNet from profiles (P, NRADII)

    The deviation from the background at three scales (the last grows with
    the radius, like the amplitude of the outer fringes is lost), and the
    mask of the radii that were measured.
    '''
    r = torch.as_tensor(RADII * pixel / 6, device=b.device, dtype=b.dtype)
    d = (b - 1) * mask
    return torch.stack([3 * d, (30 * d).clamp(-3, 3), d * r, mask], 1)


class ProfileNet(nn.Module):
    '''Candidates for (z_p, a_p, n_p - n_m) from a radial profile

    A 1D convolutional network over the :data:`NRADII` radii of the
    profile, then :data:`NCANDIDATES` hypotheses, each the logarithm of the
    three properties (times 100, about :data:`CENTER`) and a logit.  Trained
    with a winner-takes-all loss, so the candidates spread over the
    near-degenerate solutions; the Estimator ranks them against the data.
    '''

    def __init__(self, k: int = NCANDIDATES) -> None:
        super().__init__()
        self.k = k
        layers: list[nn.Module] = [nn.Conv1d(4, 48, 7, stride=2, padding=3),
                                   nn.GELU()]
        c = 48
        for out in (64, 96, 128, 192):
            layers += [Block1d(c, 1), Block1d(c, 2),
                       nn.Conv1d(c, out, 4, stride=2, padding=1), nn.GELU()]
            c = out
        layers += [Block1d(c, 1), Block1d(c, 2)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 20, 768),
                                  nn.GELU(), nn.Linear(768, 4 * k))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.head(self.body(x)).reshape(-1, self.k, 4)
        return 60 * o[..., :3], o[..., 3]

    def candidates(self, x: torch.Tensor) -> torch.Tensor:
        '''(P, K, 3): z_p [reference pixels], a_p [um], n_p - n_m'''
        p, _ = self(x)
        center = torch.as_tensor(CENTER, device=p.device, dtype=p.dtype)
        return torch.exp(p / 100 + center)
