'''Resizing of images with torch, in the manner of OpenCV

``resize`` reproduces ``cv2.resize`` with ``INTER_AREA`` (shrinking: each new
pixel is the average of the area that it covers) and ``INTER_CUBIC``
(enlarging: bicubic with the coefficient -0.75), to float rounding, so that
CATCH does not need OpenCV for its one use of it.
'''
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['resize']


def _coverage(n_old: int, n_new: int) -> torch.Tensor:
    '''Fraction of each new pixel's length that each old pixel covers'''
    step = n_old / n_new
    start = torch.arange(n_new, dtype=torch.float64)[:, None] * step
    old = torch.arange(n_old, dtype=torch.float64)[None, :]
    overlap = torch.minimum(start + step, old + 1) - torch.maximum(start, old)
    return overlap.clamp(min=0.) / step


def resize(image: np.ndarray, size: tuple[int, int],
           shrink: bool) -> np.ndarray:
    '''Resize a gray-scale image

    Parameters
    ----------
    image : numpy.ndarray
        Floating-point image of shape ``(height, width)``.
    size : tuple[int, int]
        New ``(width, height)`` [pixels], the order of ``cv2.resize``.
    shrink : bool
        True: average over the area of each new pixel (for a smaller
        image).  False: bicubic interpolation (for a larger one).

    Returns
    -------
    resized : numpy.ndarray
        The image at the new size, of the dtype of ``image``.
    '''
    width, height = size
    t = torch.from_numpy(np.ascontiguousarray(image)).to(torch.float64)
    if shrink:
        t = (_coverage(t.shape[0], height) @ t
             @ _coverage(t.shape[1], width).T)
    else:
        t = F.interpolate(t[None, None], size=(height, width),
                          mode='bicubic', align_corners=False)[0, 0]
    return t.numpy().astype(image.dtype)
