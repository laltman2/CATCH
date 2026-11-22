'''Matching the particles that a pipeline finds to the particles in a scene'''
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Match:
    '''The result of matching detections to the truth

    Attributes
    ----------
    truth : numpy.ndarray
        Indexes of the true particles that were found.
    detection : numpy.ndarray
        Index of the detection that goes with each of them.
    distance : numpy.ndarray
        Lateral distance [pixels] between the two.
    missed : numpy.ndarray
        Indexes of the true particles that were not found.
    spurious : numpy.ndarray
        Indexes of the detections that match no particle.
    '''
    truth: np.ndarray
    detection: np.ndarray
    distance: np.ndarray
    missed: np.ndarray
    spurious: np.ndarray


def match(truth_xy: ArrayLike, detection_xy: ArrayLike,
          gate: float = 15.) -> Match:
    '''Matches each detection to the nearest particle that is within a gate

    The closest pairs are matched first, and a particle or a detection
    is used once.

    Parameters
    ----------
    truth_xy : array_like
        Positions of the particles: shape ``(N, 2)``.
    detection_xy : array_like
        Positions of the detections: shape ``(M, 2)``.
    gate : float
        Largest distance [pixels] for a match.  Default: 15, which is how far
        the fit is allowed to move the center.
    '''
    truth_xy = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    detection_xy = np.asarray(detection_xy, dtype=float).reshape(-1, 2)
    distance = np.hypot(*(truth_xy[:, None, :]
                          - detection_xy[None, :, :]).transpose(2, 0, 1))
    order = np.argsort(distance, axis=None, kind='stable')
    used_truth: set[int] = set()
    used_detection: set[int] = set()
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, distance.shape)
        i, j = int(i), int(j)
        if distance[i, j] > gate:
            break
        if i in used_truth or j in used_detection:
            continue
        used_truth.add(i)
        used_detection.add(j)
        pairs.append((i, j))
    pairs.sort()
    truth = np.array([i for i, _ in pairs], dtype=int)
    detection = np.array([j for _, j in pairs], dtype=int)
    return Match(
        truth=truth, detection=detection,
        distance=distance[truth, detection],
        missed=np.setdiff1d(np.arange(len(truth_xy)), truth),
        spurious=np.setdiff1d(np.arange(len(detection_xy)), detection))
