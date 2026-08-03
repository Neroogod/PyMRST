"""Native port of ADFNE15 ``Info.m``."""

from dataclasses import dataclass
import numpy as np

from .center import center
from .length import lengths
from .orientation import orientation


@dataclass(frozen=True, slots=True)
class NetworkInfo:
    count: int
    centers: np.ndarray
    lengths: np.ndarray
    angles: np.ndarray | None = None
    dips: np.ndarray | None = None
    directions: np.ndarray | None = None


def info(values):
    polygonal = isinstance(values, (list, tuple))
    result = orientation(values)
    return NetworkInfo(len(values), np.asarray(center(values)), lengths(values),
                       None if polygonal else result.direction,
                       result.dip if polygonal else None,
                       result.direction if polygonal else None)
