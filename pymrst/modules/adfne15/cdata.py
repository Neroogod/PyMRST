"""Native port of numerical utilities in ADFNE15 ``CData.m``."""

import numpy as np

from .translate import translate


class CData:
    @staticmethod
    def almost(values, reference, tolerance=1e-14):
        return np.abs(np.asarray(values)-reference) <= tolerance

    @staticmethod
    def deal(values):
        values = np.asarray(values)
        return tuple(values[:, i] for i in range(values.shape[1]))

    @staticmethod
    def discretize(values, count):
        values = np.asarray(values, dtype=float)
        if count < 1: raise ValueError("count must be positive")
        if np.ptp(values) == 0: return np.ones(values.shape, dtype=int)
        edges = np.linspace(np.min(values), np.max(values), int(count)+1)
        return np.clip(np.digitize(values, edges[1:-1])+1, 1, count)

    @staticmethod
    def get(values, index, *, one_based=True):
        values = np.asarray(values)
        row = int(index)-1 if one_based else int(index)
        return values[min(max(row, 0), len(values)-1)]

    @staticmethod
    def leg(values, reference, tolerance=1e-14):
        values = np.asarray(values)
        equal = np.abs(values-reference) <= tolerance
        return (values < reference) & ~equal, equal, (values > reference) & ~equal

    @staticmethod
    def map(values, *functions):
        return np.asarray([function(values) for function in functions])

    @staticmethod
    def array(values, points):
        candidate = np.asarray(values)
        packed = (values if candidate.dtype == object or candidate.ndim > 2
                  else np.asarray(values, dtype=float))
        return [translate(packed, point) for point in np.asarray(points)]
