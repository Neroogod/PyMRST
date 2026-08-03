"""Native port of ADFNE15 ``Stats.m``."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Statistics:
    minimum: float
    mean: float
    maximum: float
    standard_deviation: float
    count: int


def stats(values, label=None, *, display=False):
    flat = np.asarray(values, dtype=float).reshape(-1)
    result = Statistics(float(np.nanmin(flat)), float(np.nanmean(flat)),
                        float(np.nanmax(flat)), float(np.nanstd(flat, ddof=1)),
                        flat.size)
    if display:
        print(f"----[ {label or ''} ]\nn   : {result.count:g}\n"
              f"min : {result.minimum:g}\nmean: {result.mean:g}\n"
              f"max : {result.maximum:g}\nstd : {result.standard_deviation:g}\n")
    return result
