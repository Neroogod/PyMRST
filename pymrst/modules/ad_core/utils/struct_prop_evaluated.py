"""Check whether a named structure property is nonempty."""

from __future__ import annotations

import numpy as np


def struct_prop_evaluated(structure, name):
    """Port MRST ``structPropEvaluated`` for mappings and objects."""
    item = structure[name] if isinstance(structure, dict) else getattr(structure, name)
    if item is None:
        return False
    try:
        return np.asarray(item).size > 0
    except ValueError:
        return len(item) > 0


structPropEvaluated = struct_prop_evaluated

__all__ = ["struct_prop_evaluated", "structPropEvaluated"]
