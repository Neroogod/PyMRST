"""Mutable mapping wrapper corresponding to MRST ``HandleStruct``."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from pymrst.utils.adi import value


class HandleStruct:
    """Expose dictionary fields as mutable attributes with reduction helpers."""

    def __init__(self, structure=None):
        object.__setattr__(self, "_data", dict(structure or {}))

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, item):
        self._data[name] = item

    def __contains__(self, name):
        return name in self._data

    def isfield(self, name):
        return name in self._data

    def struct_prop_evaluated(self, name):
        item = self._data[name]
        if item is None:
            return False
        if isinstance(item, (list, tuple)):
            return len(item) > 0
        return np.asarray(value(item)).size > 0

    structPropEvaluated = struct_prop_evaluated

    def as_struct(self):
        return dict(self._data)

    struct = as_struct

    def reduce(self):
        output = {}
        for name, item in self._data.items():
            if hasattr(item, "val") and hasattr(item, "jac"):
                output[name] = np.asarray(value(item))
            elif isinstance(item, (list, tuple)):
                pieces = [np.atleast_1d(value(entry)) for entry in item]
                output[name] = np.concatenate(pieces) if pieces else np.array([])
            else:
                output[name] = item
        return output

    def value(self):
        return {
            name: value(item) if hasattr(item, "val") else item
            for name, item in self._data.items()
        }

    def copy(self):
        return HandleStruct(deepcopy(self._data))


__all__ = ["HandleStruct"]
