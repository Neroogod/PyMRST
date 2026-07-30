"""Pack and unpack well-perforation properties."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.utils.adi import value


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _cell_subset(data, cells):
    subset = []
    for item in data:
        if isinstance(item, (list, tuple)):
            subset.append(_cell_subset(item, cells))
        elif item is None:
            subset.append(None)
        else:
            array = np.asarray(value(item))
            subset.append(array[cells] if array.size else array.copy())
    return subset


def pack_perforation_properties(
    well,
    pressure,
    mobility,
    density,
    dissolved,
    components,
    wellvars,
    wellvarnames,
    varmaps,
    wellmap,
    well_index,
):
    """Extract reservoir and extra variables for one well's perforations.

    Python well and map indices are zero-based. Positive entries in
    ``wellmap`` are one-based selectors, matching the MRST bookkeeping
    convention; zero entries denote padding.
    """
    cells = np.asarray(_get(well, "cells"), dtype=np.int64).ravel()
    variable_subset = [
        np.asarray(value(variable))[np.asarray(mapping) == well_index]
        for variable, mapping in zip(wellvars, varmaps)
    ]
    map_row = np.asarray(wellmap, dtype=np.int64)[well_index].ravel()
    renumbering = map_row[map_row > 0] - 1
    extra = [variable_subset[index] for index in renumbering]
    return SimpleNamespace(
        pressure=np.asarray(value(pressure))[cells],
        mob=_cell_subset(mobility, cells),
        rho=_cell_subset(density, cells),
        dissolved=_cell_subset(dissolved, cells),
        components=_cell_subset(components, cells),
        extravars=extra,
        extravars_names=list(wellvarnames),
    )


def unpack_perforation_properties(packed):
    """Return the six property groups stored by the packing helper."""
    return (
        _get(packed, "pressure"),
        _get(packed, "mob"),
        _get(packed, "rho"),
        _get(packed, "dissolved"),
        _get(packed, "components"),
        _get(packed, "extravars"),
    )


packPerforationProperties = pack_perforation_properties
unpackPerforationProperties = unpack_perforation_properties

__all__ = [
    "pack_perforation_properties",
    "packPerforationProperties",
    "unpack_perforation_properties",
    "unpackPerforationProperties",
]
