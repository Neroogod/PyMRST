"""MRST ``ad-core/models/facilities/packPerforationProperties.m``: one well's
perforation quantities - cell pressures, mobilities, densities, dissolution
ratios and component data in its perforated cells (AD values keep their
derivatives, as MRST's `reduceToDouble` leaves them), and the facility
variables belonging to the well, renumbered by the well map. Well and map
indices are 0-based; positive `wellmap` entries are 1-based selectors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _cell_subset(data, cells):
    subset = []
    for item in data:
        if isinstance(item, (list, tuple)):
            subset.append(_cell_subset(item, cells))
        elif item is None:
            subset.append(None)
        elif hasattr(item, "jac"):
            # MRST's `reduceToDouble(celldata{i}(wc))` - and `reduceToDouble`
            # is empty, so derivatives survive. Dropping them here left every
            # well's inflow without a derivative in the reservoir unknowns.
            subset.append(item[cells])
        else:
            array = np.asarray(item)
            subset.append(array[cells] if array.size else array.copy())
    return subset


def _rows(item, rows):
    """`item(rows)`, keeping derivatives when `item` is AD."""
    if hasattr(item, "jac"):
        return item[rows]
    return np.asarray(item)[rows]


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
        _rows(variable, np.flatnonzero(np.asarray(mapping) == well_index))
        for variable, mapping in zip(wellvars, varmaps)
    ]
    map_row = np.asarray(wellmap, dtype=np.int64)[well_index].ravel()
    renumbering = map_row[map_row > 0] - 1
    extra = [variable_subset[index] for index in renumbering]
    return SimpleNamespace(
        pressure=_rows(pressure, cells),
        mob=_cell_subset(mobility, cells),
        rho=_cell_subset(density, cells),
        dissolved=_cell_subset(dissolved, cells),
        components=_cell_subset(components, cells),
        extravars=extra,
        extravars_names=list(wellvarnames),
    )


packPerforationProperties = pack_perforation_properties
__all__ = ["packPerforationProperties", "pack_perforation_properties"]


if __name__ == "__main__":
    def test_cells_and_well_variables_are_selected():
        well = {"cells": np.array([2, 0])}
        packed = pack_perforation_properties(
            well, np.array([10.0, 11.0, 12.0]), [np.array([1.0, 2.0, 3.0])], [np.ones(3)],
            [[None, np.array([5.0, 6.0, 7.0])]], [], [np.array([100.0, 200.0])], ["qWs"],
            [np.array([0, 1])], np.array([[1], [1]]), 1)
        assert packed.pressure.tolist() == [12.0, 10.0] and packed.mob[0].tolist() == [3.0, 1.0]
        assert packed.dissolved[0][0] is None and packed.dissolved[0][1].tolist() == [7.0, 5.0]
        assert [v.tolist() for v in packed.extravars] == [[200.0]] and packed.extravars_names == ["qWs"]
        print("ok test_cells_and_well_variables_are_selected")

    test_cells_and_well_variables_are_selected()
    print("\nAll tests passed!")
