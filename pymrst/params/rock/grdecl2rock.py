"""Build a rock structure from GRDECL petrophysics keywords (MRST `grdecl2Rock`).

An ECLIPSE GRDECL deck stores permeability and porosity as flat per-cell arrays,
one keyword per tensor component: `PERMX`/`PERMY`/`PERMZ` for a diagonal tensor,
`PERMH` for a single horizontal value, the `PERMXY`/`PERMXZ`/`PERMYZ` family for a
full symmetric tensor, plus `PORO`, `NTG`, `ROCKTYPE` and the `MULTX/Y/Z`
multipliers. This assembles those scattered keywords into one `Rock`: an
`(nc, 1)` isotropic, `(nc, 3)` diagonal or `(nc, 6)` full-symmetric permeability
(the column layout `make_rock` expects), the porosity and net-to-gross, and any
transmissibility multipliers or fault data the deck carries.

It is the front half of every deck-driven case - the rock that pairs with the
grid `processGRDECL` builds - and mirrors MRST's keyword logic, including the
"a single PERMX means isotropic" and "fill missing diagonal components" rules.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pymrst.params.rock.rock import Rock

_PERM_KEYS = ("PERMX", "PERMY", "PERMH", "PERMZ", "PERMXX", "PERMYY", "PERMZZ")


def _consistent(grdecl: Dict[str, Any]) -> bool:
    """MRST's structural consistency check: an off-diagonal component may only
    appear if its row's and column's diagonals are present."""
    def h(*names) -> bool:
        return any(n in grdecl for n in names)

    k = np.zeros((3, 3), dtype=bool)
    k[0, 0] = h("PERMX", "PERMXX", "PERMH")
    k[0, 1] = h("PERMXY", "PERMYX")
    k[0, 2] = h("PERMXZ", "PERMZX")
    k[1, 0] = h("PERMYX", "PERMXY")
    k[1, 1] = h("PERMY", "PERMYY", "PERMH")
    k[1, 2] = h("PERMYZ", "PERMZY")
    k[2, 0] = h("PERMZX", "PERMXZ")
    k[2, 1] = h("PERMZY", "PERMYZ")
    k[2, 2] = h("PERMZ", "PERMZZ")

    b = k.any()
    b = b and (k[0, 0] or not (k[0, 1] or k[1, 0] or k[0, 2] or k[2, 0]))
    b = b and (k[1, 1] or not (k[1, 0] or k[0, 1] or k[1, 2] or k[2, 1]))
    b = b and (k[2, 2] or not (k[2, 0] or k[0, 2] or k[2, 1] or k[1, 2]))
    return bool(b)


def _set_diagonal(c: np.ndarray, i: int, j: int, k: int) -> None:
    """Fill unset diagonal components j, k from a present one i (MRST's
    `setDiagonalComponents`): PERMX alone makes an isotropic tensor."""
    if c[j, j] == 1:
        c[j, j] = c[i, i]
    if c[k, k] == 1:
        c[k, k] = c[i, i]


def _tensor_values(grdecl: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(vals, comp)`: `vals` is `(nc, K)` with column 0 all zeros, and
    `comp` is a 3x3 of 1-based column indices into `vals` per tensor component."""
    nc = 0
    for key in _PERM_KEYS:
        if nc == 0 and key in grdecl:
            nc = np.asarray(grdecl[key]).size
    assert nc > 0, "no permeability keyword found"

    cols = [np.zeros(nc)]                       # 1-based column index 1 = zeros
    comp = np.ones((3, 3), dtype=np.int64)

    def add(key: str) -> int:
        values = np.asarray(grdecl[key], dtype=float).ravel()
        if values.size != nc:
            raise ValueError(
                f"{key} contains {values.size} values; expected {nc}"
            )
        cols.append(values)
        return len(cols)                        # 1-based index of the new column

    has = grdecl.__contains__
    if has("PERMX") or has("PERMXX") or has("PERMH"):
        idx = add("PERMX" if has("PERMX") else "PERMH" if has("PERMH") else "PERMXX")
        comp[0, 0] = idx
        _set_diagonal(comp, 0, 1, 2)
    if has("PERMXY"):
        idx = add("PERMXY"); comp[0, 1] = comp[1, 0] = idx
    if has("PERMXZ"):
        idx = add("PERMXZ"); comp[0, 2] = comp[2, 0] = idx

    if has("PERMYX"):
        idx = add("PERMYX"); comp[1, 0] = comp[0, 1] = idx
    if has("PERMY") or has("PERMYY") or has("PERMH"):
        idx = add("PERMY" if has("PERMY") else "PERMH" if has("PERMH") else "PERMYY")
        comp[1, 1] = idx
        _set_diagonal(comp, 1, 2, 0)
    if has("PERMYZ"):
        idx = add("PERMYZ"); comp[1, 2] = comp[2, 1] = idx

    if has("PERMZX"):
        idx = add("PERMZX"); comp[2, 0] = comp[0, 2] = idx
    if has("PERMZY"):
        idx = add("PERMZY"); comp[2, 1] = comp[1, 2] = idx
    if has("PERMZ") or has("PERMZZ"):
        idx = add("PERMZ" if has("PERMZ") else "PERMZZ")
        comp[2, 2] = idx
        _set_diagonal(comp, 2, 0, 1)

    return np.column_stack(cols), comp


def _get_tensor(grdecl: Dict[str, Any]) -> np.ndarray:
    """The permeability tensor over all cells: `(nc, 1)`, `(nc, 3)` or
    `(nc, 6)`."""
    vals, comp = _tensor_values(grdecl)
    i, j = np.where(comp > 1)
    if i.size and np.all(i == j):
        if i.size != 3:
            raise ValueError("incomplete diagonal permeability tensor")
        sel = comp[[0, 1, 2], [0, 1, 2]]
    else:
        if i.size != 9:
            raise ValueError("incomplete full symmetric permeability tensor")
        ii = np.array([0, 0, 0, 1, 1, 2])
        jj = np.array([0, 1, 2, 1, 2, 2])
        sel = comp[ii, jj]
    if sel.size == 3 and np.all(np.diff(sel) == 0):
        sel = sel[:1]                           # isotropic -> single column
    return vals[:, sel - 1]                     # 1-based -> 0-based


def grdecl2rock(grdecl: Dict[str, Any], actmap=None) -> Rock:
    """Assemble a `Rock` from GRDECL keywords.

    `grdecl` maps ECLIPSE keyword names (upper-case) to flat per-cell arrays.
    `actmap` restricts to active cells: an index array, or a grid (its
    `cells.indexMap` is used); default keeps every cell. Multipliers and fault
    data, if present, are attached as `.multipliers` / `.faultdata`.
    """
    if not _consistent(grdecl):
        raise ValueError("permeability tensor keywords are not consistent")

    perm = _get_tensor(grdecl)
    property_count = perm.shape[0]

    if actmap is None:
        actmap = np.arange(perm.shape[0], dtype=np.int64)
    elif hasattr(actmap, "cells"):
        actmap = np.asarray(actmap.cells.indexMap, dtype=np.int64).ravel()
    else:
        raw_map = np.asarray(actmap)
        if raw_map.dtype == bool:
            if raw_map.size != perm.shape[0]:
                raise ValueError("logical actmap must contain one flag per cell")
            actmap = np.flatnonzero(raw_map.ravel())
        else:
            actmap = np.asarray(raw_map, dtype=np.int64).ravel()

    if actmap.size and (actmap.min() < 0 or actmap.max() >= perm.shape[0]):
        raise IndexError("actmap contains a cell index outside the property arrays")

    perm = perm[actmap]

    def extract(field: str) -> np.ndarray:
        values = np.asarray(grdecl[field], dtype=float).ravel()
        if values.size != property_count:
            raise ValueError(
                f"{field} contains {values.size} values; expected {property_count}"
            )
        return values[actmap].reshape(-1, 1)

    poro = extract("PORO").ravel() if "PORO" in grdecl else None
    ntg = extract("NTG").ravel() if "NTG" in grdecl else None
    rock = Rock(perm=perm, poro=poro, ntg=ntg)

    if "ROCKTYPE" in grdecl:
        rock.rocktype = extract("ROCKTYPE").ravel()

    mult = {}
    for axis in ("X", "Y", "Z"):
        for key in (f"MULT{axis}", f"MULT{axis}_"):
            if key in grdecl:
                mult[key[4:].lower()] = extract(key).ravel()
    if mult:
        rock.multipliers = mult

    if "FAULTS" in grdecl and "MULTFLT" in grdecl:
        rock.faultdata = {"faults": grdecl["FAULTS"], "multflt": grdecl["MULTFLT"]}

    return rock


if __name__ == "__main__":
    def test_single_permx_is_isotropic():
        """A lone PERMX makes an isotropic (single-column) permeability."""
        g = {"PERMX": np.array([1.0, 2.0, 3.0]), "PORO": np.array([0.1, 0.2, 0.3])}
        rock = grdecl2rock(g)
        assert rock.perm.shape == (3, 1), rock.perm.shape
        assert np.allclose(rock.perm[:, 0], [1.0, 2.0, 3.0])
        assert np.allclose(rock.poro.ravel(), [0.1, 0.2, 0.3])
        print("ok test_single_permx_is_isotropic")

    def test_distinct_permxyz_is_diagonal_three_columns():
        g = {"PERMX": np.array([1.0, 2.0]), "PERMY": np.array([3.0, 4.0]),
             "PERMZ": np.array([5.0, 6.0])}
        rock = grdecl2rock(g)
        assert rock.perm.shape == (2, 3), rock.perm.shape
        assert np.allclose(rock.perm, [[1, 3, 5], [2, 4, 6]])
        print("ok test_distinct_permxyz_is_diagonal_three_columns")

    def test_separate_equal_permxyz_stays_three_columns():
        """MRST's isotropic collapse compares *column indices*, not values: three
        separately-given keywords are three columns even when equal (only a lone
        PERMX, which reuses one column for all diagonals, collapses)."""
        g = {"PERMX": np.array([2.0]), "PERMY": np.array([2.0]),
             "PERMZ": np.array([2.0])}
        rock = grdecl2rock(g)
        assert rock.perm.shape == (1, 3), rock.perm.shape
        assert np.allclose(rock.perm, [[2.0, 2.0, 2.0]])
        print("ok test_separate_equal_permxyz_stays_three_columns")

    def test_full_symmetric_tensor_is_six_columns():
        n = 2
        g = {"PERMX": np.full(n, 1.0), "PERMY": np.full(n, 2.0),
             "PERMZ": np.full(n, 3.0), "PERMXY": np.full(n, 0.1),
             "PERMXZ": np.full(n, 0.2), "PERMYZ": np.full(n, 0.3)}
        rock = grdecl2rock(g)
        assert rock.perm.shape == (n, 6), rock.perm.shape
        # column order [xx, xy, xz, yy, yz, zz]
        assert np.allclose(rock.perm[0], [1.0, 0.1, 0.2, 2.0, 0.3, 3.0])
        print("ok test_full_symmetric_tensor_is_six_columns")

    def test_actmap_selects_active_cells():
        g = {"PERMX": np.array([1.0, 2.0, 3.0, 4.0]),
             "PORO": np.array([0.1, 0.2, 0.3, 0.4])}
        rock = grdecl2rock(g, actmap=np.array([0, 2]))
        assert np.allclose(rock.perm[:, 0], [1.0, 3.0])
        assert np.allclose(rock.poro.ravel(), [0.1, 0.3])
        print("ok test_actmap_selects_active_cells")

    def test_multipliers_are_attached():
        g = {"PERMX": np.array([1.0, 1.0]), "MULTZ": np.array([0.5, 0.7])}
        rock = grdecl2rock(g)
        assert np.allclose(rock.multipliers["z"].ravel(), [0.5, 0.7])
        print("ok test_multipliers_are_attached")

    test_single_permx_is_isotropic()
    test_distinct_permxyz_is_diagonal_three_columns()
    test_separate_equal_permxyz_stays_three_columns()
    test_full_symmetric_tensor_is_six_columns()
    test_actmap_selects_active_cells()
    test_multipliers_are_attached()
    print("\nAll tests passed!")
