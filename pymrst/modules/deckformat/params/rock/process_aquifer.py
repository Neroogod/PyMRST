"""MRST ``deckformat/params/rock/processAquifer.m``.

The function expands ECLIPSE ``AQUANCON`` connection boxes, removes inactive
or interior-neighbor cells, maps face directions to the grid half-faces, and
returns the connection table consumed by the Fetkovich ``AquiferModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pymrst.gridprocessing.grid import BOTTOM, EAST, NORTH, SOUTH, TOP, WEST, Grid


@dataclass(frozen=True)
class AquiferIndices:
    aquid: int = 0
    conn: int = 1
    pvttbl: int = 2
    J: int = 3
    C: int = 4
    alpha: int = 5
    depthconn: int = 6
    depthaq: int = 7


@dataclass(frozen=True)
class AquiferProps:
    depthaq: np.ndarray
    C: np.ndarray
    J: np.ndarray
    pvttbl: np.ndarray


@dataclass(frozen=True)
class AquiferDeckOutput:
    aquifers: np.ndarray
    aquind: AquiferIndices
    initval: Mapping[str, np.ndarray]
    aquiferprops: AquiferProps


_DIR_TO_FLAG = {
    "I-": 1,
    "I+": 2,
    "J-": 3,
    "J+": 4,
    "K-": 5,
    "K+": 6,
    "X-": 1,
    "X+": 2,
    "Y-": 3,
    "Y+": 4,
    "Z-": 5,
    "Z+": 6,
}
_FLAG_TO_TAG = {1: WEST, 2: EAST, 3: SOUTH, 4: NORTH, 5: TOP, 6: BOTTOM}


def _solution(deck: Mapping[str, Any]) -> Mapping[str, Any]:
    return deck.get("SOLUTION", deck)


def _get(deck: Mapping[str, Any], name: str) -> Any:
    sol = _solution(deck)
    return sol[name] if name in sol else sol[name.upper()]


def _num(x: Any, default: float = np.nan) -> float:
    if x is None:
        return default
    if isinstance(x, str):
        s = x.strip().strip("'\"")
        if not s or s == "1*":
            return default
        return float(s)
    return float(x)


def _aqufetp_table(raw: Any) -> np.ndarray:
    table = np.asarray(raw, dtype=float)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 7:
        raise ValueError("AQUFETP needs at least seven columns")
    if np.any(np.isnan(table[:, 2])):
        raise ValueError("defaulted aquifer pressure is not supported")
    return table[:, :7]


def _aquancon_records(raw: Any) -> list[tuple[int, int, int, int, int, int, int, int, float, float]]:
    records = []
    for rec in raw:
        if isinstance(rec, Mapping):
            aquid = int(_num(rec.get("aquifer") or rec.get("aquid") or rec.get("id")))
            imin = int(_num(rec.get("i1") or rec.get("imin")))
            imax = int(_num(rec.get("i2") or rec.get("imax")))
            jmin = int(_num(rec.get("j1") or rec.get("jmin")))
            jmax = int(_num(rec.get("j2") or rec.get("jmax")))
            kmin = int(_num(rec.get("k1") or rec.get("kmin")))
            kmax = int(_num(rec.get("k2") or rec.get("kmax")))
            direc = rec.get("dir") or rec.get("direction") or rec.get("face")
            influx = _num(rec.get("influxcoef") or rec.get("influx_coef"))
            mult = _num(rec.get("influxmultcoef") or rec.get("influx_mult") or rec.get("mult"), 1.0)
        else:
            vals = list(rec)
            if len(vals) < 10:
                raise ValueError("AQUANCON records need ten fields")
            aquid = int(_num(vals[0]))
            imin, imax = int(_num(vals[1])), int(_num(vals[2]))
            jmin, jmax = int(_num(vals[3])), int(_num(vals[4]))
            kmin, kmax = int(_num(vals[5])), int(_num(vals[6]))
            direc = vals[7]
            influx, mult = _num(vals[8]), _num(vals[9], 1.0)
        flag = _DIR_TO_FLAG[str(direc).strip().strip("'\"").upper()]
        records.append((aquid, imin, imax, jmin, jmax, kmin, kmax, flag, influx, mult))
    return records


def _cart_to_linear(i: np.ndarray, j: np.ndarray, k: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    nx, ny, _ = dims
    return i + nx * j + nx * ny * k


def _cart_to_active(g: Grid, linear: np.ndarray) -> np.ndarray:
    index_map = np.asarray(g.cells.indexMap, dtype=np.int64)
    active = np.full(int(np.prod(g.cartDims)), -1, dtype=np.int64)
    active[index_map] = np.arange(index_map.size)
    out = active[np.asarray(linear, dtype=np.int64)]
    if np.any(out < 0):
        raise ValueError("aquifer connection includes inactive cells")
    return out


def _face_area_for_flag(g: Grid, cells: np.ndarray, flags: np.ndarray) -> np.ndarray:
    areas = np.empty(cells.size, dtype=float)
    for n, (cell, flag) in enumerate(zip(cells, flags)):
        lo, hi = g.cells.facePos[cell], g.cells.facePos[cell + 1]
        faces = g.cells.faces[lo:hi]
        tags = g.cells.tags[lo:hi] if g.cells.tags is not None else None
        tag = _FLAG_TO_TAG[int(flag)]
        if tags is not None:
            hit = np.flatnonzero(tags == tag)
            if hit.size:
                areas[n] = g.faces.areas[faces[hit[0]]]
                continue
        nbs = g.faces.neighbors[faces]
        boundary = np.flatnonzero(np.any(nbs < 0, axis=1))
        if boundary.size:
            areas[n] = g.faces.areas[faces[boundary[0]]]
            continue
        raise ValueError(f"could not find face for aquifer cell {cell} flag {flag}")
    return areas


def _expand(records: Iterable[tuple[int, int, int, int, int, int, int, int, float, float]]) -> np.ndarray:
    # MRST expands the boxes with mcolon/rldecode on i, then j, then k: per
    # record, i varies slowest and k fastest.
    rows = []
    for aquid, imin, imax, jmin, jmax, kmin, kmax, flag, influx, mult in records:
        for i in range(imin - 1, imax):
            for j in range(jmin - 1, jmax):
                for k in range(kmin - 1, kmax):
                    rows.append((i, j, k, aquid, influx, mult, flag))
    return np.asarray(rows, dtype=float)


def process_aquifer(deck: Mapping[str, Any], g: Grid) -> AquiferDeckOutput:
    """Process ``AQUANCON``/``AQUFETP`` into MRST's aquifer connection table."""
    if g.cartDims is None or len(g.cartDims) != 3:
        raise ValueError("process_aquifer requires a 3D cartesian grid")
    if g.faces.areas is None or g.cells.centroids is None:
        raise ValueError("grid geometry must be computed before process_aquifer")

    aqufetp = _aqufetp_table(_get(deck, "AQUFETP"))
    rows = _expand(_aquancon_records(_get(deck, "AQUANCON")))
    if rows.size == 0:
        ind = AquiferIndices()
        props = AquiferProps(*(np.zeros(0) for _ in range(4)))
        return AquiferDeckOutput(np.zeros((0, 8)), ind, {"pressures": np.zeros(0), "volumes": np.zeros(0)}, props)

    dims = np.asarray(g.cartDims, dtype=np.int64)
    ijk = rows[:, :3].astype(np.int64)
    in_domain = np.all((ijk >= 0) & (ijk < dims), axis=1)
    rows = rows[in_domain]
    ijk = ijk[in_domain]
    linear = _cart_to_linear(ijk[:, 0], ijk[:, 1], ijk[:, 2], dims)
    active = np.zeros(int(np.prod(dims)), dtype=bool)
    active[np.asarray(g.cells.indexMap, dtype=np.int64)] = True
    keep = active[linear]
    rows, ijk, linear = rows[keep], ijk[keep], linear[keep]

    nbr = ijk.copy()
    flags = rows[:, 6].astype(np.int64)
    nbr[flags == 1, 0] -= 1
    nbr[flags == 2, 0] += 1
    nbr[flags == 3, 1] -= 1
    nbr[flags == 4, 1] += 1
    nbr[flags == 5, 2] -= 1
    nbr[flags == 6, 2] += 1
    nbr_out = np.any((nbr < 0) | (nbr >= dims), axis=1)
    nbr_lin = np.zeros(rows.shape[0], dtype=np.int64)
    inside = ~nbr_out
    nbr_lin[inside] = _cart_to_linear(nbr[inside, 0], nbr[inside, 1], nbr[inside, 2], dims)
    nbr_inactive = np.zeros(rows.shape[0], dtype=bool)
    nbr_inactive[inside] = ~active[nbr_lin[inside]]
    # MRST's `[aqcellout; aqcellin]`: the connections facing out of the box
    # first, then those facing an inactive cell.
    order = np.concatenate([np.flatnonzero(nbr_out), np.flatnonzero(nbr_inactive)])
    rows, linear = rows[order], linear[order]

    conn = _cart_to_active(g, linear)
    flags = rows[:, 6].astype(np.int64)
    influx = rows[:, 4].astype(float)
    use_area = np.isnan(influx)
    if np.any(use_area):
        influx[use_area] = _face_area_for_flag(g, conn[use_area], flags[use_area])
    alpha_raw = influx * rows[:, 5].astype(float)

    aquid_one = rows[:, 3].astype(np.int64)
    aq_index = aquid_one - 1
    n_aq = int(max(np.max(aquid_one), np.max(aqufetp[:, 0]))) if aqufetp.size else int(np.max(aquid_one))
    sum_alpha = np.bincount(aq_index, weights=alpha_raw, minlength=n_aq)
    alpha = alpha_raw / sum_alpha[aq_index]

    props = AquiferProps(
        depthaq=aqufetp[:, 1].astype(float),
        C=aqufetp[:, 4].astype(float),
        J=aqufetp[:, 5].astype(float),
        pvttbl=aqufetp[:, 6].astype(float),
    )
    aquifers = np.empty((conn.size, 8), dtype=float)
    aquifers[:, 0] = aq_index
    aquifers[:, 1] = conn
    aquifers[:, 2] = props.pvttbl[aq_index]
    aquifers[:, 3] = props.J[aq_index]
    aquifers[:, 4] = props.C[aq_index]
    aquifers[:, 5] = alpha
    aquifers[:, 6] = g.cells.centroids[conn, 2]
    aquifers[:, 7] = props.depthaq[aq_index]

    initval = {
        "pressures": aqufetp[:, 2].astype(float),
        "volumes": aqufetp[:, 3].astype(float),
    }
    return AquiferDeckOutput(aquifers, AquiferIndices(), initval, props)


__all__ = ["AquiferDeckOutput", "AquiferIndices", "AquiferProps", "process_aquifer"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.units import BARSA

    def test_process_aquifer_box_and_area_defaults():
        g = compute_geometry(cart_grid([3, 2, 1], [30.0, 20.0, 5.0]))
        deck = {
            "SOLUTION": {
                "AQUANCON": [
                    [1, 3, 3, 1, 2, 1, 1, "I+", np.nan, 1.0],
                ],
                "AQUFETP": np.array([[1, 2.5, 250 * BARSA, 1e7, 1e-9, 3e-10, 1]]),
            }
        }
        out = process_aquifer(deck, g)
        assert out.aquifers.shape == (2, 8)
        assert np.array_equal(out.aquifers[:, out.aquind.conn].astype(int), [2, 5])
        assert np.allclose(out.aquifers[:, out.aquind.alpha], [0.5, 0.5])
        assert np.allclose(out.initval["pressures"], [250 * BARSA])
        print("ok test_process_aquifer_box_and_area_defaults")

    def test_connections_come_in_mrsts_order():
        """i slowest and k fastest within a record, as MRST's mcolon expansion
        gives them (checked against MRST 2024a on the aquifertest decks)."""
        g = compute_geometry(cart_grid([2, 1, 2], [20.0, 10.0, 10.0]))
        deck = {"SOLUTION": {
            "AQUANCON": [[1, 1, 2, 1, 1, 1, 2, "J-", 1.0, 1.0]],
            "AQUFETP": np.array([[1, 2.5, 250*BARSA, 1e7, 1e-9, 3e-10, 1]])}}
        out = process_aquifer(deck, g)
        assert out.aquifers[:, out.aquind.conn].astype(int).tolist() == [0, 2, 1, 3]
        print("ok test_connections_come_in_mrsts_order")

    test_process_aquifer_box_and_area_defaults()
    test_connections_come_in_mrsts_order()
