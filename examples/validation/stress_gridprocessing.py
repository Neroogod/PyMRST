"""Stress the gridprocessing stack on large decks and check it stays valid.

Four scenarios, each the large-scale version of something the unit tests only
cover in miniature: a faulted corner-point deck, a grid that falls into many
disconnected regions, a dense NNC network, and repeated slicing. Every case is
checked against the same structural invariants, so a failure says which grid
broke and how, rather than only that a number moved.

Run directly for a report:

    python3 examples/validation/stress_gridprocessing.py            # medium
    python3 examples/validation/stress_gridprocessing.py --size large

Sizes are `small` (seconds, what the test suite runs), `medium` (the default,
~1 minute) and `large` (200k cells, several minutes and a few GB).
"""

import argparse
import time
import tracemalloc
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.process_nnc import process_nnc
from pymrst.gridprocessing.slice_grid import slice_grid
from pymrst.gridprocessing.split_disconnected_grid import split_disconnected_grid
from pymrst.gridprocessing.testgrids.one_sloping_fault import one_sloping_fault

SIZES: Dict[str, Dict[str, object]] = {
    "small": {"fault": (20, 20, 5), "islands": (12, 12, 4), "nnc": (20, 20, 10),
              "slice": (10, 10, 4), "cuts": 4},
    "medium": {"fault": (60, 60, 20), "islands": (30, 30, 8), "nnc": (40, 40, 20),
               "slice": (20, 20, 8), "cuts": 8},
    "large": {"fault": (100, 100, 20), "islands": (50, 50, 12), "nnc": (60, 60, 30),
              "slice": (30, 30, 10), "cuts": 12},
}


class InvariantError(AssertionError):
    """A grid violated a structural invariant."""


# --------------------------------------------------------------- invariants


def check_grid(g, *, name: str, expect_volume: float = None,
               tol: float = 1e-9) -> Dict[str, float]:
    """Assert the structural invariants a valid grid must satisfy.

    Returns a few summary numbers for the report. Raises `InvariantError` with
    the offending entity named, so a large-deck failure is diagnosable.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise InvariantError(f"{name}: {message}")

    cells, faces = g.cells, g.faces
    require(faces.neighbors.shape == (faces.num, 2), "neighbors is not (nf, 2)")
    require(int(cells.facePos[-1]) == cells.faces.shape[0],
            "cells.facePos does not close on cells.faces")
    require(int(faces.nodePos[-1]) == faces.nodes.shape[0],
            "faces.nodePos does not close on faces.nodes")
    require(faces.neighbors.min() >= -1 and faces.neighbors.max() < cells.num,
            "a face names a cell outside the grid")
    require(cells.faces.min() >= 0 and cells.faces.max() < faces.num,
            "a cell names a face outside the grid")
    require(faces.nodes.min() >= 0 and faces.nodes.max() < g.nodes.num,
            "a face names a node outside the grid")

    orphans = np.flatnonzero(np.all(faces.neighbors == -1, axis=1))
    require(orphans.size == 0, f"{orphans.size} orphan faces, first {orphans[:3]}")

    # Every (cell, face) in the neighbor table must appear in cells.faces.
    # Encoded as one integer per half-face and checked by binary search, so
    # this stays affordable on the grids where it matters most.
    cell_no = g.cell_no()
    listed = np.sort(cell_no.astype(np.int64) * faces.num
                     + np.asarray(cells.faces, dtype=np.int64))
    for side in (0, 1):
        real = np.flatnonzero(faces.neighbors[:, side] >= 0)
        needed = faces.neighbors[real, side].astype(np.int64) * faces.num + real
        where = np.searchsorted(listed, needed)
        found = (where < listed.size) & (listed[np.minimum(where, listed.size - 1)] == needed)
        require(bool(found.all()),
                f"{int((~found).sum())} half-faces missing from cells.faces "
                f"(side {side}, first face {real[~found][:3]})")

    summary: Dict[str, float] = {"cells": cells.num, "faces": faces.num,
                                 "nodes": g.nodes.num}
    if cells.volumes is not None:
        require(bool(np.all(np.isfinite(cells.volumes))), "non-finite cell volume")
        require(bool(np.all(cells.volumes > 0)),
                f"{int((cells.volumes <= 0).sum())} cells of non-positive volume")
        summary["volume"] = float(cells.volumes.sum())
        if expect_volume is not None:
            error = abs(summary["volume"] - expect_volume) / abs(expect_volume)
            require(error < tol,
                    f"volume {summary['volume']:.8e} != {expect_volume:.8e} "
                    f"(relative {error:.2e})")
    if faces.normals is not None and cells.volumes is not None:
        residual = _closure_residual(g)
        summary["closure"] = residual
        scale = float(np.abs(faces.normals).max())
        require(residual < 1e-6 * max(scale, 1.0),
                f"cells are not closed: max |sum of outward normals| = {residual:.3e}")
    return summary


def _closure_residual(g) -> float:
    """max |sum of outward face normals| over cells; zero for closed cells."""
    cell_no = g.cell_no()
    faces = np.asarray(g.cells.faces)
    sign = np.where(g.faces.neighbors[faces, 0] == cell_no, 1.0, -1.0)
    residual = np.zeros((g.cells.num, g.faces.normals.shape[1]))
    np.add.at(residual, cell_no, sign[:, None] * g.faces.normals[faces])
    return float(np.abs(residual).max())


# ---------------------------------------------------------------- scenarios


def stress_faulted_deck(dims) -> Dict[str, float]:
    """A sloping-fault corner-point deck, processed and given geometry."""
    deck = one_sloping_fault(list(dims), drop=5.0)
    grid = process_grdecl(deck)
    grid = compute_geometry(grid, max_block_size=20_000)
    summary = check_grid(grid, name=f"faulted deck {tuple(dims)}")
    faults = int(np.sum(np.any(grid.faces.neighbors < 0, axis=1)))
    summary["boundary faces"] = faults
    # the fault must actually have produced non-conforming geometry
    if grid.faces.num <= 3 * grid.cells.num:
        raise InvariantError("faulted deck produced a suspiciously conforming grid")
    return summary


def stress_disconnected_regions(dims) -> Dict[str, float]:
    """An ACTNUM cut into a lattice of islands by inactive planes.

    Every third i- and j-plane is switched off, leaving `ceil(nx/3)*ceil(ny/3)`
    columns that touch nowhere - a predictable component count to assert
    against. Both routes to the components are exercised and must agree: the
    automatic split inside `process_grdecl`, and `split_disconnected_grid`
    applied afterwards to the whole grid.
    """
    nx, ny, nz = dims
    i, j, _ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    active = (i % 3 != 2) & (j % 3 != 2)
    deck = one_sloping_fault([nx, ny, nz], drop=0.0)
    deck["ACTNUM"] = active.ravel(order="F").astype(np.int32)
    expected = -(-nx // 3) * -(-ny // 3)

    whole = compute_geometry(process_grdecl(deck, split_disconnected=False),
                             max_block_size=20_000)
    total = check_grid(whole, name=f"islands {tuple(dims)}")
    if whole.cells.num != int(active.sum()):
        raise InvariantError(
            f"grid has {whole.cells.num} cells, ACTNUM marks {int(active.sum())}")

    parts = split_disconnected_grid(whole)
    if len(parts) != expected:
        raise InvariantError(
            f"split found {len(parts)} components, the lattice has {expected}")
    if sum(part.cells.num for part in parts) != whole.cells.num:
        raise InvariantError("components do not account for every cell")
    if any(a.cells.num < b.cells.num for a, b in zip(parts, parts[1:])):
        raise InvariantError("components are not ordered largest first")

    auto = process_grdecl(deck)                      # the built-in split
    auto = auto if isinstance(auto, list) else [auto]
    if sorted(p.cells.num for p in auto) != sorted(p.cells.num for p in parts):
        raise InvariantError(
            "processGRDECL's own split disagrees with split_disconnected_grid")

    volume = 0.0
    for index, part in enumerate(parts[:50]):        # spot-check the largest
        volume += check_grid(compute_geometry(part),
                             name=f"island {index}")["volume"]
    total["components"] = len(parts)
    total["largest component"] = parts[0].cells.num
    return total


def stress_dense_nnc(dims) -> Dict[str, float]:
    """A dense NNC network over a partly inactive deck.

    Beyond sheer volume this pins the two filters MRST's `processNNC`
    documents: rows naming an inactive cell are dropped, and under
    `exclude_existing` so are rows that merely restate a connection the grid
    already has as a face. Self-connections are *not* filtered - MRST leaves
    those to the caller, and so do we.
    """
    nx, ny, nz = dims
    rng = np.random.default_rng(11)
    deck = one_sloping_fault([nx, ny, nz], drop=0.0)
    active = rng.random(nx * ny * nz) > 0.1          # ~10 percent inactive
    deck["ACTNUM"] = active.astype(np.int32)
    grid = process_grdecl(deck, split_disconnected=False)
    grid = compute_geometry(grid, max_block_size=20_000)
    check_grid(grid, name=f"nnc base {tuple(dims)}")

    logical = np.asarray(grid.cells.indexMap, dtype=np.int64)
    to_ijk = lambda flat: np.stack(
        np.unravel_index(flat, (nx, ny, nz), order="F"), axis=1) + 1
    inactive = np.flatnonzero(~active)

    count = 4 * nx * ny                              # dense relative to the grid
    left = rng.integers(0, logical.size, count)
    right = (left + rng.integers(1, logical.size, count)) % logical.size
    genuine = np.column_stack([to_ijk(logical[left]), to_ijk(logical[right]),
                               rng.random(count) + 0.1]).astype(float)

    # rows naming an inactive cell, which must be dropped
    dead = np.column_stack([
        to_ijk(logical[rng.integers(0, logical.size, inactive.size)]),
        to_ijk(inactive), np.full(inactive.size, 9.0)]).astype(float)

    # rows restating an existing internal face, dropped under exclude_existing
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))[:count]
    pairs = grid.faces.neighbors[internal]
    duplicate = np.column_stack([to_ijk(logical[pairs[:, 0]]),
                                 to_ijk(logical[pairs[:, 1]]),
                                 np.full(internal.size, 7.0)]).astype(float)

    resolved = process_nnc(grid, np.vstack([genuine, dead, duplicate]),
                           exclude_existing=True)
    if resolved is None:
        raise InvariantError("dense NNC network resolved to nothing")
    cells = np.asarray(resolved["cells"], dtype=np.int64)
    values = np.asarray(resolved["trans"], dtype=float)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise InvariantError(f"NNC cells has shape {cells.shape}, expected (m, 2)")
    if cells.size and (cells.min() < 0 or cells.max() >= grid.cells.num):
        raise InvariantError("an NNC names a cell outside the grid")
    if values.shape[0] != cells.shape[0]:
        raise InvariantError("NNC transmissibilities do not match the pairs")
    if not np.all(np.isfinite(values)):
        raise InvariantError("non-finite NNC transmissibility")
    if np.any(values == 9.0):
        raise InvariantError("an NNC naming an inactive cell survived")
    if np.any(values == 7.0):
        raise InvariantError("an NNC restating an existing face survived "
                             "exclude_existing")
    existing = {tuple(sorted(pair)) for pair in grid.faces.neighbors[internal]}
    if any(tuple(sorted(pair)) in existing for pair in cells):
        raise InvariantError("a resolved NNC duplicates a face of the grid")
    return {"cells": grid.cells.num, "faces": grid.faces.num,
            "requested NNCs": int(count + dead.shape[0] + duplicate.shape[0]),
            "resolved NNCs": cells.shape[0]}


def stress_repeated_slicing(dims, cuts: int) -> Dict[str, float]:
    """Slice the same grid over and over, mapping back to the original."""
    nx, ny, nz = dims
    grid = compute_geometry(cart_grid([nx, ny, nz], [float(nx), float(ny), float(nz)]))
    expected = float(nx * ny * nz)
    check_grid(grid, name="slice base", expect_volume=expected)

    mapping = None
    for index in range(cuts):
        # mid-cell, so every cut genuinely splits: a plane landing on a node
        # plane is a snapped cut and correctly splits nothing
        offset = 0.5 + round(index * (nx - 1) / max(cuts - 1, 1))
        grid, mapping = slice_grid(
            grid,
            np.array([[offset, -1.0, 0.0], [offset, ny + 1.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0]),
            return_mapping=True, gix_input=mapping,
        )
        grid = compute_geometry(grid, max_block_size=20_000)
        check_grid(grid, name=f"after cut {index + 1}", expect_volume=expected)
        parents = np.asarray(mapping.parent.cells)
        if parents.shape[0] != grid.cells.num:
            raise InvariantError("cell parent map does not cover the grid")
        if parents.min() < 0 or parents.max() >= nx * ny * nz:
            raise InvariantError("a cell parent points outside the original grid")
    return {"cells": grid.cells.num, "faces": grid.faces.num,
            "volume": float(grid.cells.volumes.sum()), "cuts": cuts}


# ------------------------------------------------------------------- driver


def scenarios(size: str) -> List[Tuple[str, Callable[[], Dict[str, float]]]]:
    cfg = SIZES[size]
    return [
        ("faulted corner-point deck", lambda: stress_faulted_deck(cfg["fault"])),
        ("many disconnected regions", lambda: stress_disconnected_regions(cfg["islands"])),
        ("dense NNC network", lambda: stress_dense_nnc(cfg["nnc"])),
        ("repeated slicing", lambda: stress_repeated_slicing(cfg["slice"], cfg["cuts"])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(SIZES), default="medium")
    parser.add_argument("--memory", action="store_true",
                        help="also report peak allocation, in a second pass")
    args = parser.parse_args()

    print(f"gridprocessing stress run, size = {args.size}")
    failed = 0
    for name, run in scenarios(args.size):
        # timed untraced: tracemalloc inflates allocation-heavy code several
        # fold, so measuring both at once would report neither honestly
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary = run()
            status, detail = "ok  ", _format(summary)
        except InvariantError as error:
            status, detail, failed = "FAIL", str(error), failed + 1
        elapsed = time.perf_counter() - started
        peak = None
        if args.memory and status == "ok  ":
            tracemalloc.start()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        print(f"  {status} {name}")
        memory = f"  peak {peak / 1e6:8.1f} MB" if peak is not None else ""
        print(f"       {elapsed:6.2f}s{memory}")
        print(f"       {detail}")
    print("all scenarios passed" if not failed else f"{failed} scenario(s) failed")
    return int(failed > 0)


def _format(summary: Dict[str, float]) -> str:
    parts = []
    for key, value in summary.items():
        if isinstance(value, float) and not float(value).is_integer():
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={int(value)}")
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
