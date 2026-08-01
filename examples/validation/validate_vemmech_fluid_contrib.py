"""Validate addFluidContribMechVEM against an MRST reference dump.

This is the poroelastic boundary traction: the load a pressure boundary
condition puts on the mechanics. It is compared node-for-node, not by
aggregate, because the whole content of the term is *how* the face load is
distributed over its nodes - and because a sign error would survive any
symmetric aggregate check.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.utils.add_fluid_contrib_mech_vem import (
    add_fluid_contrib_mech_vem,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

REF = Path(__file__).with_name("ref_vemmech_fluid_contrib.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _close(actual, expected, tol=1e-12):
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    scale = max(float(np.abs(expected).max()), 1.0)
    error = float(np.abs(actual - expected).max()) / scale
    return error < tol, f"max relative |d| = {error:.3e}"


def _grid(dims, size, coords):
    grid = create_augmented_grid(compute_geometry(cart_grid(dims, size)))
    assert np.allclose(grid.nodes.coords, coords), "grid node ordering differs"
    return grid


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_vemmech_fluid_contrib_reference.m")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    print("3D, pressure on the top boundary of a unit cube:")
    grid = _grid([4, 4, 4], [1.0, 1.0, 1.0], ref["node_coords3"])
    rock = SimpleNamespace(alpha=np.ones(grid.cells.num))
    isdir = np.zeros(3 * grid.nodes.num, dtype=bool)
    top = np.atleast_1d(ref["top_faces0"]).astype(np.int64)

    actual = add_fluid_contrib_mech_vem(
        grid, SimpleNamespace(face=top, value=1.0), rock, isdir)
    failed += _check("nodal force vector", *_close(actual, ref["fbc3_top"]))
    # a sign flip would pass any symmetric check, so pin the direction too
    resultant = actual.reshape(-1, 3).sum(axis=0)
    expected_resultant = np.asarray(ref["fbc3_top"], dtype=float).reshape(-1, 3).sum(axis=0)
    failed += _check("resultant direction and magnitude",
                     *_close(resultant, expected_resultant))

    print("3D, every outer face pressurized:")
    boundary = np.atleast_1d(ref["all_boundary0"]).astype(np.int64)
    actual = add_fluid_contrib_mech_vem(
        grid, SimpleNamespace(face=boundary, value=1.0), rock, isdir)
    failed += _check("nodal force vector", *_close(actual, ref["fbc3_all"]))
    failed += _check(
        "closed pressurized boundary is self-equilibrated",
        bool(np.allclose(actual.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-12)),
    )

    print("3D, Biot alpha of 0.5:")
    half = SimpleNamespace(alpha=0.5 * np.ones(grid.cells.num))
    actual = add_fluid_contrib_mech_vem(
        grid, SimpleNamespace(face=top, value=1.0), half, isdir)
    failed += _check("nodal force vector", *_close(actual, ref["fbc3_half"]))

    print("3D, with nine displacement dofs eliminated:")
    isdir_some = np.zeros(3 * grid.nodes.num, dtype=bool)
    isdir_some[:9] = True
    actual = add_fluid_contrib_mech_vem(
        grid, SimpleNamespace(face=top, value=1.0), rock, isdir_some)
    failed += _check("reduced force vector", *_close(actual, ref["fbc3_dir"]))

    print("3D, no boundary condition:")
    actual = add_fluid_contrib_mech_vem(grid, None, rock, isdir)
    failed += _check("zero vector of the right length",
                     *_close(actual, ref["fbc3_empty"]))

    print("2D, pressure on the right boundary of a unit square:")
    grid2 = _grid([4, 4], [1.0, 1.0], ref["node_coords2"])
    rock2 = SimpleNamespace(alpha=np.ones(grid2.cells.num))
    isdir2 = np.zeros(2 * grid2.nodes.num, dtype=bool)
    right = np.atleast_1d(ref["right_faces0"]).astype(np.int64)
    actual = add_fluid_contrib_mech_vem(
        grid2, SimpleNamespace(face=right, value=1.0), rock2, isdir2)
    failed += _check("nodal force vector", *_close(actual, ref["fbc2_right"]))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
