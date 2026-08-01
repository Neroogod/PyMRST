"""Validate the vemmech test-grid fixtures against MRST.

The fixtures were previously checked only against physics - patch tests,
rigid-body modes, the analytic compaction column. Those catch a wrong answer
but not a fixture that sets up a subtly different *problem* from MRST's. This
pins the setups themselves.

Where a construction is deterministic the comparison is field by field. Where
it is not - `triangle` and `pebi` route through a Delaunay triangulation, and
scipy's need not order or even tie-break the same way as MATLAB's - the
comparison is on invariants that any correct construction must share.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.examples.utils.make_compaction_test import (
    CompactionOptions,
    make_compaction_test,
)
from pymrst.modules.vemmech.utils.complex_3d_grid import (
    ComplexGridOptions,
    complex_3d_grid,
)
from pymrst.modules.vemmech.utils.enu2lmu_3d import enu2lmu_3d
from pymrst.modules.vemmech.utils.linear_displacement import linear_displacement
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.utils.square_layers_test import (
    layer_index,
    square_layers_test,
)
from pymrst.modules.vemmech.utils.square_test import square_test
from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

REF = Path(__file__).with_name("ref_vemmech_fixtures.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _close(actual, expected, tol=1e-10):
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    scale = max(float(np.abs(expected).max()), 1.0)
    error = float(np.abs(actual - expected).max()) / scale
    return error < tol, f"max relative |d| = {error:.3e}"


def _by_coordinate(coords_a, values_a, coords_b, values_b):
    """Compare two node fields by position, so node ordering cannot matter."""
    order_a = np.lexsort(tuple(np.round(coords_a, 9).T[::-1]))
    order_b = np.lexsort(tuple(np.round(coords_b, 9).T[::-1]))
    if not np.allclose(coords_a[order_a], coords_b[order_b]):
        return False, "node coordinates differ"
    return _close(values_a[order_a], values_b[order_b])


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_vemmech_fixtures_reference.m")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    print("ENu2LMu_3D:")
    lam, mu = enu2lmu_3d(ref["enu_E"], ref["enu_nu"])
    failed += _check("lambda", *_close(lam, ref["enu_lambda"], tol=1e-14))
    failed += _check("mu", *_close(mu, ref["enu_mu"], tol=1e-14))

    print("linearDisplacement:")
    for dim, points_key, values_key, names_key in (
            (2, "ld_points2", "ld_values2", "ld_names2"),
            (3, "ld_points3", "ld_values3", "ld_names3")):
        points = np.atleast_2d(ref[points_key])
        fields, names = linear_displacement(dim)
        expected_names = [str(n).strip() for n in np.atleast_1d(ref[names_key])]
        failed += _check(f"{dim}D field names", names == expected_names,
                         "" if names == expected_names else f"{names}")
        stacked = np.vstack([field(points) for field in fields])
        failed += _check(f"{dim}D field values",
                         *_close(stacked, ref[values_key], tol=1e-14))

    print("squareGrid, cartgrid (deterministic, compared field by field):")
    grid = compute_geometry(square_grid([4, 3], [2.0, 1.5]))
    failed += _check(f"cell count ({grid.cells.num})",
                     grid.cells.num == int(ref["sg_cart_cells"]))
    failed += _check(f"face count ({grid.faces.num})",
                     grid.faces.num == int(ref["sg_cart_faces"]))
    failed += _check(f"node count ({grid.nodes.num})",
                     grid.nodes.num == int(ref["sg_cart_nodes"]))
    failed += _check("node coordinates",
                     *_close(grid.nodes.coords, ref["sg_cart_coords"]))
    failed += _check("cell volumes", *_close(grid.cells.volumes,
                                             ref["sg_cart_vols"]))
    grid3 = compute_geometry(square_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    failed += _check("3D cell count",
                     grid3.cells.num == int(ref["sg_cart3_cells"]))
    failed += _check("3D total volume",
                     *_close(grid3.cells.volumes.sum(), ref["sg_cart3_vol"]))

    print("squareGrid, Delaunay-derived types (invariants only):")
    for index, name in enumerate(np.atleast_1d(ref["sg_types"])):
        name = str(name).strip()
        built = compute_geometry(square_grid([4, 4], [1.0, 1.0], grid_type=name))
        expected_cells = int(np.atleast_1d(ref["sg_type_cells"])[index])
        expected_area = float(np.atleast_1d(ref["sg_type_area"])[index])
        failed += _check(f"{name}: covers the domain",
                         *_close(built.cells.volumes.sum(), expected_area,
                                 tol=1e-9))
        same = built.cells.num == expected_cells
        print(f"  {'ok  ' if same else 'note'} {name}: cell count "
              f"{built.cells.num} vs MRST {expected_cells}"
              f"{'' if same else '  (Delaunay tie-breaking differs)'}")

    print("squareTest on a Cartesian grid:")
    grid, sides, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                     grid_type="cartgrid", triangulate=False,
                                     E=1e6, nu=0.25)
    failed += _check(f"cell count ({grid.cells.num})",
                     grid.cells.num == int(ref["st_cells"]))
    failed += _check(f"node count ({grid.nodes.num})",
                     grid.nodes.num == int(ref["st_nodes"]))
    failed += _check(f"side count ({len(sides)})",
                     len(sides) == int(ref["st_num_bc"]))
    failed += _check("faces per side",
                     *_close([s.face.size for s in sides], ref["st_side_faces"]))
    failed += _check("nodes per side",
                     *_close([s.nodes.size for s in sides], ref["st_side_nodes"]))
    names = [c.name for c in cases]
    expected_names = [str(n).strip() for n in np.atleast_1d(ref["st_case_names"])]
    failed += _check(f"case count ({len(names)})",
                     len(names) == int(ref["st_num_cases"]))
    failed += _check("case names", names == expected_names,
                     "" if names == expected_names
                     else f"{set(names) ^ set(expected_names)}")

    coords = np.asarray(ref["st_coords"], dtype=float)
    for label, key in (("lin_disp_1", "st_patch_u"),
                       ("original_2D_test", "st_named_u")):
        case = next(c for c in cases if c.name == label)
        u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
        failed += _check(f"solved displacement, {label}",
                         *_by_coordinate(grid.nodes.coords[:, :2], u,
                                         coords[:, :2],
                                         np.atleast_2d(ref[key])))

    print("squareLayersTest:")
    grid, _, cases = square_layers_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                        E=[1e6, 5e6], nu=[0.25, 0.25],
                                        zlayers=[0.5, 0.5])
    failed += _check(f"cell count ({grid.cells.num})",
                     grid.cells.num == int(ref["sl_cells"]))
    failed += _check("cell centroids",
                     *_close(grid.cells.centroids, ref["sl_centroids"]))
    C11 = np.asarray(cases[0].C).reshape(grid.cells.num, -1)[:, 0]
    failed += _check("layered stiffness C11", *_close(C11, ref["sl_C11"]))
    index = layer_index(grid, [0.5, 0.5], [1.0, 1.0])
    failed += _check("two layers of equal size",
                     int((index == 0).sum()) == int((index == 1).sum()))

    print("makeCompactionTest:")
    grid = create_augmented_grid(compute_geometry(cart_grid([3, 3, 3],
                                                            [1.0, 1.0, 1.0])))
    el_bc, load = make_compaction_test(grid, CompactionOptions())
    failed += _check(f"pinned entries ({el_bc.disp_nodes.size})",
                     el_bc.disp_nodes.size == int(ref["mc_num_nodes"]))
    failed += _check("unique pinned nodes",
                     np.unique(el_bc.disp_nodes).size == int(ref["mc_unique"]))
    failed += _check("pinned dofs per direction",
                     *_close(el_bc.disp_mask.sum(axis=0), ref["mc_mask_sum"]))
    failed += _check(f"loaded faces ({el_bc.force_faces.size})",
                     el_bc.force_faces.size == int(ref["mc_force_n"]))
    if el_bc.force.size:
        failed += _check("overburden traction",
                         *_close(el_bc.force[0], ref["mc_force_val"]))
    failed += _check("gravity load",
                     *_close(load(grid.cells.centroids)[0], ref["mc_load_row"]))
    C = enu2c(np.full(grid.cells.num, 1e9), np.full(grid.cells.num, 0.25))
    u, _ = vem_lin_elast(grid, C, el_bc, load)
    failed += _check("solved compaction displacement",
                     *_by_coordinate(grid.nodes.coords, u,
                                     np.asarray(ref["mc_coords"], dtype=float),
                                     np.atleast_2d(ref["mc_u"])))

    print("complex3DGrid, box case:")
    for label, triangulate, prefix in (("plain", False, "cg_plain"),
                                       ("triangulated", True, "cg_tri")):
        built, _ = complex_3d_grid(
            ComplexGridOptions(cart_dims=[3, 3, 3], L=[1.0, 1.0, 1.0],
                               triangulate=triangulate), "box")
        built = compute_geometry(built)
        failed += _check(f"{label}: cell count ({built.cells.num})",
                         built.cells.num == int(ref[f"{prefix}_cells"]))
        failed += _check(f"{label}: face count ({built.faces.num})",
                         built.faces.num == int(ref[f"{prefix}_faces"]))
        failed += _check(f"{label}: total volume",
                         *_close(built.cells.volumes.sum(),
                                 ref[f"{prefix}_vol"]))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
