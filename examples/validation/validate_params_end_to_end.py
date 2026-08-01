"""End-to-end params comparison on an inactive corner-point grid."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.modules.incomp import incomp_tpfa, init_single_fluid
from pymrst.params import (
    add_source, add_well, fluxside, gravity_vector, pore_volume, pside,
    vertical_well,
)
from pymrst.params.rock import grdecl2rock
from pymrst.solvers import compute_trans, init_state
from pymrst.utils.units import BARSA, CENTI, POISE


HERE = Path(__file__).parent


def _flat(ref, key, dtype=float):
    return np.asarray(ref[key], dtype=dtype).ravel()


def _check(name, actual, expected, rtol=2e-10, atol=1e-14, exact=False):
    actual = np.asarray(actual).ravel()
    expected = np.asarray(expected).ravel()
    ok = np.array_equal(actual, expected) if exact else np.allclose(
        actual, expected, rtol=rtol, atol=atol
    )
    if not ok:
        raise AssertionError(
            f"{name}: shapes {actual.shape}/{expected.shape}, "
            f"max difference {np.max(np.abs(actual-expected)):.3e}"
        )
    print(f"  ok   {name}")


def _lexorder(points):
    points = np.asarray(points)
    return np.lexsort(tuple(points[:, axis] for axis in reversed(range(points.shape[1]))))


def main():
    path = HERE/"ref_params_end_to_end.mat"
    if not path.exists():
        raise SystemExit("Run dump_params_end_to_end_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)
    grdecl = {
        "cartDims": _flat(ref, "in_cartDims", np.int64),
        "COORD": _flat(ref, "in_COORD"), "ZCORN": _flat(ref, "in_ZCORN"),
        "ACTNUM": _flat(ref, "in_ACTNUM", np.int64),
        "PERMX": _flat(ref, "in_PERMX"), "PERMY": _flat(ref, "in_PERMY"),
        "PERMZ": _flat(ref, "in_PERMZ"), "PORO": _flat(ref, "in_PORO"),
        "NTG": _flat(ref, "in_NTG"),
    }
    grid = compute_geometry(process_grdecl(grdecl))
    rock = grdecl2rock(grdecl, grid)
    gvec = gravity_vector(3)

    target = np.array([0, 2, 6, 9])
    lookup = {global_id: active for active, global_id in enumerate(grid.cells.indexMap)}
    cells = np.array([lookup[value] for value in target])
    wells = add_well(
        None, grid, rock, cells, well_type="rate", val=1e-5, name="CP-I",
        well_dir="xyzx", radius=[.009, .010, .011, .012],
        skin=[0, .5, 1, 1.5], wi=[-1, 2e-12, -1, -1], compi=[1],
        status=True, gravity=gvec,
    )
    wells = vertical_well(
        wells, grid, rock, 1, 1, [], well_type="bhp", val=90*BARSA,
        name="CP-P", compi=[1], radius=.01, gravity=gvec,
    )
    bc = pside(None, grid, "xmin", 120*BARSA, sat=1)
    bc = fluxside(bc, grid, "xmax", -2e-6, sat=1)
    source_cell = lookup[5]
    src = add_source(None, [source_cell], 1e-6, sat=1)
    ht = compute_trans(grid, rock)
    pv = pore_volume(grid, rock)
    fluid = init_single_fluid(1*CENTI*POISE, 1000.0)
    state = incomp_tpfa(
        init_state(grid, wells, 100*BARSA), grid, ht, fluid,
        bc=bc, src=src, wells=wells, gravity=gvec,
    )

    _check("active index map", grid.cells.indexMap, _flat(ref,"indexMap")-1, exact=True)
    _check("corner-point centroids", grid.cells.centroids, ref["cellCentroids"])
    _check("corner-point volumes", grid.cells.volumes, _flat(ref,"cellVolumes"))
    _check("active rock permeability", rock.perm, ref["rockPerm"])
    _check("active porosity", rock.poro, _flat(ref,"rockPoro"))
    _check("active NTG", rock.ntg, _flat(ref,"rockNtg"))
    for index, well in enumerate(wells, 1):
        _check(f"well {index} cells", well.cells, _flat(ref,f"w{index}cells")-1, exact=True)
        _check(f"well {index} WI", well.WI, _flat(ref,f"w{index}WI"), rtol=3e-10)
        _check(f"well {index} dZ", well.dZ, _flat(ref,f"w{index}dZ"))
        _check(f"well {index} rR", well.rR, _flat(ref,f"w{index}rR"), rtol=3e-10)
    py_bc_centroids = grid.faces.centroids[bc.face]
    ml_bc_centroids = np.atleast_2d(ref["bcCentroids"])
    py_order, ml_order = _lexorder(py_bc_centroids), _lexorder(ml_bc_centroids)
    _check("boundary centroids", py_bc_centroids[py_order],
           ml_bc_centroids[ml_order], atol=2e-12)
    _check("boundary values", bc.value[py_order],
           _flat(ref,"bcValue")[ml_order])
    py_sat = np.asarray(bc.sat)[py_order]
    ml_sat = np.asarray(ref["bcSat"]).reshape(-1, 1)[ml_order]
    _check("boundary saturation", py_sat, ml_sat)
    _check("source cell", src.cell, _flat(ref,"srcCell")-1, exact=True)
    _check("source rate", src.rate, _flat(ref,"srcRate"))
    _check("source saturation", src.sat, ref["srcSat"])
    ht_sum = np.bincount(grid.cell_no(), weights=ht, minlength=grid.cells.num)
    _check("per-cell half-transmissibility", ht_sum, _flat(ref,"hTSum"), rtol=5e-9)
    _check("pore volume", pv, _flat(ref,"pv"))
    _check("cell pressure", state.pressure, _flat(ref,"pressure"), rtol=2e-8, atol=1e-5)
    half_faces = grid.cells.faces
    cell_no = grid.cell_no()
    signs = np.where(grid.faces.neighbors[half_faces, 0] == cell_no, 1.0, -1.0)
    divergence = np.bincount(
        cell_no, weights=signs*state.flux[half_faces], minlength=grid.cells.num
    )
    _check("cell flux divergence", divergence, _flat(ref,"cellDivergence"),
           rtol=2e-8, atol=1e-12)
    _check("injector connection flux", state.wellSol[0].flux, _flat(ref,"wflux1"), rtol=2e-8)
    _check("producer connection flux", state.wellSol[1].flux, _flat(ref,"wflux2"), rtol=2e-8)
    _check("injector BHP", [state.wellSol[0].pressure], [ref["bhp1"]], rtol=2e-8)
    _check("producer BHP", [state.wellSol[1].pressure], [ref["bhp2"]], rtol=2e-8)
    print("\nEnd-to-end inactive corner-point params workflow matches MATLAB MRST.")


if __name__ == "__main__":
    main()
