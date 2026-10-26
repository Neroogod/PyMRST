"""MRST ``agglom/utils/convertBC2Coarse.m``: fine boundary conditions moved
to the coarse faces. Flux conditions are summed per coarse face with
flux-weighted saturations; pressure conditions take the value of the first
fine face found on each coarse face (MRST's `unique` order). A mixture of
the two is not supported."""

from typing import Any

import numpy as np


def convert_bc2_coarse(bc: Any, G: Any, CG: Any, nsub: np.ndarray, sub: np.ndarray,
                       coarse_f: np.ndarray, sgn: np.ndarray) -> Any:
    """MRST `bc_cg = convertBC2Coarse(bc, G, CG, nsub, sub, coarse_f, sgn)`."""
    from pymrst.params.wells_and_bc import add_bc
    from pymrst.params.wells_and_bc.types import BoundaryCondition
    types = np.array([str(t).lower() for t in np.atleast_1d(bc.type)])
    faces = np.asarray(bc.face, dtype=np.int64).ravel()
    values = np.asarray(bc.value, dtype=float).ravel()
    types = np.resize(types, faces.size)
    is_flux, is_pressure = types == "flux", types == "pressure"
    if np.all(is_flux):
        fface = faces[is_flux]
        f2c = np.zeros(G.faces.num, dtype=np.int64)
        f2c[sub] = coarse_f
        cf = np.zeros(CG.faces.num, dtype=bool)
        cf[f2c[fface]] = True
        flux = np.zeros(G.faces.num)
        flux[fface] = values[is_flux]
        bc_sat = np.atleast_2d(np.asarray(bc.sat, dtype=float))
        sat = np.zeros((G.faces.num, bc_sat.shape[1]))
        sat[fface] = bc_sat[is_flux]
        weighted = np.column_stack([(sgn*flux[sub])[:, None]*sat[sub], flux[sub]])
        accum = np.column_stack([np.bincount(coarse_f, weighted[:, k], minlength=CG.faces.num)
                                 for k in range(weighted.shape[1])])
        with np.errstate(divide="ignore", invalid="ignore"):
            coarse_sat = accum[:, :-1]/accum[:, -1:]
        return add_bc(None, np.flatnonzero(cf), "flux", accum[cf, -1], sat=np.abs(coarse_sat[cf]))
    if np.all(is_pressure):
        cg = getattr(CG, "coarse_grid", CG)
        ff_cf = np.full(cg.parent.faces.num, np.nan)
        cfacesno = np.repeat(np.arange(CG.faces.num), np.diff(np.asarray(cg.conn_pos)))
        ff_cf[cg.fconn] = cfacesno
        cbcf = ff_cf[faces]
        cbf, ci = np.unique(cbcf, return_index=True)
        out = BoundaryCondition()
        out.face = cbf.astype(np.int64)
        out.type = [types[k] for k in ci]
        out.value = values[ci]
        return out
    raise ValueError("convertBC2Coarse: only flux boundary conditions supported")


convertBC2Coarse = convert_bc2_coarse
__all__ = ["convert_bc2_coarse", "convertBC2Coarse"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.agglom.utils.sign_of_fine_faces_on_coarse_faces import (
        sign_of_fine_faces_on_coarse_faces,
    )
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_ui import partition_ui
    from pymrst.modules.coarsegrid.sub_faces import sub_faces
    from pymrst.modules.coarsegrid.utils.coarse_grid_as_grid import coarse_grid_as_grid
    from pymrst.params.wells_and_bc import add_bc

    G = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
    CG = coarse_grid_as_grid(generate_coarse_grid(G, partition_ui(G, [2, 1])))
    nsub, sub = sub_faces(G, CG)
    sgn, coarse_f = sign_of_fine_faces_on_coarse_faces(G, CG, nsub, sub)
    west = np.flatnonzero(np.isclose(G.faces.centroids[:, 0], 0.0))

    def test_flux_conditions_sum_onto_the_coarse_face():
        bc = add_bc(None, west, "flux", 1.5, sat=np.array([1.0, 0.0]))
        out = convert_bc2_coarse(bc, G, CG, nsub, sub, coarse_f, sgn)
        assert out.face.size == 1 and np.isclose(out.value[0], 3.0)
        assert np.allclose(out.sat, [[1.0, 0.0]])
        print("ok test_flux_conditions_sum_onto_the_coarse_face")

    def test_pressure_conditions_take_one_value_per_coarse_face():
        bc = add_bc(None, west, "pressure", np.array([2.0, 4.0]))
        out = convert_bc2_coarse(bc, G, CG, nsub, sub, coarse_f, sgn)
        assert out.face.size == 1 and out.value[0] == 2.0 and out.type == ["pressure"]
        print("ok test_pressure_conditions_take_one_value_per_coarse_face")

    test_flux_conditions_sum_onto_the_coarse_face()
    test_pressure_conditions_take_one_value_per_coarse_face()
    print("\nAll tests passed!")
