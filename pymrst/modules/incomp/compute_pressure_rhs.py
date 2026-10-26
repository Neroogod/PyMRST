"""MRST ``incomp/computePressureRHS.m``: the right-hand-side pieces of the
incompressible pressure system.

Returns, as MRST does, `ff` (per half-face: minus the Dirichlet pressure on
pressure faces), `gg` (per cell: the source rates), `hh` (per face: minus the
Neumann flux), `gp` (per half-face: the gravity potential `omega*(x_f -
x_c).g`, or `G.grav_pressure(G, omega)` when the grid carries one), `dF`
(the Dirichlet-face mask) and `dC` (the Dirichlet values, in face order).
Indices are 0-based.
"""

from typing import Any, Tuple

import numpy as np


def compute_pressure_rhs(g: Any, omega: np.ndarray, bc: Any, src: Any) -> Tuple[np.ndarray, ...]:
    """MRST `[ff, gg, hh, gp, dF, dC] = computePressureRHS(g, omega, bc, src)`."""
    grav_pressure_fn = getattr(g, "grav_pressure", None)
    gp = grav_pressure_fn(g, omega) if grav_pressure_fn is not None else _grav_pressure(g, omega)
    ff = np.zeros(gp.shape)
    gg = np.zeros(g.cells.num)
    hh = np.zeros(g.faces.num)
    cf = _half_faces(g)
    if src is not None and np.size(src.cell):
        cells = np.asarray(src.cell, dtype=np.int64).ravel()
        assert cells.max() < g.cells.num and cells.min() >= 0, (
            "Source terms refer to cell not existant in grid.")
        ss = np.bincount(cells, np.asarray(src.rate, dtype=float).ravel(), minlength=g.cells.num)
        ii = np.bincount(cells, minlength=g.cells.num) > 0
        gg[ii] = gg[ii] + ss[ii]
    dF = np.zeros(g.faces.num, dtype=bool)
    dC = np.zeros(0)
    if bc is not None and np.size(bc.face):
        faces = np.asarray(bc.face, dtype=np.int64).ravel()
        values = np.asarray(bc.value, dtype=float).ravel()
        types = np.array([str(t).lower() for t in np.atleast_1d(bc.type)])
        types = np.resize(types, faces.size)
        assert faces.max() < g.faces.num and faces.min() >= 0, (
            "Boundary condition refer to face not existant in grid.")
        assert np.all(np.bincount(faces, minlength=g.faces.num) <= 1), (
            "There are repeated faces in boundary condition.")
        is_press = types == "pressure"
        face = faces[is_press]
        dC = values[is_press]
        position = np.zeros(g.faces.num, dtype=np.int64)
        position[face] = np.arange(face.size)
        dF[face] = True
        i = dF[cf]
        ff[i] = -dC[position[cf[i]]]
        dC = dC[position[dF]]
        is_flux = types == "flux"
        hh[faces[is_flux]] = -values[is_flux]
    return ff, gg, hh, gp, dF, dC


def _half_faces(g: Any) -> np.ndarray:
    faces = np.asarray(g.cells.faces, dtype=np.int64)
    return faces[:, 0] if faces.ndim > 1 else faces


def _grav_pressure(g: Any, omega: np.ndarray) -> np.ndarray:
    from pymrst.params.gravity.gravity import gravity
    g_vec = gravity()
    cf = _half_faces(g)
    if np.linalg.norm(g_vec[:g.griddim]) > 0:
        dim = g.griddim
        assert 1 <= dim < 4
        cellno = np.repeat(np.arange(g.cells.num), np.diff(np.asarray(g.cells.facePos)))
        cvec = (np.asarray(g.faces.centroids)[cf, :dim]
                - np.asarray(g.cells.centroids)[cellno, :dim])
        return np.asarray(omega)[cellno]*(cvec @ g_vec[:dim])
    return np.zeros(cf.size)


computePressureRHS = compute_pressure_rhs
__all__ = ["compute_pressure_rhs", "computePressureRHS"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.wells_and_bc import add_bc, add_source

    G = compute_geometry(cart_grid([3, 1, 2]))

    def test_dirichlet_neumann_and_sources():
        bc = add_bc(None, np.array([0]), "pressure", 5.0)
        bc = add_bc(bc, np.array([3]), "flux", 2.0)
        src = add_source(None, np.array([1, 4]), np.array([3.0, 3.0]))
        with isolated_gravity("off"):
            ff, gg, hh, gp, dF, dC = compute_pressure_rhs(G, np.ones(G.cells.num), bc, src)
        assert dF[0] and dF.sum() == 1 and np.array_equal(dC, [5.0])
        cf = _half_faces(G)
        assert np.all(ff[cf == 0] == -5.0) and np.all(ff[cf != 0] == 0)
        assert hh[3] == -2.0 and np.count_nonzero(hh) == 1
        assert np.array_equal(gg, [0, 3, 0, 0, 3, 0]) and not np.any(gp)
        print("ok test_dirichlet_neumann_and_sources")

    def test_gravity_potential_per_half_face():
        with isolated_gravity("reset", "on"):
            gp = compute_pressure_rhs(G, 2*np.ones(G.cells.num), None, None)[3]
        cf = _half_faces(G)
        cellno = np.repeat(np.arange(G.cells.num), np.diff(G.cells.facePos))
        dz = G.faces.centroids[cf, 2] - G.cells.centroids[cellno, 2]
        assert np.allclose(gp, 2*9.80665*dz)
        print("ok test_gravity_potential_per_half_face")

    test_dirichlet_neumann_and_sources()
    test_gravity_potential_per_half_face()
    print("\nAll tests passed!")
