"""MRST ``incomp/capPressureRHS.m``: the capillary contribution to the
half-face gravity term of the incompressible pressure system, for the
`'wetting'`, `'nonwetting'` or `'symmetric'` formulation."""

from typing import Any, Union

import numpy as np


def cap_pressure_rhs(g: Any, mob: np.ndarray, pc: np.ndarray, pc_form: str) -> Union[np.ndarray, int]:
    """MRST `cc = capPressureRHS(g, mob, pc, pc_form)`."""
    from pymrst.utils.gridtools.get_cell_no_faces import get_cell_no_faces
    from pymrst.utils.gridtools.get_neighbourship import get_neighbourship
    pc = np.asarray(pc, dtype=float).ravel()
    if not np.any(np.abs(pc) > 0):
        return 0
    N = get_neighbourship(g, topological=True, include_boundary=True)
    cellno, cf, _ = get_cell_no_faces(g)
    internal = ~np.any(N < 0, axis=1)
    pcf = np.zeros(N.shape)
    pcf[N >= 0] = pc[N[N >= 0]]
    pcf = pcf.sum(axis=1)
    pcf[internal] = pcf[internal]/2
    totmob = mob.sum(axis=1)
    dpc = pc[cellno] - pcf[cf]
    form = pc_form.lower()
    if form == "wetting":
        return mob[cellno, 1]/totmob[cellno]*dpc
    if form == "nonwetting":
        return -(mob[cellno, 0]/totmob[cellno])*dpc
    if form in ("symmetric", "symetric"):
        return (mob[cellno, 1] - mob[cellno, 0])/totmob[cellno]*dpc
    raise ValueError("no such capillary formulation implemented")


capPressureRHS = cap_pressure_rhs
__all__ = ["cap_pressure_rhs", "capPressureRHS"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    G = compute_geometry(cart_grid([2, 1]))

    def test_zero_pc_gives_zero():
        assert cap_pressure_rhs(G, np.ones((2, 2)), np.zeros(2), "nonwetting") == 0
        print("ok test_zero_pc_gives_zero")

    def test_nonwetting_form_on_the_shared_face():
        mob = np.array([[1.0, 1.0], [1.0, 3.0]])
        cc = cap_pressure_rhs(G, mob, np.array([2.0, 4.0]), "nonwetting")
        cellno = np.repeat(np.arange(2), np.diff(G.cells.facePos))
        shared = G.cells.faces[:, 0] == 1 if G.cells.faces.ndim > 1 else G.cells.faces == 1
        # Face average pc = 3: cell 0 gets -(1/2)*(2-3), cell 1 -(1/4)*(4-3).
        assert np.allclose(cc[shared & (cellno == 0)], 0.5)
        assert np.allclose(cc[shared & (cellno == 1)], -0.25)
        print("ok test_nonwetting_form_on_the_shared_face")

    def test_unknown_form_raises():
        try:
            cap_pressure_rhs(G, np.ones((2, 2)), np.ones(2), "other")
        except ValueError:
            print("ok test_unknown_form_raises")
            return
        raise AssertionError("expected ValueError")

    test_zero_pc_gives_zero()
    test_nonwetting_form_on_the_shared_face()
    test_unknown_form_raises()
    print("\nAll tests passed!")
