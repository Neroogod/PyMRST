"""Phase 390: MPFA tensor assembly for general 3-D cell corners."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.modules.mpfa import computeMultiPointTrans, incompMPFA
from pymrst.modules.mpfa.incomp_mpfa import incomp_mpfa
from pymrst.modules.mpfa.private.compute_multi_point_trans_tensor_assembly import (
    computeMultiPointTransTensorAssembly,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc
from pymrst.solvers.state import init_res_sol


def _bipyramid():
    """Two square pyramids sharing a base; each apex touches four faces."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, -1.0],
    ])
    face_nodes = [
        [0, 3, 2, 1],
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
        [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5],
    ]
    nodes = np.array(sum(face_nodes, []), dtype=np.int64)
    node_pos = np.r_[0, np.cumsum([len(v) for v in face_nodes])]
    neighbors = np.array([[0, 1]] + [[0, -1]]*4 + [[1, -1]]*4)
    cell_faces = np.array([0, 1, 2, 3, 4, 0, 5, 6, 7, 8])
    g = Grid(
        Cells(2, np.array([0, 5, 10]), cell_faces, np.arange(2)),
        Faces(9, node_pos, nodes, neighbors),
        Nodes(6, coords),
        3,
        ["bipyramid"],
    )
    return compute_geometry(g)


def _linear(points):
    coefficients = np.array([3.0, 2.0, 1.0])[:points.shape[1]]
    return points @ coefficients + 5.0


def main():
    g = _bipyramid()
    perm = np.tile([2.0, 0.2, 0.1, 1.0, 0.15, 0.8], (2, 1))*1e-13
    rock = make_rock(g, perm, 0.2)

    legacy_rejected = False
    try:
        computeMultiPointTrans(g, rock)
    except AssertionError:
        legacy_rejected = True

    tensor = computeMultiPointTransTensorAssembly(g, rock)
    boundary = np.flatnonzero(np.any(g.faces.neighbors < 0, axis=1))
    bc = {int(f): float(_linear(g.faces.centroids[f:f + 1])[0])
          for f in boundary}
    result = incomp_mpfa(g, tensor, bc, return_result=True)
    exact = _linear(np.asarray(g.cells.centroids))
    patch_error = float(np.max(np.abs(result.pressure - exact)))
    cell_balance = np.asarray(tensor["e_div"] @ result.subface_flux).ravel()
    conservation_error = float(np.max(np.abs(cell_balance[:g.cells.num])))
    bc_struct = add_bc(
        None,
        boundary,
        "pressure",
        np.array([bc[int(f)] for f in boundary]),
    )
    public_state = incompMPFA(
        init_res_sol(g, 0.0),
        g,
        tensor,
        init_single_fluid(1.0, 0.0),
        bc=bc_struct,
        useTensorAssembly=True,
    )
    public_dispatch_error = float(
        np.max(np.abs(public_state.pressure - result.pressure))
    )

    # On nice corners, selecting tensor assembly must preserve the established
    # legacy solution.
    regular = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    regular_rock = make_rock(
        regular, np.tile([2.0, 0.4, 1.0], (regular.cells.num, 1))*1e-13, 0.2
    )
    regular_boundary = np.flatnonzero(
        np.any(regular.faces.neighbors < 0, axis=1)
    )
    regular_bc = {
        int(f): float(_linear(regular.faces.centroids[f:f + 1])[0])
        for f in regular_boundary
    }
    p_legacy = incomp_mpfa(
        regular, computeMultiPointTrans(regular, regular_rock), regular_bc
    )
    p_tensor = incomp_mpfa(
        regular,
        computeMultiPointTrans(
            regular, regular_rock, useTensorAssembly=True
        ),
        regular_bc,
    )
    nice_corner_error = float(np.max(np.abs(p_tensor - p_legacy)))

    print("Phase 390: general-corner MPFA tensor assembly")
    print(f"  legacy rejects four-face apex: {legacy_rejected}")
    print(f"  bipyramid linear patch error: {patch_error:.3e}")
    print(f"  bipyramid cell conservation: {conservation_error:.3e}")
    print(f"  public tensor-dispatch parity: {public_dispatch_error:.3e}")
    print(f"  nice-corner legacy/tensor parity: {nice_corner_error:.3e}")

    failed = (
        not legacy_rejected
        or patch_error > 1e-10
        or conservation_error > 1e-20
        or public_dispatch_error > 1e-10
        or nice_corner_error > 1e-10
    )
    print("\nPhase 390 passed." if not failed else "\nPhase 390 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
