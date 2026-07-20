"""Validate MPSA-W interaction-region assembly surface."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics import (
    mpsaw_assemble,
    mpsaw_interaction_regions,
)


def _rigid_modes(g):
    dim = g.griddim
    x = g.cells.centroids[:, :dim]
    modes = []
    for k in range(dim):
        u = np.zeros((g.cells.num, dim))
        u[:, k] = 1.0
        modes.append(u.reshape(-1))
    if dim == 2:
        modes.append(np.column_stack([-x[:, 1], x[:, 0]]).reshape(-1))
    else:
        modes.extend([
            np.column_stack([np.zeros(g.cells.num), -x[:, 2], x[:, 1]]).reshape(-1),
            np.column_stack([x[:, 2], np.zeros(g.cells.num), -x[:, 0]]).reshape(-1),
            np.column_stack([-x[:, 1], x[:, 0], np.zeros(g.cells.num)]).reshape(-1),
        ])
    return modes


def _check_case(name, g):
    regions = mpsaw_interaction_regions(g)
    assert len(regions) == g.nodes.num
    cell_node_count = 0
    for c in range(g.cells.num):
        faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
        nodes = np.unique(np.concatenate([
            g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            for f in faces
        ]))
        cell_node_count += nodes.size
    covered = sum(r.cells.size for r in regions)
    assert covered == cell_node_count
    assert all(r.continuity_points.shape[1] == g.griddim for r in regions)

    asm = mpsaw_assemble(g, young=1.0e9, poisson=0.25)
    assert asm.B.shape == (g.cells.num * g.griddim, g.cells.num * g.griddim)
    assert asm.div.shape == (g.cells.num, g.cells.num * g.griddim)
    assert asm.strain_op.shape[0] == g.cells.num * (6 if g.griddim == 3 else 3)
    assert asm.stress_op.shape == asm.strain_op.shape
    scale = max(float(np.max(np.abs(asm.B.data))) if asm.B.nnz else 0.0, 1.0)
    for mode in _rigid_modes(g):
        assert np.linalg.norm(asm.B @ mode, ord=np.inf) < 1e-7 * scale
    print(
        f"  ok {name}: regions={len(regions)}, "
        f"dofs={asm.B.shape[0]}, nnz={asm.B.nnz}"
    )


def main():
    print("Phase 105: MPSA-W interaction-region assembly")
    g2 = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    _check_case("2D Cartesian", g2)
    g3 = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    _check_case("3D Cartesian", g3)
    print("Phase 105 validation passed: MPSA-W assembly invariants hold.")


if __name__ == "__main__":
    main()
