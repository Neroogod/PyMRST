"""Validate VEM mechanics on non-Cartesian polygonal/polyhedral grids."""

import numpy as np

from pymrst.gridprocessing import (
    compute_geometry,
    pebi,
    tetrahedral_grid,
    triangle_grid,
)
from pymrst.modules.vemmech import (
    ElasticBC,
    calculate_stress_vem,
    create_augmented_grid,
    enu2c,
    enu2c_2d,
    vem_assemble,
    vem_lin_elast,
)


def _elastic_tensor(g):
    e = np.full(g.cells.num, 1.0e9)
    nu = np.full(g.cells.num, 0.25)
    return enu2c_2d(e, nu) if g.griddim == 2 else enu2c(e, nu)


def _rigid_modes(g):
    dim = g.griddim
    coords = g.nodes.coords[:, :dim]
    modes = []
    for ax in range(dim):
        u = np.zeros((g.nodes.num, dim))
        u[:, ax] = 1.0
        modes.append(u.reshape(-1))
    if dim == 2:
        u = np.column_stack([-coords[:, 1], coords[:, 0]])
        modes.append(u.reshape(-1))
    else:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        modes.extend([
            np.column_stack([np.zeros_like(x), -z, y]).reshape(-1),
            np.column_stack([z, np.zeros_like(x), -x]).reshape(-1),
            np.column_stack([-y, x, np.zeros_like(x)]).reshape(-1),
        ])
    return modes


def _boundary_solve(g, C):
    dim = g.griddim
    coords = g.nodes.coords[:, :dim]
    xmin = np.min(coords[:, 0])
    xmax = np.max(coords[:, 0])
    disp_nodes = np.flatnonzero(np.isclose(coords[:, 0], xmin))
    disp_mask = np.ones((disp_nodes.size, dim), dtype=bool)
    disp_vals = np.zeros((disp_nodes.size, dim))

    face_centroids = g.faces.centroids[:, :dim]
    boundary = np.any(g.faces.neighbors == -1, axis=1)
    xmax_face = np.max(face_centroids[boundary, 0])
    bfaces = np.flatnonzero(boundary & np.isclose(face_centroids[:, 0], xmax_face))
    force = np.zeros((bfaces.size, dim))
    force[:, 0] = 1.0e5
    bc = ElasticBC(
        disp_nodes=disp_nodes,
        disp_mask=disp_mask,
        disp_vals=disp_vals,
        force_faces=bfaces,
        force=force,
    )
    zero_load = lambda x: np.zeros_like(x)
    uu, extra = vem_lin_elast(g, C, bc, zero_load)
    stress, strain = calculate_stress_vem(g, uu, C=C)
    return uu, stress, strain, extra


def _check_grid(name, g):
    g = create_augmented_grid(compute_geometry(g))
    C = _elastic_tensor(g)
    S = vem_assemble(g, C)
    dense_skew = (S - S.T).tocoo()
    skew = 0.0 if dense_skew.nnz == 0 else float(np.max(np.abs(dense_skew.data)))
    assert skew < 1e-5, f"{name}: stiffness matrix is not symmetric"

    scale = max(float(np.max(np.abs(S.data))) if S.nnz else 0.0, 1.0)
    for mode in _rigid_modes(g):
        residual = S @ mode
        err = float(np.linalg.norm(residual, ord=np.inf))
        assert err < 5e-8 * scale, f"{name}: rigid mode residual {err:g}"

    uu, stress, strain, extra = _boundary_solve(g, C)
    assert np.all(np.isfinite(uu)), f"{name}: displacement has nonfinite values"
    assert np.all(np.isfinite(stress)), f"{name}: stress has nonfinite values"
    assert np.all(np.isfinite(strain)), f"{name}: strain has nonfinite values"
    assert extra["A"].shape[0] > 0, f"{name}: reduced system is empty"
    print(
        f"  ok {name}: cells={g.cells.num}, nodes={g.nodes.num}, "
        f"max|u|={np.max(np.abs(uu)):.3e}"
    )


def _triangle_case():
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 4), np.linspace(0.0, 1.0, 3))
    p = np.column_stack([x.ravel(), y.ravel()])
    return triangle_grid(p)


def _pebi_case():
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 5), np.linspace(0.0, 1.0, 4))
    p = np.column_stack([x.ravel(), y.ravel()])
    return pebi(compute_geometry(triangle_grid(p)))


def _tetra_case():
    p = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    t = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    return tetrahedral_grid(p, t)


def main():
    print("Phase 98: VEM arbitrary-grid mechanics validation")
    _check_grid("triangular 2D grid", _triangle_case())
    _check_grid("PEBI polygonal 2D grid", _pebi_case())
    _check_grid("tetrahedral 3D grid", _tetra_case())
    print("Phase 98 validation passed: non-Cartesian VEM slices are stable.")


if __name__ == "__main__":
    main()
