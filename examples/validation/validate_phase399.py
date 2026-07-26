"""Phase 399: MRST-layout mimetic inner-product assembly."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.mimetic import computeMimeticIP, compute_mimetic_ip
from pymrst.params.rock import make_rock


def _grid():
    g = cart_grid([6, 5], [6.0, 5.0])
    coordinates = np.asarray(g.nodes.coords, dtype=float)
    x, y = coordinates.T.copy()
    coordinates[:, 0] += (
        0.22*np.sin(np.pi*x/6.0)*np.sin(np.pi*y/5.0)
    )
    coordinates[:, 1] += (
        0.13*np.sin(2.0*np.pi*x/6.0)*np.sin(np.pi*y/5.0)
    )
    g.nodes.coords = coordinates
    return compute_geometry(g)


def main():
    g = _grid()
    tensor = np.tile([2.0, 0.35, 0.8], (g.cells.num, 1))
    rock = make_rock(g, tensor, 0.25)
    system = computeMimeticIP(
        g, rock, Type="comp_hybrid", InnerProduct="ip_simple"
    )
    nhf = len(g.cells.faces)
    shape_ok = system.B.shape == (nhf, nhf) and system.BI.shape == (nhf, nhf)
    symmetry_error = max(
        float(np.max(np.abs((system.B - system.B.T).data))),
        float(np.max(np.abs((system.BI - system.BI.T).data))),
    )
    minimum_eigenvalue = min(
        np.linalg.eigvalsh(
            system.BI[
                g.cells.facePos[c]:g.cells.facePos[c + 1],
                g.cells.facePos[c]:g.cells.facePos[c + 1],
            ].toarray()
        ).min()
        for c in range(g.cells.num)
    )

    # Mimetic consistency: BI*(face-center minus cell-center)*grad = N*K*grad.
    gradient = np.array([1.7, -0.6])
    cellno = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    faces = np.asarray(g.cells.faces, dtype=int)
    neighbors = np.asarray(g.faces.neighbors)
    sign = 2.0*(neighbors[faces, 0] == cellno) - 1.0
    cvec = g.faces.centroids[faces] - g.cells.centroids[cellno]
    normals = g.faces.normals[faces]*sign[:, None]
    expected = np.empty(nhf)
    for cell in range(g.cells.num):
        ix = slice(g.cells.facePos[cell], g.cells.facePos[cell + 1])
        k = np.array([[2.0, 0.35], [0.35, 0.8]])
        expected[ix] = normals[ix] @ (k @ gradient)
    consistency = float(np.max(np.abs(
        system.BI @ (cvec @ gradient) - expected
    )))

    tpf_mixed = compute_mimetic_ip(
        g, rock, type="mixed", inner_product="ip_tpf"
    )
    tpf_hybrid = compute_mimetic_ip(
        g, rock, type="hybrid", inner_product="ip_tpf"
    )
    inverse_error = float(np.max(np.abs(
        (tpf_mixed.B @ tpf_hybrid.BI).toarray() - np.eye(nhf)
    )))
    alias_error = float(np.max(np.abs(
        system.BI.toarray()
        - compute_mimetic_ip(
            g, rock, type="hybrid", inner_product="ip_simple"
        ).BI.toarray()
    )))

    print("Phase 399: mimetic inner-product assembly")
    print(f"  block matrix dimensions: {shape_ok}")
    print(f"  symmetry error: {symmetry_error:.3e}")
    print(f"  minimum local BI eigenvalue: {minimum_eigenvalue:.3e}")
    print(f"  affine full-tensor consistency: {consistency:.3e}")
    print(f"  ip_tpf B/BI inverse parity: {inverse_error:.3e}")
    print(f"  snake/MRST entry-point parity: {alias_error:.3e}")

    failed = (
        not shape_ok
        or symmetry_error > 1e-12
        or minimum_eigenvalue <= 0
        or consistency > 1e-11
        or inverse_error > 1e-12
        or alias_error > 1e-14
    )
    print("\nPhase 399 passed." if not failed else "\nPhase 399 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
