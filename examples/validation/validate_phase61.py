"""Phase 61: 2D VEM operators, validated against MATLAB MRST.

This checks the newly ported 2D branch of ``vemmech``:

- ``calculateQF`` node/edge integration weights
- ``VEM2D_div`` divergence operator
- 2D ``Enu2C`` plane-strain tensor
- ``VEM_assemble`` stiffness matrix
"""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.vem import (
    calculate_qf,
    create_augmented_grid,
    enu2c_2d,
    vem2d_div,
    vem_assemble,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p61_vem2d.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def _dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x, dtype=float)


def compare(name, py, ml, rtol=1e-10, atol=1e-12) -> int:
    py = _dense(py).ravel()
    ml = _dense(ml).ravel()
    scale = max(float(np.max(np.abs(ml))), 1.0)
    diff = float(np.max(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase61_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    nx, ny = int(ref["nx"]), int(ref["ny"])
    L = np.asarray(ref["L"], dtype=float).ravel()
    E = float(ref["E"])
    nu = float(ref["nu"])

    g = create_augmented_grid(compute_geometry(cart_grid([nx, ny], L)))
    C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))
    qf, qf_vol = calculate_qf(g)
    div = vem2d_div(g)
    S = vem_assemble(g, C)

    ml_nodes = np.asarray(ref["cell_nodes"], dtype=np.int64).ravel() - 1
    ml_pos = np.asarray(ref["cell_node_pos"], dtype=np.int64).ravel() - 1
    qf_mrst_order = np.zeros_like(_arr(ref, "qf"))
    qf_vol_mrst_order = np.zeros_like(_arr(ref, "qf_vol").ravel())
    for c in range(g.cells.num):
        py_sl = slice(g.cells.nodePos[c], g.cells.nodePos[c + 1])
        ml_sl = slice(ml_pos[c], ml_pos[c + 1])
        py_lookup = {node: i for i, node in enumerate(g.cells.nodes[py_sl])}
        order = [py_lookup[node] for node in ml_nodes[ml_sl]]
        qf_mrst_order[ml_sl] = qf[py_sl][order]
        qf_vol_mrst_order[ml_sl] = qf_vol[py_sl][order]

    failures = 0
    print("Phase 61: 2D VEM operators vs MATLAB MRST")
    py_sorted = np.concatenate([
        np.sort(g.cells.nodes[g.cells.nodePos[c]:g.cells.nodePos[c + 1]] + 1)
        for c in range(g.cells.num)
    ])
    ml_sorted = np.concatenate([
        np.sort(np.asarray(ref["cell_nodes"], dtype=np.int64).ravel()[
            ml_pos[c]:ml_pos[c + 1]
        ])
        for c in range(g.cells.num)
    ])
    failures += compare("cell node sets", py_sorted, ml_sorted, 0, 0)
    failures += compare("cell nodePos", g.cells.nodePos + 1,
                        ref["cell_node_pos"], 0, 0)
    failures += compare("calculateQF qf", qf_mrst_order, _arr(ref, "qf"),
                        rtol=1e-12)
    failures += compare("calculateQF qf_vol", qf_vol_mrst_order,
                        _arr(ref, "qf_vol"),
                        rtol=1e-12)
    failures += compare("VEM2D_div", div, ref["div"], rtol=1e-12)
    failures += compare("Enu2C 2D", C, _arr(ref, "C"), rtol=1e-14)
    failures += compare("VEM_assemble stiffness", S, ref["S"], rtol=1e-10)

    # Independent physics sanity check: divergence is exact on linear fields.
    u = np.column_stack([
        0.3 * g.nodes.coords[:, 0],
        -0.1 * g.nodes.coords[:, 1],
    ]).ravel()
    lin_err = np.max(np.abs(div @ u - 0.2 * g.cells.volumes))
    ok = lin_err < 1e-12
    print(f"  {'ok  ' if ok else 'FAIL'} linear divergence exactness: "
          f"max abs diff = {lin_err:.3e}")
    failures += 0 if ok else 1

    if failures:
        print(f"\n2D VEM validation failed ({failures} checks).")
    else:
        print("\n2D VEM operators match MATLAB MRST.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
