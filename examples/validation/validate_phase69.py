"""Phase 69: TEDFM VEM wrappers, validated against MATLAB MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.vem import (
    c2d_tedfm,
    calculate_qc_tedfm,
    calculate_qf_tedfm,
    create_augmented_grid,
    enu2c,
    enu2c_2d,
    vem3d_div_tedfm,
    vem_assemble_tedfm,
    vem_div_tedfm,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p69_vem_tedfm.mat"


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
            f"Missing {REF.name}. Run dump_phase69_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    E = float(ref["E"])
    nu = float(ref["nu"])

    g2 = create_augmented_grid(compute_geometry(cart_grid([3, 2], [3, 2])))
    gw2 = SimpleNamespace(Matrix=g2, griddim=g2.griddim)
    C2 = enu2c_2d(np.full(g2.cells.num, E), np.full(g2.cells.num, nu))
    qf2, qfvol2 = calculate_qf_tedfm(gw2)
    D2 = c2d_tedfm(C2, gw2)
    S2 = vem_assemble_tedfm(gw2, C2)
    ml_nodes2 = np.asarray(ref["G2_cell_nodes"], dtype=np.int64).ravel() - 1
    ml_pos2 = np.asarray(ref["G2_cell_node_pos"], dtype=np.int64).ravel() - 1
    qf2_mrst_order = np.zeros_like(_arr(ref, "qf2"))
    qfvol2_mrst_order = np.zeros_like(_arr(ref, "qfvol2").ravel())
    for c in range(g2.cells.num):
        py_sl = slice(g2.cells.nodePos[c], g2.cells.nodePos[c + 1])
        ml_sl = slice(ml_pos2[c], ml_pos2[c + 1])
        py_lookup = {node: i for i, node in enumerate(g2.cells.nodes[py_sl])}
        order = [py_lookup[node] for node in ml_nodes2[ml_sl]]
        qf2_mrst_order[ml_sl] = qf2[py_sl][order]
        qfvol2_mrst_order[ml_sl] = qfvol2[py_sl][order]

    g3 = create_augmented_grid(compute_geometry(cart_grid([2, 2, 2],
                                                          [2, 2, 2])))
    gw3 = SimpleNamespace(Matrix=g3, griddim=g3.griddim)
    C3 = enu2c(np.full(g3.cells.num, E), np.full(g3.cells.num, nu))
    qc3, qf3, qcvol3 = calculate_qc_tedfm(gw3)
    D3 = c2d_tedfm(C3, gw3)
    div3 = vem3d_div_tedfm(gw3)
    vdiv3 = vem_div_tedfm(gw3)
    S3 = vem_assemble_tedfm(gw3, C3)

    failures = 0
    print("Phase 69: TEDFM VEM wrappers vs MATLAB MRST")
    failures += compare("2D calculateQFTEDFM qf", qf2_mrst_order,
                        _arr(ref, "qf2"), rtol=1e-12)
    failures += compare("2D calculateQFTEDFM qf_vol", qfvol2_mrst_order,
                        _arr(ref, "qfvol2"), rtol=1e-12)
    failures += compare("2D C2DTEDFM", D2, _arr(ref, "D2"), rtol=1e-14)
    failures += compare("2D VEM_assembleTEDFM", S2, ref["S2"], rtol=1e-10)

    failures += compare("3D calculateQCTEDFM qc", qc3, _arr(ref, "qc3"),
                        rtol=1e-12)
    failures += compare("3D calculateQCTEDFM qf", qf3, _arr(ref, "qf3"),
                        rtol=1e-12)
    failures += compare("3D calculateQCTEDFM qcvol", qcvol3,
                        _arr(ref, "qcvol3"), rtol=1e-12)
    failures += compare("3D C2DTEDFM", D3, _arr(ref, "D3"), rtol=1e-14)
    failures += compare("3D VEM3D_divTEDFM", div3, ref["div3"],
                        rtol=1e-12)
    failures += compare("3D VEM_divTEDFM", vdiv3, ref["vdiv3"],
                        rtol=1e-12)
    failures += compare("3D VEM_assembleTEDFM", S3, ref["S3"], rtol=1e-10)

    if failures:
        print(f"\nTEDFM VEM validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST TEDFM VEM wrappers.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
