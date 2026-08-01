"""Check the VEM quadrature rules themselves against MRST.

`validate_phase403` pins `polygon_int`, which *uses* the triangle rule. That
is a strong test of the pair, but it cannot separate them: a rule whose points
and weights were both wrong in a compensating way would integrate polynomials
correctly and pass. This compares the raw tables.

Before the `vem/utils` split these four functions lived in one file with
`polygon_int`, and the parity auditor credited them because it credits by
file. Splitting them showed that nothing named them - so nothing pinned them.
This is that gap closed.

Run `dump_vem_quadrature_reference.m` in MATLAB first.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.vem.utils.poly_dim import poly_dim
from pymrst.modules.vem.utils.squeeze_block_diag import squeeze_block_diag
from pymrst.modules.vem.utils.tetrahedron_quad_rule import tetrahedron_quad_rule
from pymrst.modules.vem.utils.triangle_quad_rule import triangle_quad_rule

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "ref_vem_quadrature.mat"
TOLERANCE = 1e-12


def check(failures, label, got, want) -> None:
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    if got.shape != want.shape:
        failures.append(f"{label}: shape {got.shape} vs MRST {want.shape}")
        return
    if not np.allclose(got, want, rtol=TOLERANCE, atol=0):
        error = np.max(np.abs(got - want))
        failures.append(f"{label}: max abs diff {error:.3e}")
        return
    print(f"  {label:22} {got.size:3d} value(s)  ok")


def main() -> int:
    if not REFERENCE.exists():
        print(f"missing {REFERENCE.name}; run "
              f"dump_vem_quadrature_reference.m in MATLAB first")
        return 2
    ref = loadmat(REFERENCE)
    failures = []

    print("polyDim")
    ks = ref["polyDim_k"].ravel().astype(int)
    dims = ref["polyDim_dim"].ravel().astype(int)
    ours = np.array([[poly_dim(int(k), int(d)) for d in dims] for k in ks])
    check(failures, "polyDim table", ours, ref["polyDim_vals"])

    print("triangleQuadRule")
    for k in (1, 2, 3, 7):
        Xq, w, V, vol = triangle_quad_rule(k)
        check(failures, f"tri k={k} points", Xq, ref[f"tri{k}_Xq"])
        check(failures, f"tri k={k} weights", w, ref[f"tri{k}_w"])
        check(failures, f"tri k={k} vertices", V, ref[f"tri{k}_V"])
        check(failures, f"tri k={k} volume", vol, ref[f"tri{k}_vol"])

    print("tetrahedronQuadRule")
    for k in (2, 3):
        Xq, w, V, vol = tetrahedron_quad_rule(k)
        check(failures, f"tet k={k} points", Xq, ref[f"tet{k}_Xq"])
        check(failures, f"tet k={k} weights", w, ref[f"tet{k}_w"])
        check(failures, f"tet k={k} vertices", V, ref[f"tet{k}_V"])
        check(failures, f"tet k={k} volume", vol, ref[f"tet{k}_vol"])

    print("squeezeBlockDiag")
    A = ref["sbd_A"]
    check(failures, "squeeze columns",
          squeeze_block_diag(A, [2, 2, 2], A.shape[0], 2).toarray(),
          ref["sbd_down"])
    check(failures, "squeeze rows",
          squeeze_block_diag(A, [2, 2, 2], 2, A.shape[1]).toarray(),
          ref["sbd_across"])

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("VEM quadrature rules, polyDim and squeezeBlockDiag match MRST.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
