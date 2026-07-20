"""Phase 68: VEM stress utilities against MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.vem import (
    cal_stress_eigs_vem,
    calculate_stress_vem,
    create_augmented_grid,
    enu2c,
    enu2c_2d,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p68_vem_stress.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def compare(name, py, ml, rtol=1e-10, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ml))), 1.0)
    diff = float(np.max(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def _sort_eigs(vals):
    return np.sort(np.asarray(vals, dtype=float), axis=1)


def _eigen_residual_2d(stress, sigm, evec):
    stress = np.asarray(stress, dtype=float)
    sigm = np.asarray(sigm, dtype=float)
    evec = np.asarray(evec, dtype=float)
    worst = 0.0
    for s, vals, vec_flat in zip(stress, sigm, evec):
        tensor = np.array([[s[0], s[2]], [s[2], s[1]]])
        vecs = vec_flat.reshape(2, 2, order="F")
        for k in range(2):
            r = tensor @ vecs[:, k] - vals[k] * vecs[:, k]
            worst = max(worst, float(np.max(np.abs(r))))
    return worst


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase68_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failures = 0
    print("Phase 68: VEM stress utilities vs MATLAB MRST")

    g2 = create_augmented_grid(compute_geometry(
        cart_grid([int(ref["nx2"]), int(ref["ny2"])], _arr(ref, "L2"))
    ))
    C2 = enu2c_2d(
        np.full(g2.cells.num, float(ref["E2"])),
        np.full(g2.cells.num, float(ref["nu2"])),
    )
    stress2, strain2 = calculate_stress_vem(g2, _arr(ref, "uu2"), C2)
    sigm2, evec2 = cal_stress_eigs_vem(g2, stress2)

    failures += compare("2D stress", stress2, _arr(ref, "stress2"), rtol=1e-10)
    failures += compare("2D strain", strain2, _arr(ref, "strain2"), rtol=1e-12)
    failures += compare("2D principal stress", _sort_eigs(sigm2),
                        _sort_eigs(_arr(ref, "sigm2")), rtol=1e-10)
    eig_res = _eigen_residual_2d(stress2, sigm2, evec2)
    ok = eig_res < 1e-8
    print(f"  {'ok  ' if ok else 'FAIL'} 2D eigenvector residual: "
          f"max abs diff = {eig_res:.3e}")
    failures += 0 if ok else 1

    g3 = create_augmented_grid(compute_geometry(
        cart_grid(
            [int(ref["nx3"]), int(ref["ny3"]), int(ref["nz3"])],
            _arr(ref, "L3"),
        )
    ))
    C3 = enu2c(
        np.full(g3.cells.num, float(ref["E3"])),
        np.full(g3.cells.num, float(ref["nu3"])),
    )
    stress3, strain3 = calculate_stress_vem(g3, _arr(ref, "uu3"), C3)
    sigm3, _ = cal_stress_eigs_vem(g3, stress3)

    failures += compare("3D stress", stress3, _arr(ref, "stress3"), rtol=1e-10)
    failures += compare("3D strain", strain3, _arr(ref, "strain3"), rtol=1e-12)
    failures += compare("3D principal stress", _sort_eigs(sigm3),
                        _sort_eigs(_arr(ref, "sigm3")), rtol=1e-10)

    if failures:
        print(f"\nPhase 68 failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST VEM stress utilities.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
