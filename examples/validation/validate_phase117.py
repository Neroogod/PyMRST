"""Phase 117: compositional performance benchmark structure vs MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.examples.book_ii.compositional_performance_example import (
    make_performance_case,
    performance_backends,
    run_compositional_performance_example,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p117_compositional_performance.mat"


def _labels(raw) -> list[str]:
    arr = np.asarray(raw, dtype=object).ravel()
    return [str(x[0] if isinstance(x, np.ndarray) else x) for x in arr]


def compare(name, py, ref, rtol=0.0, atol=0.0) -> int:
    py = np.asarray(py)
    ref = np.asarray(ref)
    if py.dtype.kind in {"U", "S", "O"} or ref.dtype.kind in {"U", "S", "O"}:
        ok = list(py.ravel()) == list(ref.ravel())
        diff = 0.0 if ok else np.inf
    elif py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.max(np.abs(py.astype(float) - ref.astype(float)))) if py.size else 0.0
        scale = max(float(np.max(np.abs(ref.astype(float)))), 1.0)
        ok = diff <= atol + rtol * scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
        f"(rtol {rtol:.1e}, atol {atol:.1e})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p117_compositional_performance.mat missing (run dump_phase117_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    full_dims = tuple(np.asarray(ref["dims"], dtype=int).ravel())
    backends = performance_backends()
    rows = run_compositional_performance_example((3, 3, 2))
    _, _, mix, info, _, model, state0 = make_performance_case((3, 3, 2))

    print("\nPhase 117: compositional performance benchmark structure vs MATLAB")
    failures = 0
    failures += compare("full benchmark cells", np.prod(full_dims), ref["cells"], atol=0.0)
    failures += compare("full benchmark degrees of freedom", np.prod(full_dims) * mix.n,
                        ref["degrees_of_freedom"], atol=0.0)
    failures += compare("SPE5 component count", mix.n, ref["ncomp"], atol=0.0)
    failures += compare("SPE5 initial composition", info.initial, ref["initial"],
                        rtol=0.0, atol=1.0e-14)
    failures += compare("SPE5 injection composition", info.injection, ref["injection"],
                        rtol=0.0, atol=1.0e-14)
    failures += compare("SPE5 initial pressure", info.pressure, ref["pressure"],
                        rtol=0.0, atol=1.0e-8)
    failures += compare("SPE5 temperature", info.T, ref["temperature"],
                        rtol=0.0, atol=1.0e-12)
    mrst_like = [backends[i] for i in (0, 1, 3, 4)]
    failures += compare("backend labels", [b.label() for b in mrst_like], _labels(ref["backend_labels"]))
    failures += compare("backend use_mex flags", [b.use_mex for b in mrst_like],
                        ref["backend_use_mex"], atol=0.0)
    failures += compare("backend row_major flags", [b.row_major for b in mrst_like],
                        ref["backend_row_major"], atol=0.0)
    failures += compare("backend deferred flags", [b.deferred_assembly for b in mrst_like],
                        ref["backend_deferred"], atol=0.0)
    failures += compare("python row-major backend label", [backends[2].label()],
                        ["Diagonal (RowMajor)"])
    failures += compare("python row-major backend flags",
                        [backends[2].use_mex, backends[2].row_major, backends[2].deferred_assembly],
                        [False, True, False], atol=0.0)
    failures += compare("scaled case cells", model.g.cells.num, 18, atol=0.0)
    failures += compare("scaled state composition rows", state0.z.shape[0], 18, atol=0.0)
    failures += compare("available timing rows", [row.available for row in rows],
                        [b.available for b in backends], atol=0.0)

    if failures:
        print(f"\nPhase 117 validation failed ({failures} checks).")
    else:
        print("\nCompositional performance benchmark structure matches MATLAB; available diagonal kernel rows ran successfully.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
