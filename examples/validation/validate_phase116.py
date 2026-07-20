"""Phase 116: EOS separator tree vs MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.examples.book_ii.compositional_separator_example import (
    run_compositional_separator_example,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p116_compositional_separators.mat"


def _phase_rows(values):
    return np.asarray([np.asarray(v, dtype=float).reshape(-1)[0] for v in values])


def _phase_fractions(values):
    return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in values])


def _phase_component_streams(values):
    return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in values])


def _phase_totals(values):
    return _phase_rows([np.asarray(v, dtype=float).sum(axis=1) for v in values])


def compare(name, py, ref, rtol=0.0, atol=0.0) -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.nanmax(np.abs(py - ref))) if py.size else 0.0
        ref_scale = max(float(np.nanmax(np.abs(ref))), 1.0)
        ok = diff <= atol + rtol * ref_scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
        f"(rtol {rtol:.1e}, atol {atol:.1e})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p116_compositional_separators.mat missing (run dump_phase116_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    out = run_compositional_separator_example(stream_mole=np.asarray(ref["stream_mole"], dtype=float))
    single = out["single"]
    tree = out["tree"]

    print("\nPhase 116: EOS separator tree vs MATLAB")
    failures = 0
    failures += compare("tree phase destination", out["tree_separator"].phase_destination,
                        ref["tree_phase_destination"], atol=0.0)
    failures += compare("single surface rates", _phase_rows(single.surface_rates),
                        ref["single_q"], rtol=1.0e-8, atol=1.0e-14)
    failures += compare("single phase streams", _phase_totals(single.phase_streams),
                        ref["single_stream"], rtol=1.0e-8, atol=1.0e-12)
    failures += compare("single phase mole fractions", _phase_fractions(single.phase_mole_fractions),
                        ref["single_frac"], rtol=1.0e-8, atol=1.0e-10)
    failures += compare("single molar densities", _phase_rows(single.molar_densities),
                        ref["single_density"], rtol=1.0e-8, atol=1.0e-6)

    failures += compare("tree surface rates", _phase_rows(tree.surface_rates),
                        ref["tree_q"], rtol=5.0e-5, atol=1.0e-9)
    failures += compare("tree final streams", _phase_totals(tree.phase_streams),
                        ref["tree_stream"], rtol=1.0e-6, atol=1.0e-9)
    failures += compare("tree final mole fractions", _phase_fractions(tree.phase_mole_fractions),
                        ref["tree_frac"], rtol=1.0e-6, atol=1.0e-9)
    failures += compare("tree molar densities", _phase_rows(tree.molar_densities),
                        ref["tree_density"], rtol=5.0e-7, atol=1.0e-3)
    failures += compare("tree surface component streams", _phase_component_streams(tree.surface_mole_streams),
                        ref["tree_surface_streams"], rtol=1.0e-6, atol=1.0e-9)

    if failures:
        print(f"\nPhase 116 validation failed ({failures} checks).")
    else:
        print("\nEOS separator and separator tree match MATLAB within the current parity envelope.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
