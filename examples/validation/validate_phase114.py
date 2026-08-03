"""Phase 114: native SPE1 K-value compositional solve vs MATLAB MRST.

This validator compares the two-report-step native PyMRST compositional SPE1
run against ``dump_phase114_reference.m``.  The EOS/K-table layer is already
strictly validated in phase 111; this phase checks end-to-end schedule behavior
with tolerances that expose the remaining facility-model differences.

This phase used to run under a 5 bar pressure envelope and observe about 4.6
bar. That gap is now closed - the overall-composition model agrees with MATLAB
to 4.6e-04 bar - and the tolerances below are sized to the result rather than
to the old discrepancy. The cause is worth recording, because the explanation
the file previously carried was wrong.

The old note blamed "pseudo-component composition interpolation". Phase 111
rules that out: every K-value and property table matches MATLAB to 1e-13 or
better. Rebuilding those tables on MRST's literal raw coordinates in fact makes
things *worse* (32 bar), because pressure ~1e7 against z ~1 triangulates into
slivers - the normalization in ``_interpolate_to_mesh_2d`` is load-bearing, not
a deviation. Also ruled out by measurement: timestep control (neither side
subdivides a report step), convergence (the gap sat at 4.5726 bar from tol 1e-5
through 1e-11 while iteration counts grew), and transmissibility, well index,
relperm and initial state - phase 24 matches WI to 4e-28 and the black-oil SPE1
history to ~1e-14 relative, and phase 110 matches the initial state.

What it actually was: **the compositional model had no rock or water
compressibility**. ``self.pv`` is fixed at construction from grid and rock
alone, and the accumulation terms used that same constant pore volume at both
time levels, so ``pvMultR`` dropped out of the system entirely. The water
equation was ``pv (sW - sW0)/dt`` - no ``b_w`` at all - and since water is
immobile below Swcr the flux term vanishes and that pins ``sW`` to its initial
value. MRST keeps the deck's ``bW``/``muW`` through
``convertBlackOilModelToCompositionalModel`` (it strips only
``rsSat, bO, bG, muO, muG``) and carries ``pvMultR`` with the rock, so it has
both.

The signature was an areal dipole, uniform through all three layers: +3.1 bar
at the injector, -2.2 bar at the producer, field mean near zero, growing to
+4.5/-2.6 by the second step. PyMRST held a steeper injector-to-producer
gradient - same gas injected at a higher BHP, ~2% less oil at the same 200 bar
- which is what a system missing part of its compressibility looks like.

Restoring both terms (``water_b`` and ``rock_pv_mult``, wired from the deck's
PVTW and ROCK in ``run_compositional_spe1._deck_water_and_rock``) moves the
overall model from 4.5726 bar to 4.6e-04, and PyMRST's ``sW`` spread from
2.8e-17 to 7.32e-04 against MATLAB's 7.32e-04. Rock compressibility was the
larger of the two (4.57 -> 0.40 bar on its own); water closed the remainder.

A second, smaller discrepancy turned out to be in the reference rather than in
PyMRST. The natural-variable model sat at 7.0e-02 bar against the old .mat -
150x looser than the overall model - but PyMRST's two formulations agreed with
*each other* to 1.0e-10 bar, while MATLAB's two disagreed by 7.08e-02 bar. The
reference had been dumped at MRST's default 1e-3 tolerances and was simply not
converged. Re-running `dump_phase114_reference.m` with tolerances tightened to
1e-7 collapses MATLAB's own spread to 6.3e-06 bar, and both PyMRST models then
match it to ~6e-05 bar. Nothing in PyMRST changed.

That is also why the Newton-iteration check is a direction rather than an
equality - see `_compare_iterations`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_native_compositional_spe1,
)
from pymrst.utils.units import BARSA

HERE = Path(__file__).parent
REF = HERE / "ref_p114_spe1_native_compositional.mat"


def _saturations(out):
    model = out.compositional_model
    rows = []
    for st in out.compositional_states:
        if hasattr(st, "sO") and st.sO is not None:
            s_o, s_g = st.sO, st.sG
        else:
            fp = backend_flow_props(model.mix, model.eos_model, st.pressure, model.T, st.z)
            s_o = (1.0 - st.sW) * fp.sL
            s_g = (1.0 - st.sW) * fp.sV
        rows.append(np.column_stack([st.sW, s_o, s_g]))
    return np.stack(rows)


def _pack(out):
    return {
        "pressure": np.vstack([st.pressure for st in out.compositional_states]),
        "components": np.stack([st.z for st in out.compositional_states]),
        "s": _saturations(out),
        "bhp": np.vstack([st.bhp for st in out.compositional_states]),
        "qOs": np.vstack([st.qOs for st in out.compositional_states]),
        "qGs": np.vstack([st.qGs for st in out.compositional_states]),
        "iterations": np.asarray(out.iterations, dtype=float),
    }


def compare(name, py, ref, rtol=0.0, atol=0.0, scale=1.0, unit="") -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.max(np.abs(py - ref))) if py.size else 0.0
        ref_scale = max(float(np.max(np.abs(ref))), 1.0)
        ok = diff <= atol + rtol * ref_scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff / scale:.3e} {unit} "
        f"(rtol {rtol:.1e}, atol {atol / scale:.1e} {unit})"
    )
    return 0 if ok else 1


# One budget for both formulations. They briefly needed separate ones - the
# natural-variable model sat ~150x looser - but that was the reference being
# under-converged, not a real difference, and both now land at ~6e-05 bar.
# Sized to what the run achieves with an order of magnitude or so of headroom
# for platform noise.
TOLERANCES = {
    "pressure": 5.0e-3 * BARSA,
    "components": 1.0e-7,
    "s": 1.0e-6,
    "bhp": 5.0e-3 * BARSA,
    "qOs": 5.0e-8,
}


def _validate_model(label: str, natural: bool, ref: dict) -> int:
    out = run_native_compositional_spe1(
        nsteps=2,
        natural=natural,
        min_dt=6.0 * 3600.0,
        interpolation="linear",
    )
    py = _pack(out)
    prefix = "natural" if natural else "overall"
    tol = TOLERANCES
    print(f"\n{label}")
    failures = 0
    failures += compare("pressure", py["pressure"], ref[f"{prefix}_pressure"],
                        rtol=0.0, atol=tol["pressure"], scale=BARSA, unit="bar")
    failures += compare("overall pseudo components", py["components"], ref[f"{prefix}_components"],
                        rtol=0.0, atol=tol["components"])
    failures += compare("phase saturations", py["s"], ref[f"{prefix}_s"],
                        rtol=0.0, atol=tol["s"])
    failures += compare("well BHP", py["bhp"], ref[f"{prefix}_bhp"],
                        rtol=0.0, atol=tol["bhp"], scale=BARSA, unit="bar")
    failures += compare("well oil surface rate", py["qOs"], ref[f"{prefix}_qOs"],
                        rtol=0.0, atol=tol["qOs"])
    failures += compare("well gas surface rate", py["qGs"], ref[f"{prefix}_qGs"],
                        rtol=0.0, atol=1.0e-6)
    failures += _compare_iterations(py["iterations"], ref[f"{prefix}_iterations"])
    return failures


def _compare_iterations(py, ref) -> int:
    """PyMRST must not need more Newton iterations than MATLAB.

    Matching iteration counts exactly stopped being meaningful once the
    reference was converged properly: MATLAB now needs [4 7] and [9 10] where
    PyMRST needs [3 4], so an equality check fails on PyMRST being faster.
    Iteration count is a property of the solver, not of the physics - the
    states themselves are compared above, to 5e-3 bar. What is still worth
    asserting is the direction: if PyMRST ever starts needing *more* iterations
    than MATLAB, something has regressed in the Newton path.
    """
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ok = py.shape == ref.shape and bool(np.all(py <= ref))
    print(f"  {'ok  ' if ok else 'FAIL'} Newton iterations: PyMRST {py.astype(int).tolist()} "
          f"vs MATLAB {ref.astype(int).tolist()} (PyMRST must not exceed)")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p114_spe1_native_compositional.mat missing (run dump_phase114_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    print("\nPhase 114: native SPE1 K-value compositional solve vs MATLAB")
    failures = 0
    failures += _validate_model("Overall-composition model", False, ref)
    failures += _validate_model("Natural-variable model", True, ref)
    if failures:
        print(f"\nPhase 114 validation failed ({failures} checks).")
    else:
        print("\nNative SPE1 compositional run matches MATLAB within the tightened envelope.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
