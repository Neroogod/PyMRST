"""Validate the computations behind MRST's interactiveDiagnostics:
`computeTOFandTracer` with the viewer's options plus well times of flight and
first arrival, `computeWellPairs`, `computeFandPhi` (and
`computeFandPhiFromDist`), `computeLorenz` and `computeSweep`.

The reference is MRST 2024a on the final state of simulatorWorkflowExample's
fine model (`dump_diagnostics_utils_reference.m`); the port runs the same
model (MRST's `makeModel3` draws) and applies the same calls. Each function is
checked twice where it matters: fed MRST's own inputs (`EXACT_RTOL`, the
function alone) and end to end from the port's run (`RUN_RTOL`, which carries
the fine run's 1e-9). The interactive viewer itself is not ported.
"""

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
EXACT_RTOL = 1e-12
RUN_RTOL = 1e-7


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    finite = np.isfinite(b)
    if not np.array_equal(finite, np.isfinite(a)):
        return np.inf
    if not finite.any():
        return 0.0
    return float(np.abs(a[finite] - b[finite]).max()/max(np.abs(b[finite]).max(), 1e-300))


def _close(name: str, a, b, tol: float) -> bool:
    err = _rel(a, b)
    return report(err < tol, f"{name} rel. diff {err:.1e}")


def _col(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim < 2 else x


def _rtd(mat) -> SimpleNamespace:
    """A MATLAB RTD struct as the port takes it: `pairIx` 0-based."""
    rtd = SimpleNamespace(**{k: getattr(mat, k) for k in mat._fieldnames})
    rtd.pairIx = np.atleast_2d(rtd.pairIx) - 1
    return rtd


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.simulator_workflow_example import (
        simulator_workflow_example,
    )
    from pymrst.modules.diagnostics.utils import (
        compute_f_and_phi, compute_lorenz, compute_sweep,
        compute_tof_and_tracer, compute_well_pairs,
    )
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock.pore_volume import pore_volume

    ref = loadmat(HERE/"ref_diagnostics_utils.mat", squeeze_me=True, struct_as_record=False)
    Dr, WPr = ref["Dsave"], ref["WPsave"]
    with isolated_gravity():
        out = simulator_workflow_example(plot=False, draws=(ref["draws_a"], ref["draws_s"]))
    G, rock, W, st = out.G, out.rock, out.W, out.states[-1]

    D = compute_tof_and_tracer(st, G, rock, wells=W, processCycles=True, computeWellTOFs=True,
                               firstArrival=True)
    ok = report(np.array_equal(D.inj + 1, np.atleast_1d(Dr.inj))
                and np.array_equal(D.prod + 1, np.atleast_1d(Dr.prod)),
                "computeTOFandTracer: injectors and producers identical")
    for name in ("tof", "itracer", "ptracer", "itof", "ptof", "ifa", "pfa"):
        ok &= _close(f"computeTOFandTracer: {name}", _col(getattr(D, name)), _col(getattr(Dr, name)),
                     RUN_RTOL)
    ok &= report(np.array_equal(D.ipart + 1, Dr.ipart) and np.array_equal(D.ppart + 1, Dr.ppart),
                 "computeTOFandTracer: partitions identical")

    WP = compute_well_pairs(st, G, rock, W, D)
    ok &= report(list(WP.pairs) == [str(p) for p in np.atleast_1d(WPr.pairs)]
                 and np.array_equal(WP.pairIx + 1, np.atleast_2d(WPr.pairIx)),
                 f"computeWellPairs: {len(WP.pairs)} pairs and their indices identical")
    ok &= _close("computeWellPairs: pair volumes", WP.vols, WPr.vols, RUN_RTOL)
    for side in ("inj", "prod"):
        wells = getattr(WP, side)
        for key in ("alloc", "ralloc", "z"):
            refs = np.atleast_1d(getattr(WPr, f"{side}_{key}"))
            if len(wells) == 1:
                refs = [getattr(WPr, f"{side}_{key}")]
            for k, w in enumerate(wells):
                actual = np.reshape(getattr(w, key), np.shape(refs[k]))
                if key == "ralloc":
                    # The rate left unallocated is round-off on both sides (the
                    # tracers sum to one), so it is measured against the rate.
                    rate = np.abs(np.asarray(w.alloc).sum(axis=1)).max()
                    err = np.abs(actual - refs[k]).max()/rate
                    ok &= report(err < RUN_RTOL, f"computeWellPairs: {side} {w.name} unallocated "
                                 f"rate {np.abs(actual).max():.1e} (MRST {np.abs(refs[k]).max():.1e}), "
                                 f"diff relative to the rate {err:.1e}")
                    continue
                tol = EXACT_RTOL if key == "z" else RUN_RTOL
                ok &= _close(f"computeWellPairs: {side} {w.name} {key}", actual, refs[k], tol)

    pv = pore_volume(G, rock)
    F, Phi = compute_f_and_phi(pv, D.tof)
    ok &= _close("computeFandPhi: F end to end", F, ref["F"], RUN_RTOL)
    ok &= _close("computeFandPhi: Phi end to end", Phi, ref["Phi"], RUN_RTOL)
    Fx, Phix = compute_f_and_phi(pv, Dr.tof)
    ok &= _close("computeFandPhi on MRST's tof: F", Fx, ref["F"], EXACT_RTOL)
    ok &= _close("computeFandPhi on MRST's tof: Phi", Phix, ref["Phi"], EXACT_RTOL)
    ok &= _close("computeLorenz", compute_lorenz(ref["F"], ref["Phi"]), ref["Lc"], EXACT_RTOL)
    ok &= _close("computeLorenz end to end", compute_lorenz(F, Phi), ref["Lc"], RUN_RTOL)
    Ev, tD = compute_sweep(ref["F"], ref["Phi"])
    ok &= _close("computeSweep: Ev", Ev, ref["Ev"], EXACT_RTOL)
    ok &= _close("computeSweep: tD", tD, ref["tD"], EXACT_RTOL)
    ok &= _close("computeLorenz, one curve per column", compute_lorenz(ref["F2"], ref["Phi2"]),
                 ref["Lc2"], EXACT_RTOL)
    Ev2, tD2 = compute_sweep(ref["F2"], ref["Phi2"])
    ok &= _close("computeSweep on a matrix (MATLAB's linear order): Ev", Ev2, ref["Ev2"], EXACT_RTOL)
    ok &= _close("computeSweep on a matrix: tD", tD2, ref["tD2"], EXACT_RTOL)

    base = _rtd(ref["RTD"])
    estimated = _rtd(ref["RTD"])
    estimated.creator = "estimateRTD"
    for tag, rtd, kwargs, Fref, Phiref in (
            ("an RTD", base, {}, ref["Fr"], ref["Phir"]),
            ("a summed RTD", _rtd(ref["RTDs"]), {"sum": True}, ref["Frs"], ref["Phirs"]),
            ("an estimated RTD", estimated, {}, ref["Fre"], ref["Phire"])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fp, Phip = compute_f_and_phi(rtd, **kwargs)
        ok &= _close(f"computeFandPhi from {tag}: F", Fp, _col(Fref), EXACT_RTOL)
        ok &= _close(f"computeFandPhi from {tag}: Phi", Phip, _col(Phiref), EXACT_RTOL)
    print("\nThe diagnostics computations match MRST" if ok
          else "\nThe diagnostics computations do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
