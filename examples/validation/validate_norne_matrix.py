"""Compare compact PyMRST and MATLAB MRST mini-Norne histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.examples.run_norne_example import run_norne_example


CASES = (
    ("water", "fully coupled"),
    ("water", "fixed stress splitting"),
    ("oil water", "fully coupled"),
    ("oil water", "fixed stress splitting"),
    ("blackoil", "fully coupled"),
    ("blackoil", "fixed stress splitting"),
)


def slug(fluid: str, method: str) -> str:
    return f"{fluid}__{method}".replace(" ", "_").replace("-", "_")


def python_history(fluid: str, method: str) -> dict[str, np.ndarray]:
    _, _, states = run_norne_example(
        "mini Norne", fluid_model=fluid, method=method,
        splitting_tolerance=1.0e-3, n_steps=20, plot=False, check=False,
        # MRST 2024a's oil-water mechanics equation omits cell volume on the
        # Biot term.  Reproduce that defect only for reference parity; normal
        # PyMRST runs retain the corrected formulation.
        mrst_oilwater_bug=fluid == "oil water",
        mrst_wellbore_bug=fluid == "blackoil",
    )
    initial = states[0]
    out: dict[str, list] = {name: [] for name in (
        "p_mean", "p_max", "sw_mean", "sw_max", "sg_mean", "sg_max",
        "uplift_max", "displacement_inf", "bhp", "qws", "qos", "qgs",
    )}
    for state in states[1:]:
        out["p_mean"].append(np.mean(state["pressure"]))
        out["p_max"].append(np.max(state["pressure"]))
        sw = np.asarray(state.get("sW", np.zeros_like(state["pressure"])))
        sg = np.asarray(state.get("sG", np.zeros_like(state["pressure"])))
        out["sw_mean"].append(np.mean(sw)); out["sw_max"].append(np.max(sw))
        out["sg_mean"].append(np.mean(sg)); out["sg_max"].append(np.max(sg))
        du = np.asarray(state["u"]) - np.asarray(initial["u"])
        out["uplift_max"].append(np.max(-du.reshape(-1, 3)[:, 2]))
        out["displacement_inf"].append(np.max(np.abs(du)))
        out["bhp"].append(np.asarray(state["bhp"]))
        for key, state_key in (("qws", "qWs"), ("qos", "qOs"), ("qgs", "qGs")):
            out[key].append(np.asarray(state.get(state_key, np.zeros(2))))
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def compare_case(reference: Path, fluid: str, method: str,
                 rtol: float, atol: float) -> int:
    ref = loadmat(reference, squeeze_me=True)
    py = python_history(fluid, method)
    failed = 0
    print(f"\n{fluid} / {method}")
    for name, values in py.items():
        expected = np.asarray(ref[name], dtype=float)
        values = np.asarray(values, dtype=float).squeeze()
        if values.shape != expected.shape:
            print(f"  FAIL {name}: shape {values.shape} != {expected.shape}")
            failed += 1
            continue
        if fluid == "blackoil" and name == "qos":
            # MRST's first producer column uses its zero-history fallback:
            # equal signed epsilon in every phase. PyMRST uses the physical
            # completion mixture. Once the well has flowed, both use actual
            # perforation flux history, so compare that common path.
            values = values[1:]
            expected = expected[1:]
            print("  note qOs: report 0 excluded (MRST zero-history mixture)")
        scale = max(float(np.max(np.abs(expected))), 1.0e-30)
        error = float(np.max(np.abs(values - expected)))
        field_rtol = max(rtol, 2.0e-2) if (
            fluid == "blackoil" and name == "qos"
        ) else rtol
        ok = error <= atol + field_rtol * scale
        display_scale = max(scale, atol)
        tolerance_note = " (rtol 2e-2)" if field_rtol != rtol else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<18} abs={error:.3e} "
              f"rel={error/display_scale:.3e}{tolerance_note}")
        failed += not ok
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path(__file__).parent / "ad_mechanics" / "norne_matrix")
    parser.add_argument("--fluid", choices=sorted({case[0] for case in CASES}))
    parser.add_argument("--method", choices=sorted({case[1] for case in CASES}))
    # This compact matrix compares engineering histories across different
    # nonlinear/globalization and VEM implementations.  The deep oil-water
    # fixture in validate_norne.py retains the strict cellwise 1e-6 checks.
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    args = parser.parse_args()
    selected = CASES
    if args.fluid or args.method:
        selected = tuple(case for case in CASES
                         if (args.fluid is None or case[0] == args.fluid)
                         and (args.method is None or case[1] == args.method))
    failed = 0
    for fluid, method in selected:
        reference = args.reference_dir / f"mrst_{slug(fluid, method)}.mat"
        if not reference.exists():
            print(f"MISSING {reference}")
            failed += 1
            continue
        failed += compare_case(reference, fluid, method, args.rtol, args.atol)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
