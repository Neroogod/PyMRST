"""Phase 110: SPE1 black-oil to compositional deck conversion vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    setup_compositional_spe1,
)
from pymrst.utils.units import BARSA

HERE = Path(__file__).parent
REF = HERE / "ref_p110_spe1_compositional.mat"


def compare(name, py, ref, atol, scale=1.0, unit="") -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    diff = float(np.max(np.abs(py - ref))) if py.size or ref.size else 0.0
    ok = py.shape == ref.shape and diff <= atol
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff / scale:.3e} {unit} "
        f"(atol {atol / scale:.1e} {unit})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p110_spe1_compositional.mat missing (run dump_phase110_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    deck, model, state0, comp_state0, comp_model, well_components = setup_compositional_spe1(
        interpolation="linear",
    )

    well_order = [w.name for w in model.wells]
    py_well_cells = np.asarray([w.cells[0] + 1 for w in model.wells], dtype=int)
    py_well_values = np.asarray([w.val for w in model.wells], dtype=float)
    py_well_compi = np.vstack([w.compi for w in model.wells])
    py_well_components = np.vstack([well_components[name] for name in well_order])

    print("\nPhase 110: SPE1 deck/state compositional conversion")
    failures = 0
    failures += compare("cartesian dimensions", model.G.cartDims, ref["cartDims"], 0.0)
    failures += compare("initial pressure", comp_state0.pressure, ref["pressure"], 1e-10 * BARSA, BARSA, "bar")
    failures += compare("initial saturation matrix", np.column_stack([comp_state0.sW, comp_state0.sO, comp_state0.sG]), ref["s"], 1e-13)
    failures += compare("overall pseudo components", comp_state0.z, ref["components"], 1e-8)
    failures += compare("liquid pseudo composition", comp_state0.x, ref["x"], 1e-8)
    failures += compare("vapour pseudo composition", comp_state0.y, ref["y"], 1e-12)
    failures += compare("liquid fraction L", comp_state0.L, ref["L"], 1e-12)
    failures += compare("temperature placeholder", comp_state0.T, ref["T"], 1e-13)
    failures += compare("well cells", py_well_cells, ref["wellCells"], 0.0)
    failures += compare("well control values", py_well_values, ref["wellValues"], 1e-10 * BARSA, BARSA, "bar")
    failures += compare("well compi", py_well_compi, ref["wellCompi"], 1e-14)
    failures += compare("well components = compi(2:end)", py_well_components, ref["wellComponents"], 1e-14)
    failures += compare("deck timesteps", deck["tsteps"], ref["tsteps"], 1e-8)

    # Both MRST and the PyMRST compositional SPE1 port switch the producer
    # schedule control to BHP before running the converted setup.
    control_type = str(ref["scheduleControl2Type"]).strip()
    failures += compare("MRST producer BHP switch value", ref["scheduleControl2Val"], [200.0 * BARSA], 1e-12 * BARSA, BARSA, "bar")
    failures += compare("MRST producer limits removed", ref["scheduleControl2HasLims"], [0], 0.0)
    print(f"  ok   MRST compositional producer control type = {control_type!r}")
    print(f"  ok   PyMRST pseudo mixture names = {comp_model.mixture.names}")
    print(f"  ok   PyMRST wells ordered as {well_order}")

    if failures:
        print(f"\nSPE1 compositional conversion validation failed ({failures} checks).")
    else:
        print("\nSPE1 black-oil to compositional conversion matches MATLAB.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
