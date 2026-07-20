"""Phase 135: compositional accuracy example plotting-data parity."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.examples.book_ii.compositional_accuracy_example import (
    accuracy_component_profiles,
    accuracy_cumulative_iterations,
    accuracy_gas_saturation_profiles,
    accuracy_property_profiles,
    run_compositional_accuracy_example,
)
from pymrst.utils.units import DAY


def _check(name: str, ok: bool) -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def main() -> int:
    out = run_compositional_accuracy_example(nx=6, nstep=3, dt=5.0 * DAY)
    names = out["names"]
    states = out["states"]
    reports = out["reports"]
    model = out["packed_problems"][2].model

    comp = accuracy_component_profiles(states, names, model.mix.names, step_stride=1)
    gas = accuracy_gas_saturation_profiles(model, states, names, step_stride=1)
    props = accuracy_property_profiles(model, states[0], step_stride=1)
    its = accuracy_cumulative_iterations(reports, names)

    print("\nPhase 135: compositional accuracy plotting-data parity")
    failures = 0
    failures += _check("packed problems preserve MRST accuracy ordering",
                       names == ["Explicit", "Natural", "Overall"]
                       and [p.directory for p in out["packed_problems"]] == ["compAccExample"] * 3)
    failures += _check("component profile frames follow MRST 1:stride:n loop",
                       np.array_equal(comp["steps"], np.array([1, 2, 3]))
                       and comp["markers"] == ("-", "--", "--")
                       and comp["line_widths"] == (1.0, 2.5, 5.0))
    failures += _check("component profile data has case/cell/component shape",
                       len(comp["frames"]) == 3
                       and comp["frames"][0]["profiles"]["Explicit"].shape == (6, model.mix.n)
                       and np.allclose(comp["frames"][0]["profiles"]["Overall"].sum(axis=1), 1.0))
    failures += _check("gas saturation profile mirrors MRST states{i}{step}.s(:,2)",
                       len(gas["frames"]) == 3
                       and gas["line_styles"] == ("-", "-", "--")
                       and gas["frames"][0]["profiles"]["Natural"].shape == (6,)
                       and np.all(gas["frames"][0]["profiles"]["Overall"] >= 0.0))
    failures += _check("property profile exposes viscosity/density sections",
                       len(props["frames"]) == 3
                       and props["legend"] == ("mu_l", "mu_v", "rho_l", "rho_v", "S_g", "z (CO2)")
                       and props["frames"][0]["mu_liquid_cp"].shape == (6,)
                       and props["frames"][0]["rho_vapour"].shape == (6,))
    failures += _check("property profile keeps gas saturation and CO2 composition overlays",
                       props["frames"][0]["gas_saturation"].shape == (6,)
                       and props["frames"][0]["component"].shape == (6,)
                       and props["frames"][0]["component_index"] == 1)
    failures += _check("implicit cumulative-iteration table mirrors MATLAB plot",
                       its["names"] == ("Natural", "Overall")
                       and its["iterations"].shape == (3, 2)
                       and its["cumulative"].shape == (3, 2)
                       and np.all(np.diff(its["cumulative"], axis=0) >= 0))
    failures += _check("runner returns all MRST plot data sections",
                       set(["component_profiles", "gas_saturation_profiles",
                            "property_profiles", "cumulative_iterations"]).issubset(out))

    if failures:
        print(f"\nPhase 135 failed ({failures} checks).")
    else:
        print("\nAccuracy example now exposes MRST-shaped plotting data.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
