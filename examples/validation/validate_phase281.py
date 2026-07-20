"""Phase 281: CompositionalFlowPropertyFunctions aggregate surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import flash, simple_mixture  # noqa: E402
from pymrst.modules.compositional.properties import (  # noqa: E402
    CompositionalFlowPropertyFunctions,
    compositional_density,
    compositional_flow_property_functions,
    compositional_viscosity_lv,
    fugacity_lv,
    phase_compressibility_factors_lv,
)
from pymrst.modules.compositional.properties.contents import list_properties  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 281: CompositionalFlowPropertyFunctions aggregate surface")
    mix = simple_mixture()
    p = np.array([5.0e6, 6.0e6])
    T = np.array([350.0, 355.0])
    z = np.array([[0.50, 0.30, 0.20], [0.45, 0.35, 0.20]])
    fr = flash(mix, p, T, z)

    aggregate = CompositionalFlowPropertyFunctions(mix)
    via_helper = compositional_flow_property_functions(mix)
    props = aggregate.evaluate(p, T, fr.x, fr.y, fr.Z_L, fr.Z_V)
    props_auto_z = via_helper(p, T, fr.x, fr.y)

    rho = compositional_density(mix, p, T, fr.x, fr.y, fr.Z_L, fr.Z_V)
    mu = compositional_viscosity_lv(mix, p, T, fr.x, fr.y, fr.Z_L, fr.Z_V)
    fug = fugacity_lv(mix, p, T, fr.x, fr.y)
    zfac = phase_compressibility_factors_lv(mix, p, T, fr.x, fr.y)
    catalog = list_properties()

    failures = 0
    failures += _check(
        "aggregate exposes all MRST LV property keys",
        all(key in props for key in [
            "ComponentPhaseMoleFractionsLV",
            "ComponentPhaseMassFractionsLV",
            "PhaseCompressibilityFactorsLV",
            "CompositionalDensity",
            "CompositionalViscosityLV",
            "FugacityLV",
            "PhaseMixingCoefficientsLV",
        ]),
    )
    failures += _check(
        "aggregate density/viscosity/fugacity match individual functions",
        all(np.allclose(a, b) for a, b in zip(props["CompositionalDensity"], rho))
        and all(np.allclose(a, b) for a, b in zip(props["CompositionalViscosityLV"], mu))
        and all(np.allclose(a, b) for a, b in zip(props["FugacityLV"], fug)),
    )
    failures += _check(
        "aggregate can compute omitted Z factors from phase compositions",
        all(np.allclose(a, b) for a, b in zip(props_auto_z["PhaseCompressibilityFactorsLV"], zfac)),
    )
    failures += _check(
        "properties catalog marks CompositionalFlowPropertyFunctions as implemented aggregate",
        catalog["CompositionalFlowPropertyFunctions"]["status"] == "implemented_aggregate"
        and catalog["CompositionalFlowPropertyFunctions"]["python_module"] == "compositional_flow_property_functions",
    )

    if failures:
        print(f"\nPhase 281 failed ({failures} checks).")
        return 1
    print("\nCompositionalFlowPropertyFunctions now has a real PyMRST aggregate import surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
