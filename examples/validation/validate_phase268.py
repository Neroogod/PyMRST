"""Phase 268: table-backed compositional fluid constructor parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids import (  # noqa: E402
    CoolPropsCompositionalMixture,
    TableCompositionalMixture,
    list_fluids,
    table_compositional_mixture,
)
from pymrst.modules.compositional.utils import (  # noqa: E402
    coolPropFluidsStructs,
    cool_prop_fluids_structs,
    list_utils,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    fluids = coolPropFluidsStructs()
    fluids_alias = cool_prop_fluids_structs()
    names = TableCompositionalMixture.getFluidList()
    mix = TableCompositionalMixture(["Methane", "n-Decane"], ["C1", "C10"])
    mix_alias = table_compositional_mixture("CarbonDioxide", outnames="CO2")
    fluid_catalog = list_fluids()
    util_catalog = list_utils()

    missing_raised = False
    try:
        TableCompositionalMixture(["NotAFluid"])
    except ValueError:
        missing_raised = True

    coolprop_missing_or_listed = True
    try:
        cool_names = CoolPropsCompositionalMixture.getFluidList()
        coolprop_missing_or_listed = isinstance(cool_names, tuple) and len(cool_names) > 0
    except ImportError:
        coolprop_missing_or_listed = True

    print("\nPhase 268: table-backed compositional fluid constructor parity")
    failures = 0
    failures += _check("coolPropFluidsStructs returns MRST table records",
                       len(fluids) == 122
                       and fluids == fluids_alias
                       and fluids[0].name == "R507A"
                       and any(f.name == "Methane" for f in fluids)
                       and any(f.name == "n-Decane" for f in fluids))
    failures += _check("TableCompositionalMixture fluid list follows table order",
                       names[0] == "R507A"
                       and "Methane" in names
                       and "n-Decane" in names
                       and len(names) == len(fluids))
    failures += _check("TableCompositionalMixture builds real Mixture values",
                       mix.names == ("C1", "C10")
                       and mix.name == "CoolProp - Tabulated"
                       and np.allclose(mix.Tcrit, [190.564, 617.7])
                       and np.allclose(mix.Pcrit, [4599200.0, 2103000.0])
                       and np.allclose(mix.acentricFactors, [0.01142, 0.4884])
                       and np.allclose(mix.molarMass, [0.0160428, 0.14228168])
                       and np.allclose(mix.Vcrit, mix.molarMass / np.array([162.660002678, 233.3419552])))
    failures += _check("single-name constructor and outname alias work",
                       mix_alias.names == ("CO2",)
                       and np.allclose(mix_alias.Tcrit, [304.1282])
                       and np.allclose(mix_alias.Pcrit, [7377300.0]))
    failures += _check("unknown table names are rejected", missing_raised)
    failures += _check("CoolPropsCompositionalMixture is optional-dependency honest",
                       coolprop_missing_or_listed)
    failures += _check("catalog statuses reflect newly implemented table surfaces",
                       fluid_catalog["TableCompositionalMixture"]["status"] == "implemented"
                       and fluid_catalog["CoolPropsCompositionalMixture"]["status"] == "implemented_optional_dependency"
                       and util_catalog["coolPropFluidsStructs"]["status"] == "implemented")

    if failures:
        print(f"\nPhase 268 failed ({failures} checks).")
        return 1
    print("\nTableCompositionalMixture and coolPropFluidsStructs now mirror MRST's stored CoolProp table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
