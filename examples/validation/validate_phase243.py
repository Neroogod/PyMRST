"""Phase 243: compositional property Contents.m catalog parity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.properties.contents import (  # noqa: E402
    MRST_CONTENTS_PROPERTIES,
    PROPERTIES,
    list_mrst_contents_properties,
    list_properties,
)
from pymrst.modules.compositional.properties.components.contents import (  # noqa: E402
    COMPONENT_PROPERTIES,
    MRST_CONTENTS_COMPONENT_PROPERTIES,
    list_component_properties,
    list_mrst_contents_component_properties,
)


EXPECTED_PROPERTIES = (
    ("ComponentPhaseMassFractionsLV", ""),
    ("ComponentPhaseMoleFractionsLV", ""),
    ("CompositionalDensity", ""),
    ("CompositionalFlowPropertyFunctions", ""),
    ("CompositionalViscosityLV", ""),
    ("FugacityLV", ""),
    ("PhaseCompressibilityFactorsLV", ""),
    ("PhaseMixingCoefficientsLV", ""),
)
EXPECTED_COMPONENT_PROPERTIES = (("EquationOfStateComponent", ""),)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    listed = list_mrst_contents_properties()
    component_listed = list_mrst_contents_component_properties()
    catalog = list_properties()

    print("\nPhase 243: compositional property Contents.m catalog parity")
    failures = 0
    failures += _check("properties Contents.m entries match MRST order/text",
                       listed == EXPECTED_PROPERTIES)
    failures += _check("component Contents.m entries match MRST order/text",
                       component_listed == EXPECTED_COMPONENT_PROPERTIES)
    failures += _check("Contents constants are immutable tuple metadata",
                       isinstance(MRST_CONTENTS_PROPERTIES, tuple)
                       and isinstance(MRST_CONTENTS_COMPONENT_PROPERTIES, tuple))
    failures += _check("helpers return defensive copies",
                       listed == MRST_CONTENTS_PROPERTIES
                       and listed is not MRST_CONTENTS_PROPERTIES
                       and component_listed == MRST_CONTENTS_COMPONENT_PROPERTIES
                       and component_listed is not MRST_CONTENTS_COMPONENT_PROPERTIES
                       and catalog is not PROPERTIES
                       and catalog["FugacityLV"] is not PROPERTIES["FugacityLV"])
    failures += _check("all property Contents entries are represented in catalog",
                       set(catalog) == {name for name, _ in listed})
    failures += _check("MRST flow-property aggregate listing is explicit",
                       catalog["CompositionalFlowPropertyFunctions"]["status"] == "implemented_aggregate"
                       and catalog["CompositionalFlowPropertyFunctions"]["python_module"]
                       == "compositional_flow_property_functions")
    implemented = [
        meta["python_module"]
        for meta in catalog.values()
        if meta["status"] in ("implemented", "implemented_aggregate")
    ]
    failures += _check("implemented property modules import",
                       all(importlib.import_module(
                           f"pymrst.modules.compositional.properties.{module}"
                       ) is not None for module in implemented))
    failures += _check("component property catalog remains implemented/importable",
                       tuple(COMPONENT_PROPERTIES) == tuple(name for name, _ in component_listed)
                       and tuple(list_component_properties()) == ("EquationOfStateComponent",)
                       and importlib.import_module(
                           "pymrst.modules.compositional.properties.components.equation_of_state_component"
                       ) is not None)

    if failures:
        print(f"\nPhase 243 failed ({failures} checks).")
        return 1
    print("\nCompositional property catalogs now mirror MRST Contents.m, including the aggregate listing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
