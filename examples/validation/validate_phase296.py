"""Phase 296: MRST fluids/PropertyModel base class."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.fluids import (  # noqa: E402
    CompositionalPropertyModel,
    PropertyModel,
    list_fluids,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 296: MRST fluids/PropertyModel base class")
    mix = simple_mixture()
    base = PropertyModel(mix)
    alias = PropertyModel(f=mix)
    prop = CompositionalPropertyModel(mix)
    meta = list_fluids()["PropertyModel"]
    failures = 0
    failures += _check("PropertyModel stores MRST fluid property",
                       base.fluid is mix and alias.fluid is mix)
    failures += _check("PropertyModel exposes pythonic mixture alias",
                       base.mixture is mix)
    failures += _check("mixture alias writes back to fluid",
                       _mutates_mixture_alias(base))
    failures += _check("CompositionalPropertyModel inherits PropertyModel",
                       isinstance(prop, PropertyModel) and prop.fluid is mix)
    failures += _check("fluids contents metadata points at property_model",
                       meta["python_module"] == "property_model"
                       and meta["status"] == "implemented_base_surface")
    if failures:
        print(f"\nPhase 296 failed ({failures} checks).")
        return 1
    print("\nMRST fluids/PropertyModel is available as a real base class.")
    return 0


def _mutates_mixture_alias(model: PropertyModel) -> bool:
    old = model.fluid
    marker = object()
    model.mixture = marker
    ok = model.fluid is marker
    model.mixture = old
    return ok and model.fluid is old


if __name__ == "__main__":
    raise SystemExit(main())
