"""Catalog for MRST compositional fluid ports."""

MRST_CONTENTS_FLUIDS = (
    ("BlackOilPropertyModel", "Black-oil property model for compositional models"),
    ("CompositionalMixture", ""),
    ("CompositionalPropertyModel", "Property model for compositional models"),
    ("CoolPropsCompositionalMixture", "Create MRST compositional fluid by using the open CoolProp library"),
    ("getBenchmarkMixture", "Grab bag of different fluids for use in examples etc"),
    ("PropertyModel", "Base class for properties in the compositional module"),
    ("TableCompositionalMixture", "Create MRST compositional from stored tables (taken from CoolProp)"),
)

FLUIDS = {
    "BlackOilPropertyModel": {
        "python_module": "black_oil_property_model",
        "status": "implemented",
        "summary": "MRST-style black-oil property model with function-backed and K-value table-backed density/viscosity surfaces.",
    },
    "CompositionalMixture": {
        "python_module": "compositional_mixture",
        "status": "implemented",
        "summary": "MRST-style mixture/EOS component data surface backed by PyMRST Mixture.",
    },
    "CompositionalPropertyModel": {
        "python_module": "compositional_property_model",
        "status": "implemented",
        "summary": "EOS property model for density, molar density, viscosity and volume-shift surfaces.",
    },
    "CoolPropsCompositionalMixture": {
        "python_module": "coolprops_compositional_mixture",
        "status": "implemented_optional_dependency",
        "summary": "MRST CoolProp-backed fluid constructor; intentionally separate from native table/benchmark fluids.",
    },
    "getBenchmarkMixture": {
        "python_module": "get_benchmark_mixture",
        "status": "implemented",
        "summary": "Benchmark compositional mixtures used by examples.",
    },
    "PropertyModel": {
        "python_module": "property_model",
        "status": "implemented_base_surface",
        "summary": "MRST base property model storing the associated compositional fluid.",
    },
    "TableCompositionalMixture": {
        "python_module": "table_compositional_mixture",
        "status": "implemented",
        "summary": "MRST table-backed CoolProp fluid constructor; no native table database is present yet.",
    },
}


def list_mrst_contents_fluids() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's fluids/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_FLUIDS)


def list_fluids() -> dict[str, dict[str, str | None]]:
    """Return MRST fluid names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in FLUIDS.items()}


__all__ = ["FLUIDS", "MRST_CONTENTS_FLUIDS", "list_fluids", "list_mrst_contents_fluids"]
