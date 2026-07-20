"""Catalog for MRST compositional/models/natvars/sequential ports."""

MRST_CONTENTS_NATVARS_SEQUENTIAL = (
    ("PressureNaturalVariablesModel", "Two phase oil/water system without dissolution"),
    ("PressureReducedLinearSystem", ""),
    ("transportEquationsNaturalVars", ""),
    ("TransportNaturalVariablesModel", ""),
)

NATVARS_SEQUENTIAL_FILES = {
    "PressureNaturalVariablesModel": {
        "python_module": "pressure_natural_variables_model",
        "status": "ported",
    },
    "PressureReducedLinearSystem": {
        "python_module": "pressure_reduced_linear_system",
        "status": "ported",
    },
    "transportEquationsNaturalVars": {
        "python_module": "transport_equations_natural_vars",
        "status": "ported",
    },
    "TransportNaturalVariablesModel": {
        "python_module": "transport_natural_variables_model",
        "status": "ported",
    },
}


def list_mrst_contents_natvars_sequential() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's natvars/sequential/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_NATVARS_SEQUENTIAL)


def list_natvars_sequential_files() -> dict[str, dict[str, str]]:
    """Return MRST sequential natural-variable names mapped to PyMRST metadata."""
    return {name: dict(meta) for name, meta in NATVARS_SEQUENTIAL_FILES.items()}


__all__ = [
    "MRST_CONTENTS_NATVARS_SEQUENTIAL",
    "NATVARS_SEQUENTIAL_FILES",
    "list_mrst_contents_natvars_sequential",
    "list_natvars_sequential_files",
]
