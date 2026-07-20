"""Catalog for MRST compositional/models/natvars ports."""

MRST_CONTENTS_NATVARS = (
    ("equationsNaturalVariables", "Equations for natural variables formulation"),
    ("NaturalVariablesCompositionalModel", "Natural variables model for compositional problems"),
    ("ReducedLinearizedSystem", "Linearized system which performs a Schur complement before solving"),
)

NATVARS_FILES = {
    "equationsNaturalVariables": {
        "python_module": "equations_natural_variables",
        "status": "ported",
    },
    "NaturalVariablesCompositionalModel": {
        "python_module": "natural_variables_compositional_model",
        "status": "ported",
    },
    "ReducedLinearizedSystem": {
        "python_module": "reduced_linearized_system",
        "status": "ported",
    },
}


def list_mrst_contents_natvars() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's models/natvars/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_NATVARS)


def list_natvars_files() -> dict[str, dict[str, str]]:
    """Return MRST natural-variable names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in NATVARS_FILES.items()}


__all__ = [
    "MRST_CONTENTS_NATVARS",
    "NATVARS_FILES",
    "list_mrst_contents_natvars",
    "list_natvars_files",
]
