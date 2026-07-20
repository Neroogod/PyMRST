"""Catalog for MRST compositional/models/eos ports."""

MRST_CONTENTS_EOS = (
    ("EquationOfStateModel", "Equation of state model. Implements generalized two-parameter cubic"),
    ("EquilibriumConstantModel", "Equilibrium constant EOS model for problems where functions of the"),
)

EOS_MODELS = {
    "EquationOfStateModel": {
        "python_module": "equation_of_state_model",
        "status": "ported",
    },
    "EquilibriumConstantModel": {
        "python_module": "equilibrium_constant_model",
        "status": "ported",
    },
}


def list_mrst_contents_eos() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's models/eos/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_EOS)


def list_eos_models() -> dict[str, dict[str, str]]:
    """Return MRST EOS model names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in EOS_MODELS.items()}


__all__ = ["EOS_MODELS", "MRST_CONTENTS_EOS", "list_eos_models", "list_mrst_contents_eos"]
