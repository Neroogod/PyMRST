"""Catalog for MRST compositional model ports."""

MRST_CONTENTS_MODELS = (
    ("GenericNaturalVariablesModel", ""),
    ("GenericOverallCompositionModel", ""),
    ("ThreePhaseCompositionalModel", "Base class for compositional models"),
)

MODELS = {
    "GenericNaturalVariablesModel": {
        "python_module": "generic_natural_variables_model",
        "status": "implemented_core_surface",
        "summary": "Natural-variable solver with MRST-style phase flags, primary-variable surface, state validation and flash refresh helpers.",
    },
    "GenericOverallCompositionModel": {
        "python_module": "generic_overall_composition_model",
        "status": "implemented_generic_surface",
        "summary": "Overall-composition model with MRST generic component names, output-state functions and primary-variable helpers.",
    },
    "ThreePhaseCompositionalModel": {
        "python_module": "three_phase_compositional_model",
        "status": "implemented_base_surface",
        "summary": "MRST base-model utility surface for EOS ownership, phase/component names, variable-field lookup, validation and flash helpers.",
    },
}


def list_mrst_contents_models() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's models/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_MODELS)


def list_models() -> dict[str, dict[str, str]]:
    """Return MRST model names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in MODELS.items()}


__all__ = ["MODELS", "MRST_CONTENTS_MODELS", "list_models", "list_mrst_contents_models"]
