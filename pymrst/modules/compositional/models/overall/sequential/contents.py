"""Catalog for MRST compositional/models/overall/sequential ports."""

MRST_CONTENTS_OVERALL_SEQUENTIAL = (
    ("PressureCompositionalModel", "Two phase oil/water system without dissolution"),
    ("pressureEquationCompositional", "Undocumented Utility Function"),
    ("PressureOverallCompositionModel", "Two phase oil/water system without dissolution"),
    ("TransportCompositionalModel", "Two phase oil/water system without dissolution"),
    ("transportEquationCompositional", "Undocumented Utility Function"),
    ("transportEquationOverallComposition", ""),
    ("TransportOverallCompositionModel", "Two phase oil/water system without dissolution"),
)

OVERALL_SEQUENTIAL_FILES = {
    "PressureCompositionalModel": {
        "python_module": "pressure_compositional_model",
        "status": "ported",
    },
    "pressureEquationCompositional": {
        "python_module": "pressure_equation_compositional",
        "status": "ported",
    },
    "PressureOverallCompositionModel": {
        "python_module": "pressure_overall_composition_model",
        "status": "ported",
    },
    "TransportCompositionalModel": {
        "python_module": "transport_compositional_model",
        "status": "ported",
    },
    "transportEquationCompositional": {
        "python_module": "transport_equation_compositional",
        "status": "ported",
    },
    "transportEquationOverallComposition": {
        "python_module": "transport_equation_overall_composition",
        "status": "ported",
    },
    "TransportOverallCompositionModel": {
        "python_module": "transport_overall_composition_model",
        "status": "ported",
    },
}


def list_mrst_contents_overall_sequential() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's overall/sequential/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_OVERALL_SEQUENTIAL)


def list_overall_sequential_files() -> dict[str, dict[str, str]]:
    """Return MRST sequential overall-composition names mapped to PyMRST metadata."""
    return {name: dict(meta) for name, meta in OVERALL_SEQUENTIAL_FILES.items()}


__all__ = [
    "MRST_CONTENTS_OVERALL_SEQUENTIAL",
    "OVERALL_SEQUENTIAL_FILES",
    "list_mrst_contents_overall_sequential",
    "list_overall_sequential_files",
]
