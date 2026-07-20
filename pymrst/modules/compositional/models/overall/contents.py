"""Catalog for MRST compositional/models/overall ports."""

MRST_CONTENTS_OVERALL = (
    ("OverallCompositionCompositionalModel", "Overall composition model for compositional problems"),
)

OVERALL_FILES = {
    "OverallCompositionCompositionalModel": {
        "python_module": "overall_composition_compositional_model",
        "status": "ported",
    },
}


def list_mrst_contents_overall() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's models/overall/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_OVERALL)


def list_overall_files() -> dict[str, dict[str, str]]:
    """Return MRST overall-composition names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in OVERALL_FILES.items()}


__all__ = [
    "MRST_CONTENTS_OVERALL",
    "OVERALL_FILES",
    "list_mrst_contents_overall",
    "list_overall_files",
]
