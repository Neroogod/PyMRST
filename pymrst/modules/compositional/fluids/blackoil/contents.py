"""Catalog for MRST compositional/fluids/blackoil ports."""

MRST_CONTENTS_BLACKOIL = (
    ("blackOilToMassFraction", "Evaluate properties"),
    ("convertBlackOilModelToCompositionalModel", "Convert blackoil model to compositional-like model"),
    ("convertBlackOilStateToCompositional", "Convert BO state to compositional-like state"),
    ("convertRSRVtoKValue", "Convert RS/RV blackoil to K-value. Experimental!"),
)

BLACKOIL_FILES = {
    "blackOilToMassFraction": {
        "python_module": "black_oil_to_mass_fraction",
        "status": "ported",
    },
    "convertBlackOilModelToCompositionalModel": {
        "python_module": "convert_black_oil_model_to_compositional_model",
        "status": "ported",
    },
    "convertBlackOilStateToCompositional": {
        "python_module": "convert_black_oil_state_to_compositional",
        "status": "ported",
    },
    "convertRSRVtoKValue": {
        "python_module": "convert_rsrv_to_k_value",
        "status": "ported_experimental",
    },
}


def list_mrst_contents_blackoil() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's fluids/blackoil/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_BLACKOIL)


def list_blackoil_files() -> dict[str, dict[str, str]]:
    """Return MRST blackoil utility names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in BLACKOIL_FILES.items()}


__all__ = [
    "BLACKOIL_FILES",
    "MRST_CONTENTS_BLACKOIL",
    "list_blackoil_files",
    "list_mrst_contents_blackoil",
]
