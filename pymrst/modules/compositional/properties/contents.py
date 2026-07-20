"""Catalog for MRST compositional property state-function ports."""

MRST_CONTENTS_PROPERTIES = (
    ("ComponentPhaseMassFractionsLV", ""),
    ("ComponentPhaseMoleFractionsLV", ""),
    ("CompositionalDensity", ""),
    ("CompositionalFlowPropertyFunctions", ""),
    ("CompositionalViscosityLV", ""),
    ("FugacityLV", ""),
    ("PhaseCompressibilityFactorsLV", ""),
    ("PhaseMixingCoefficientsLV", ""),
)

PROPERTIES = {
    "ComponentPhaseMassFractionsLV": {
        "summary": "Mole-to-mass fractions for liquid/vapour phases.",
        "python_module": "component_phase_mass_fractions_lv",
        "status": "implemented",
    },
    "ComponentPhaseMoleFractionsLV": {
        "summary": "Liquid/vapour mole fractions from flash state.",
        "python_module": "component_phase_mole_fractions_lv",
        "status": "implemented",
    },
    "CompositionalDensity": {
        "summary": "EOS-derived liquid/vapour mass densities.",
        "python_module": "compositional_density",
        "status": "implemented",
    },
    "CompositionalFlowPropertyFunctions": {
        "summary": "MRST Contents-name aggregate over the implemented liquid/vapour compositional flow-property functions.",
        "python_module": "compositional_flow_property_functions",
        "status": "implemented_aggregate",
    },
    "CompositionalViscosityLV": {
        "summary": "Lohrenz-Bray-Clark liquid/vapour viscosities.",
        "python_module": "compositional_viscosity_lv",
        "status": "implemented",
    },
    "FugacityLV": {
        "summary": "Liquid/vapour fugacity coefficients and fugacity values.",
        "python_module": "fugacity_lv",
        "status": "implemented",
    },
    "PhaseCompressibilityFactorsLV": {
        "summary": "Liquid/vapour Peng-Robinson Z-factors.",
        "python_module": "phase_compressibility_factors_lv",
        "status": "implemented",
    },
    "PhaseMixingCoefficientsLV": {
        "summary": "Peng-Robinson phase mixing coefficients.",
        "python_module": "phase_mixing_coefficients_lv",
        "status": "implemented",
    },
}


def list_mrst_contents_properties() -> tuple[tuple[str, str], ...]:
    """Return the exact property entries listed by MRST's Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_PROPERTIES)


def list_properties() -> dict[str, dict[str, str | None]]:
    """Return the MRST-name to PyMRST-status property catalog."""
    return {k: dict(v) for k, v in PROPERTIES.items()}


if __name__ == "__main__":
    def test_catalog_contains_mrst_property_names():
        assert [name for name, _ in MRST_CONTENTS_PROPERTIES] == [
            "ComponentPhaseMassFractionsLV",
            "ComponentPhaseMoleFractionsLV",
            "CompositionalDensity",
            "CompositionalFlowPropertyFunctions",
            "CompositionalViscosityLV",
            "FugacityLV",
            "PhaseCompressibilityFactorsLV",
            "PhaseMixingCoefficientsLV",
        ]
        assert "CompositionalDensity" in PROPERTIES
        assert "CompositionalViscosityLV" in PROPERTIES
        assert "FugacityLV" in PROPERTIES
        assert PROPERTIES["CompositionalFlowPropertyFunctions"]["status"] == "implemented_aggregate"
        print("ok test_catalog_contains_mrst_property_names")

    test_catalog_contains_mrst_property_names()
    print("\nAll tests passed!")


__all__ = [
    "MRST_CONTENTS_PROPERTIES",
    "PROPERTIES",
    "list_mrst_contents_properties",
    "list_properties",
]
