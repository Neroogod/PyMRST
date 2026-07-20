"""Catalog for MRST-style compositional component property files."""

MRST_CONTENTS_COMPONENT_PROPERTIES = (
    ("EquationOfStateComponent", ""),
)

COMPONENT_PROPERTIES = {
    "EquationOfStateComponent": {
        "mrst_file": "properties/components/EquationOfStateComponent.m",
        "python_module": "equation_of_state_component",
        "summary": "EOS component object exposing phase composition and component density.",
    },
}


def list_mrst_contents_component_properties() -> tuple[tuple[str, str], ...]:
    """Return the exact component-property entries listed by MRST's Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_COMPONENT_PROPERTIES)


def list_component_properties() -> tuple[str, ...]:
    """Return implemented component-property names."""
    return tuple(COMPONENT_PROPERTIES)


if __name__ == "__main__":
    assert list_mrst_contents_component_properties() == (("EquationOfStateComponent", ""),)
    assert "EquationOfStateComponent" in list_component_properties()
