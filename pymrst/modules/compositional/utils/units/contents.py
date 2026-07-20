"""Catalog for MRST compositional unit-helper ports."""

MRST_CONTENTS_UNITS = (
    ("fromCelsius", ""),
    ("fromFahrenheit", ""),
)

UNITS = {
    "fromCelsius": {"python_module": "from_celsius", "status": "implemented"},
    "fromFahrenheit": {"python_module": "from_fahrenheit", "status": "implemented"},
}


def list_mrst_contents_units() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's utils/units/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_UNITS)


def list_units() -> dict[str, dict[str, str]]:
    """Return MRST compositional unit helpers mapped to PyMRST modules."""
    return {name: dict(meta) for name, meta in UNITS.items()}


__all__ = ["MRST_CONTENTS_UNITS", "UNITS", "list_mrst_contents_units", "list_units"]
