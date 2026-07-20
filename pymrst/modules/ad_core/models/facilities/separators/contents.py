"""Catalog for MRST separator facility model ports."""

MRST_CONTENTS_SEPARATORS = (
    ("BasicSeparator", ""),
    ("EOSSeparator", ""),
    ("SeparatorGroup", ""),
)

SEPARATORS = {
    "BasicSeparator": {
        "python_module": "basic_separator",
        "status": "implemented",
        "summary": "Surface separator using component mass fractions and pure phase densities.",
    },
    "EOSSeparator": {
        "python_module": "eos_separator",
        "status": "implemented",
        "summary": "EOS flash separator for component mole streams.",
    },
    "SeparatorGroup": {
        "python_module": "separator_group",
        "status": "implemented",
        "summary": "MRST-style stock-tank/intermediate separator tree.",
    },
}


def list_mrst_contents_separators() -> tuple[tuple[str, str], ...]:
    """Return the exact separator entries listed by MRST's Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_SEPARATORS)


def list_separators() -> dict[str, dict[str, str]]:
    """Return MRST separator names mapped to PyMRST implementation metadata."""
    return {k: dict(v) for k, v in SEPARATORS.items()}


if __name__ == "__main__":
    assert list_mrst_contents_separators() == (
        ("BasicSeparator", ""),
        ("EOSSeparator", ""),
        ("SeparatorGroup", ""),
    )
    assert set(list_separators()) == {"BasicSeparator", "EOSSeparator", "SeparatorGroup"}


__all__ = [
    "MRST_CONTENTS_SEPARATORS",
    "SEPARATORS",
    "list_mrst_contents_separators",
    "list_separators",
]
