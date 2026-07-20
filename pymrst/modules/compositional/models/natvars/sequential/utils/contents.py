"""Catalog for MRST compositional/models/natvars/sequential/utils ports."""

MRST_CONTENTS_NATVARS_SEQUENTIAL_UTILS = (
    ("correctSequentialBHP", "Undocumented Utility Function"),
    ("getUpdatesFromPressure", "Undocumented Utility Function"),
)

NATVARS_SEQUENTIAL_UTILS = {
    "correctSequentialBHP": {
        "python_module": "correct_sequential_bhp",
        "status": "ported",
    },
    "getUpdatesFromPressure": {
        "python_module": "get_updates_from_pressure",
        "status": "ported",
    },
}


def list_mrst_contents_natvars_sequential_utils() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's natvars/sequential/utils/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_NATVARS_SEQUENTIAL_UTILS)


def list_natvars_sequential_utils() -> dict[str, dict[str, str]]:
    """Return MRST natural-variable utility names mapped to PyMRST metadata."""
    return {name: dict(meta) for name, meta in NATVARS_SEQUENTIAL_UTILS.items()}


__all__ = [
    "MRST_CONTENTS_NATVARS_SEQUENTIAL_UTILS",
    "NATVARS_SEQUENTIAL_UTILS",
    "list_mrst_contents_natvars_sequential_utils",
    "list_natvars_sequential_utils",
]
