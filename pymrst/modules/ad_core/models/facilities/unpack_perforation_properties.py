"""MRST ``ad-core/models/facilities/unpackPerforationProperties.m``: the
groups stored by `packPerforationProperties` - pressure, mobilities,
densities, dissolution, components and the well's extra variables."""

from __future__ import annotations

from typing import Any


def _get(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def unpack_perforation_properties(packed):
    """Return the six property groups stored by the packing helper."""
    return (
        _get(packed, "pressure"),
        _get(packed, "mob"),
        _get(packed, "rho"),
        _get(packed, "dissolved"),
        _get(packed, "components"),
        _get(packed, "extravars"),
    )


unpackPerforationProperties = unpack_perforation_properties
__all__ = ["unpackPerforationProperties", "unpack_perforation_properties"]


if __name__ == "__main__":
    def test_the_six_groups_in_mrsts_order():
        packed = {"pressure": 1, "mob": 2, "rho": 3, "dissolved": 4, "components": 5, "extravars": 6}
        assert unpack_perforation_properties(packed) == (1, 2, 3, 4, 5, 6)
        print("ok test_the_six_groups_in_mrsts_order")

    test_the_six_groups_in_mrsts_order()
    print("\nAll tests passed!")
