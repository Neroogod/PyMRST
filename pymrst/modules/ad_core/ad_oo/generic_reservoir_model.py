"""MRST ``ad-core/models/GenericReservoirModel.m``.

An empty marker class. MRST's generic models inherit it so that code can ask
`isa(model, 'GenericReservoirModel')` - which `ReservoirModel` does to pick the
generic facility and to treat the conservation equations as mass rather than
surface volume. This port answers the same question through the class flag
`isGenericReservoirModel`, which the marker sets.
"""

from __future__ import annotations


class GenericReservoirModel:
    """Marker: the conservation equations are written in component mass."""

    isGenericReservoirModel = True


__all__ = ["GenericReservoirModel"]


if __name__ == "__main__":
    def test_the_marker_sets_the_flag():
        assert GenericReservoirModel.isGenericReservoirModel is True
        print("ok test_the_marker_sets_the_flag")

    test_the_marker_sets_the_flag()
    print("\nAll tests passed!")
