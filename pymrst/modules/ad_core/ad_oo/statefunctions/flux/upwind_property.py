"""MRST ``ad-core/statefunctions/flux/upwind/UpwindProperty.m``.

Mixin for a face property that is upwinded through a pluggable
`UpwindDiscretization`.
"""

from __future__ import annotations

from typing import Any


class UpwindProperty:
    """Holds an `UpwindDiscretization` and delegates `face_upstream` to it."""

    def init_upwind_property(self, upstream: Any) -> None:
        self.UpwindDiscretization = upstream

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cellvalue: Any) -> Any:
        return self.UpwindDiscretization.face_upstream(model, state, flag,
                                                       cellvalue)

    faceUpstream = face_upstream


__all__ = ["UpwindProperty"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_upwinding_goes_through_the_discretization():
        prop = UpwindProperty()
        prop.init_upwind_property(SimpleNamespace(
            face_upstream=lambda model, state, flag, x: ("up", flag, x)))
        assert prop.face_upstream(None, None, True, 3) == ("up", True, 3)
        print("ok test_upwinding_goes_through_the_discretization")

    test_upwinding_goes_through_the_discretization()
    print("\nAll tests passed!")
