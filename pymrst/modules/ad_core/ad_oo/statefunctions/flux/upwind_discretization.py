"""MRST ``ad-core/statefunctions/flux/upwind/UpwindDiscretization.m``.

Base for the rule that picks a face value from the two cells' values given an
upwind flag.
"""

from __future__ import annotations

from typing import Any


class UpwindDiscretization:
    """Interface: `face_upstream(model, state, flag, cellvalues)`."""

    def __init__(self, model: Any = None):
        pass

    def face_upstream(self, model: Any, state: Any, flag: Any,
                      cellvalues: Any) -> Any:
        raise NotImplementedError("Base class not intended for direct usage")

    faceUpstream = face_upstream


__all__ = ["UpwindDiscretization"]


if __name__ == "__main__":
    def test_the_base_refuses_to_upwind():
        try:
            UpwindDiscretization().face_upstream(None, None, None, None)
            raise SystemExit("should have raised")
        except NotImplementedError:
            pass
        print("ok test_the_base_refuses_to_upwind")

    test_the_base_refuses_to_upwind()
    print("\nAll tests passed!")
