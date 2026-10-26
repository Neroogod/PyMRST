"""MRST ``ad-core/statefunctions/flux/PermeabilityGradientDiscretization.m``.

Base for the rule that turns a face potential difference into a flux driver:
`K grad(Theta)` in some discretisation. TPFA is the one MRST's black-oil
models use.
"""

from __future__ import annotations

from typing import Any


class PermeabilityGradientDiscretization:
    """Interface: `get_permeability_gradient(model, state, dp) -> face values`."""

    def get_permeability_gradient(self, model: Any, state: Any,
                                  face_potential_difference: Any) -> Any:
        return None

    getPermeabilityGradient = get_permeability_gradient


__all__ = ["PermeabilityGradientDiscretization"]


if __name__ == "__main__":
    def test_the_base_returns_nothing_as_in_mrst():
        assert PermeabilityGradientDiscretization().get_permeability_gradient(
            None, None, 1.0) is None
        print("ok test_the_base_returns_nothing_as_in_mrst")

    test_the_base_returns_nothing_as_in_mrst()
    print("\nAll tests passed!")
