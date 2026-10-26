"""Per-phase minimum saturations for equilibrium fill (MRST
`getMinimumPhaseSaturations`).

The saturation each phase is held at where it is *not* the phase filling the
cell: connate water everywhere, zero residual oil, and - the part that is easy
to miss - gas at `1 - swcon` rather than zero.

That gas value is not a typo in MRST and not a "minimum" in the ordinary sense.
It is the fill target for a gas cap: the most gas a cell can hold once connate
water has taken its share. Reading it as a floor and setting it to zero gives a
gas cap that initialises with no gas at all above the contact, which is a
plausible-looking state and a wrong one.

**A divergence, recorded rather than silently fixed.** PyMRST's equilibrium
fill uses `equilibrium._connate_saturations`, which sets water to `swcon` and
every other phase to zero - so for three-phase models it does *not* apply the
gas rule above. This module is MRST's function, ported faithfully; it is not
wired into the fill, because doing so changes three-phase equilibrium
initialisation and the characterization gate does not cover a three-phase
equilibrium case that would catch a mistake.

`test_the_fill_helper_diverges_on_gas` pins the difference so it stays visible
and someone can decide it deliberately, rather than discovering it from a gas
cap that starts empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_minimum_phase_saturations(model: Any) -> np.ndarray:
    """`s_min` per active phase, in the model's phase order.

    Returns one entry per *active* phase, so a two-phase oil-water model gets
    two - indexed through `get_phase_index`, never positionally, because the
    active set is what determines the layout and a three-phase formula applied
    to a two-phase array writes gas into the oil slot.
    """
    names = [str(n).upper() for n in model.get_phase_names()]
    s_min = np.zeros(len(names))

    fluid = getattr(model, "raw_fluid", None) or model.fluid
    sw_min = 0.0
    if "W" in names:
        connate = getattr(fluid, "swcon", None)
        if connate is None:
            connate = getattr(fluid, "sWcon", 0.0)
        sw_min = float(np.ravel(connate)[0]) if connate is not None else 0.0
        s_min[model.get_phase_index("W")] = sw_min

    if "O" in names:
        # Explicit, though the array is already zero: MRST writes it, and the
        # zero is a statement (oil carries no residual here) rather than an
        # unfilled slot.
        s_min[model.get_phase_index("O")] = 0.0

    if "G" in names:
        # The fill target for a gas cap, not a floor. See the module docstring.
        s_min[model.get_phase_index("G")] = 1.0 - sw_min

    return s_min


getMinimumPhaseSaturations = get_minimum_phase_saturations

__all__ = ["get_minimum_phase_saturations", "getMinimumPhaseSaturations"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(phases, swcon=0.12):
        names = list(phases)
        return SimpleNamespace(
            fluid=SimpleNamespace(swcon=swcon), raw_fluid=None,
            n_phases=len(names),
            get_phase_names=lambda: names,
            get_phase_index=lambda p: names.index(p))

    def test_water_takes_the_connate_saturation():
        s = get_minimum_phase_saturations(_model(["W", "O"]))
        assert s[0] == 0.12 and s[1] == 0.0, s
        print("ok test_water_takes_the_connate_saturation")

    def test_gas_takes_one_minus_connate_water():
        """The rule that is easy to read as a typo. A gas cap fills to
        whatever connate water leaves it, not to zero."""
        s = get_minimum_phase_saturations(_model(["W", "O", "G"]))
        assert s[2] == 1.0 - 0.12, s
        print(f"ok test_gas_takes_one_minus_connate_water (s_g = {s[2]:.2f})")

    def test_a_fluid_without_connate_water_gives_zero():
        model = _model(["W", "O", "G"], swcon=0.0)
        s = get_minimum_phase_saturations(model)
        assert s[0] == 0.0 and s[2] == 1.0
        print("ok test_a_fluid_without_connate_water_gives_zero")

    def test_phases_are_placed_by_index_not_position():
        """A model listing gas before oil must still get gas's value in gas's
        slot; positional writing would put it in oil's."""
        s = get_minimum_phase_saturations(_model(["W", "G", "O"]))
        assert s[1] == 1.0 - 0.12 and s[2] == 0.0, s
        print("ok test_phases_are_placed_by_index_not_position")

    def test_the_fill_helper_diverges_on_gas():
        """Pins the recorded divergence: `_connate_saturations` gives gas zero
        where this gives `1 - swcon`. Kept visible so it is decided, not
        discovered."""
        from pymrst.modules.ad_core.initialization.equilibrium import (
            EquilibriumRegion, _connate_saturations,
        )
        model = _model(["W", "O", "G"])
        region = EquilibriumRegion.__new__(EquilibriumRegion)
        region.s_min = None
        fill = _connate_saturations(model, region)
        mrst = get_minimum_phase_saturations(model)
        assert fill[0] == mrst[0], "water agrees"
        assert fill[2] == 0.0 and mrst[2] > 0.0, (fill, mrst)
        print(f"ok test_the_fill_helper_diverges_on_gas "
              f"(fill s_g={fill[2]:.2f}, MRST s_g={mrst[2]:.2f})")

    test_water_takes_the_connate_saturation()
    test_gas_takes_one_minus_connate_water()
    test_a_fluid_without_connate_water_gives_zero()
    test_phases_are_placed_by_index_not_position()
    test_the_fill_helper_diverges_on_gas()
    print("\nAll tests passed!")
