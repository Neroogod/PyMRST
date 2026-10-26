"""Shared helpers for the black-oil PVT flow properties (MRST flowprops).

The black-oil PVT state functions (`BlackOilShrinkageFactors`,
`BlackOilViscosity`, `BlackOilDensity`) all read the same three pieces of
saturated-state information off the state - the dissolved-gas ratio ``rs``, the
vaporised-oil ratio ``rv``, and the per-cell "is the other hydrocarbon phase
present?" flags that switch a cell between saturated and under-saturated. This
module centralises that read so the three classes cannot drift apart.

The flags come from the hydrocarbon `status` primary variable (the same one
`GenericBlackOilModelAD.init_state_ad` resolves): ``status >= 2`` means free gas
is present (oil is saturated), and ``status`` odd means oil is present (gas is
saturated). This is exactly the split the pre-graph PVT statefunctions used;
factoring it here keeps the new MRST-shaped hierarchy bit-for-bit with them.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from pymrst.modules.ad_core.statefunctions.state_function import State


def pvt_args(model: Any, state: State) -> Tuple[Any, Any, Any, Any, Any]:
    """``(p, rs, gas_present, rv, oil_present)`` for the PVT closures.

    ``rs``/``rv`` are ``None`` when the model does not carry them, and the
    saturated-phase flags are ``None`` too in that case (a dead-oil /
    immiscible fluid), so ``model.fluid.b(p, rs, sat, rv, sat_g)`` reduces to
    the pressure-only evaluation.
    """
    p = state["pressure"]
    f = model.fluid
    if not (f.disgas or f.vapoil):
        return p, None, None, None, None
    status = state["status"]
    gas_present = status >= 2
    oil_present = (status % 2) == 1
    rs = state["rs"] if f.disgas else None
    rv = state["rv"] if f.vapoil else None
    return p, rs, gas_present, rv, oil_present


def phase_pressures(prop: Any, model: Any, state: State) -> List[Any]:
    """Per-phase pressures if capillary pressure is active, else p repeated.

    ``prop`` must declare a boolean ``use_phase_pressures`` and, when true, a
    dependency on ``PhasePressures``.
    """
    if getattr(prop, "use_phase_pressures", False):
        return prop.get_evaluated(state, "PhasePressures")
    return [state["pressure"]] * model.n_phases


__all__ = ["phase_pressures", "pvt_args"]


if __name__ == "__main__":
    import numpy as np

    class _Fluid:
        disgas = True
        vapoil = False

    class _Model:
        fluid = _Fluid()
        n_phases = 3

    def test_pvt_args_reads_saturated_flags():
        """status encodes the saturated flags: >=2 gas present, odd oil present."""
        st = State(pressure=np.array([1.0, 2.0, 3.0]),
                   status=np.array([1, 2, 3]),
                   rs=np.array([10.0, 20.0, 30.0]))
        p, rs, gas, rv, oil = pvt_args(_Model(), st)
        assert np.allclose(rs, [10, 20, 30])
        assert list(gas) == [False, True, True]       # status >= 2
        assert list(oil) == [True, False, True]        # status odd
        assert rv is None
        print("ok test_pvt_args_reads_saturated_flags")

    def test_pvt_args_dead_oil_is_pressure_only():
        """A fluid with neither disgas nor vapoil returns None flags."""
        class _DF:
            disgas = False
            vapoil = False

        class _DM:
            fluid = _DF()
            n_phases = 2
        p, rs, gas, rv, oil = pvt_args(_DM(), State(pressure=np.array([1.0])))
        assert rs is None and gas is None and rv is None and oil is None
        print("ok test_pvt_args_dead_oil_is_pressure_only")

    def test_phase_pressures_repeats_without_pc():
        """Without capillary pressure, every phase sees the cell pressure."""
        class _P:
            use_phase_pressures = False
        pp = phase_pressures(_P(), _Model(), State(pressure=np.array([5.0, 6.0])))
        assert len(pp) == 3 and all(np.allclose(x, [5.0, 6.0]) for x in pp)
        print("ok test_phase_pressures_repeats_without_pc")

    test_pvt_args_reads_saturated_flags()
    test_pvt_args_dead_oil_is_pressure_only()
    test_phase_pressures_repeats_without_pc()
    print("\nAll tests passed!")
