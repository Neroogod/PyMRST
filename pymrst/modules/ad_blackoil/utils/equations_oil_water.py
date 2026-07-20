"""MRST `ad-blackoil/utils/equationsOilWater.m` entry point."""

from pymrst.utils.adi import init_variables_adi


def equations_oil_water(state0, state, model, dt: float):
    """Assemble oil-water equations for `TwoPhaseOilWaterModelAD`.

    This utility exposes the same residual assembly that the validated
    hand-rolled oil-water model uses internally, but under the MRST-shaped
    `ad_blackoil.utils` layout.
    """
    p, sw, qws, qos, bhp = init_variables_adi(
        state.pressure, state.sW, state.qWs, state.qOs, state.bhp
    )
    return model.assemble(state0, p, sw, qws, qos, bhp, dt)


__all__ = ["equations_oil_water"]
