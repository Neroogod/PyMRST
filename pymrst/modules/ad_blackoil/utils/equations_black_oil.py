"""MRST `ad-blackoil/utils/equationsBlackOil.m` entry point."""

from pymrst.utils.adi import init_variables_adi


def equations_black_oil(state0, state, model, dt: float):
    """Assemble three-phase black-oil equations for `ThreePhaseBlackOilModelAD`.

    The hydrocarbon primary variable follows MRST's status convention: `rs`
    in undersaturated oil cells and `sG` in saturated cells.
    """
    st1 = state.status == 1
    x = state.rs * st1.astype(float) + state.s[:, 2] * (~st1).astype(float)
    p, sw, x, qws, qos, qgs, bhp = init_variables_adi(
        state.pressure, state.s[:, 0], x,
        state.qWs, state.qOs, state.qGs, state.bhp,
    )
    return model.assemble(state0, state, p, sw, x, qws, qos, qgs, bhp, dt)


__all__ = ["equations_black_oil"]
