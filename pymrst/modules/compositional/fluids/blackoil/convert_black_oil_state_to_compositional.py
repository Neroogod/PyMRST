"""Port of MRST ``convertBlackOilStateToCompositional.m``."""

import numpy as np

from pymrst.modules.compositional.fluids.blackoil.black_oil_to_mass_fraction import (
    black_oil_to_mass_fraction,
)
from pymrst.modules.compositional.natural_variables import NaturalVariablesState


def _field(state, name: str):
    if isinstance(state, dict):
        return state[name]
    if hasattr(state, name):
        return getattr(state, name)
    return state[name]


def convert_black_oil_state_to_compositional(model, state) -> NaturalVariablesState:
    """Convert a black-oil state to PyMRST's natural-variable state.

    The output mirrors MRST's converted state fields: pressure, saturations,
    overall pseudo-component fractions ``components`` (``z`` in PyMRST), liquid
    composition ``x``, vapour composition ``y``, liquid molar/mass fraction
    proxy ``L``, and temperature placeholder.
    """
    p = np.asarray(_field(state, "pressure"), dtype=float).ravel()
    keys = _state_keys(state)
    if "s" in keys:
        sat = np.asarray(_field(state, "s"), dtype=float)
        s_w = sat[:, 0]
        s_o = sat[:, 1]
        s_g = sat[:, 2]
    else:
        s_w = np.asarray(_field(state, "sW"), dtype=float).ravel()
        s_o = np.asarray(_field(state, "sO"), dtype=float).ravel()
        s_g = np.asarray(_field(state, "sG"), dtype=float).ravel()
    rs = np.asarray(_field(state, "rs"), dtype=float).ravel()
    props = black_oil_to_mass_fraction(model.fluid, p, s_o, s_g, rs, disgas=getattr(model, "disgas", True))
    z = np.column_stack([props.zo, props.zg])
    x = np.column_stack([props.xo, props.xg])
    y = np.column_stack([props.yo, props.yg])
    denom = props.rhoO * s_o + props.rhoG * s_g
    L = np.where(denom > 0.0, props.rhoO * s_o / denom, 1.0)

    nw = len(model.wells)
    q_ws = np.asarray(_field(state, "qWs"), dtype=float).copy() if "qWs" in keys else np.zeros(nw)
    q_os = np.asarray(_field(state, "qOs"), dtype=float).copy() if "qOs" in keys else np.zeros(nw)
    q_gs = np.asarray(_field(state, "qGs"), dtype=float).copy() if "qGs" in keys else np.zeros(nw)
    bhp = np.asarray(_field(state, "bhp"), dtype=float).copy() if "bhp" in keys else np.zeros(nw)

    out = NaturalVariablesState(
        pressure=p.copy(),
        z=z,
        qOs=q_os,
        qGs=q_gs,
        bhp=bhp,
        sW=s_w.copy(),
        qWs=q_ws,
        sT=np.ones_like(p),
        x=x,
        y=y,
        sO=s_o.copy(),
        sG=s_g.copy(),
        flag=np.where(s_g > 0.0, 0, 1).astype(np.int8),
    )
    out.L = L
    out.T = np.full_like(p, 273.15)
    out.blackoil_mass_fractions = props
    return out


def _state_keys(state) -> set[str]:
    if isinstance(state, dict):
        return set(state.keys())
    if hasattr(state, "keys"):
        return set(state.keys())
    if hasattr(state, "__contains__"):
        names = {"pressure", "s", "rs", "qWs", "qOs", "qGs", "bhp"}
        return {name for name in names if name in state}
    return set(vars(state))


if __name__ == "__main__":
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial

    def test_convert_initial_spe1_state():
        _, model, state = setup_spe1_tutorial()
        converted = convert_black_oil_state_to_compositional(model, state)
        assert converted.z.shape == (model.G.cells.num, 2)
        assert np.allclose(converted.z.sum(axis=1), 1.0)
        assert np.allclose(converted.x.sum(axis=1), 1.0)
        assert np.allclose(converted.y, np.array([0.0, 1.0]))
        assert np.all(converted.sW > 0.0)
        print("ok test_convert_initial_spe1_state")

    test_convert_initial_spe1_state()
    print("\nAll tests passed!")
