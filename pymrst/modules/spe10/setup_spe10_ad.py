"""MRST ``spe10/setupSPE10_AD.m``.

SPE10 Model 2 (or a box of it: layers `layers`, columns `I`, rows `J`, 1-based)
as a `GenericBlackOilModel` with water and oil: Corey relative permeabilities
of exponent 2 with residual saturations 0.2 (`coreyPhaseRelpermAD`), slightly
compressible water, oil `B` and viscosity from a pressure table
(`tableByPressureLinearAD`), rock compressibility 1e-6/psia about 6000 psia,
and gravity. Porosities below `minporo` are raised to it. Four producers at
4000 psia in the corner columns and one water injector in the middle column,
at 5000 stb/day scaled by the subset's share of the field's pore volume.
Steps of `dt` over `T`, ramped up in `rampup` steps. With `make2D` (one layer
only) the grid is 2D with the x-permeability. The model's minimum pressure is
1000 psia. Returns `(state, model, schedule)`.

MRST computes the geometry with `mcomputeGeometry` when its MEX is built and
`computeGeometry` otherwise; PyMRST uses `computeGeometry`.

`setup_spe10_ad_ad` (in `ad_blackoil/examples/spe10/setup_spe10_ad.py`) is
PyMRST's earlier version on the legacy models, kept for its callers.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.timesteps.rampup_timesteps import rampup_timesteps
from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule
from pymrst.modules.ad_props.simple.corey_phase_relperm_ad import corey_phase_relperm_ad
from pymrst.modules.ad_props.simple.table_by_pressure_linear_ad import (
    table_by_pressure_linear_ad,
)
from pymrst.modules.spe10.get_spe10_rock import get_spe10_rock
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.adi import ad_exp
from pymrst.utils.matlab_sum import matlab_sum
from pymrst.utils.units import CENTI, DAY, FT, INCH, POISE, POUND, PSIA, STB

FIELD_PORE_VOLUME = 1.9141e+05


def setup_spe10_ad(*, layers: Any = None, dt: float = 30*DAY, T: float = 2000*DAY,
                   rampup: int = 8, I: Any = None, J: Any = None, make2D: bool = False,
                   minporo: float = 0.01) -> Tuple[Any, Any, Any]:
    """MRST `[state, model, schedule] = setupSPE10_AD(...)`."""
    layers = np.arange(1, 86) if layers is None else np.atleast_1d(layers)
    I = np.arange(1, 61) if I is None else np.atleast_1d(I)
    J = np.arange(1, 221) if J is None else np.atleast_1d(J)
    srw, sro = 0.2, 0.2
    pRef = 6000*PSIA
    bW0 = 1/1.01
    p = np.array([300, 800, 8000, 8001])*PSIA
    b = 1/np.array([1.05, 1.02, 1.01, 1.01])
    mu = np.array([2.85, 2.99, 3, 3])*CENTI*POISE
    bO, muO = table_by_pressure_linear_ad(p, b, mu)
    cR = 1e-6/PSIA
    fluid = {
        "krW": corey_phase_relperm_ad(2, srw, 1, srw + sro),
        "krO": corey_phase_relperm_ad(2, sro, 1, srw + sro),
        "bW": lambda p, *varargin: bW0*ad_exp((p - pRef)*3e-6/PSIA),
        "muW": lambda p, *varargin: 0*p + 0.3*CENTI*POISE,
        "bO": bO,
        "muO": muO,
        "rhoWS": 64*POUND/FT**3,
        "rhoOS": 53*POUND/FT**3,
        "cR": cR,
        "pvMultR": lambda p: 1 + cR*(p - pRef),
    }
    raw = get_spe10_rock(I, J, layers)
    if raw is None:
        raise FileNotFoundError("SPE10 model 2 dataset (spe10_rock.mat) not found")
    poro = np.asarray(raw.poro, dtype=float).copy()
    perm = np.asarray(raw.perm, dtype=float)
    volFrac = matlab_sum(poro)/FIELD_PORE_VOLUME
    poro[poro < minporo] = minporo
    nl, i, j = layers.size, I.size, J.size
    assert 0 < i <= 60
    assert 0 < j <= 220
    cartDims = [i, j, nl]
    physDims = list(np.array(cartDims)*np.array([20, 10, 2])*FT)
    if make2D:
        assert nl == 1
        cartDims, physDims = cartDims[:2], physDims[:2]
        perm = perm[:, :1]
    G = compute_geometry(cart_grid(cartDims, physDims))
    rock = make_rock(G, perm, poro)
    model = GenericBlackOilModel(G, rock, fluid, gravity=np.array([0, 0, 9.80665]),
                                 water=True, oil=True, gas=False)
    model.minimumPressure = 1000*PSIA
    state = init_res_sol(G, pRef, np.array([srw, 1 - srw]))

    def make_prod(W, name, I_, J_):
        return vertical_well(W, G, rock, I_ - 1, J_ - 1, None, name=name, radius=5*INCH,
                             sign=-1, well_type="bhp", val=4000*PSIA,
                             compi=np.array([0.5, 0.5]))

    nI, nJ = G.cartDims[0], G.cartDims[1]
    W = None
    W = make_prod(W, "P1", 1, 1)
    W = make_prod(W, "P2", nI, 1)
    W = make_prod(W, "P3", nI, nJ)
    W = make_prod(W, "P4", 1, nJ)
    W = vertical_well(W, G, rock, int(np.ceil(nI/2)) - 1, int(np.ceil(nJ/2)) - 1, None,
                      name="I1", radius=5*INCH, well_type="rate",
                      val=volFrac*5000*STB/DAY, compi=np.array([1.0, 0.0]), sign=1)
    steps = rampup_timesteps(T, dt, rampup)
    schedule = simple_schedule(steps, W=W)
    return state, model, schedule


setupSPE10_AD = setup_spe10_ad

__all__ = ["setupSPE10_AD", "setup_spe10_ad"]


if __name__ == "__main__":
    def test_a_2d_layer():
        state, model, schedule = setup_spe10_ad(layers=1, J=np.arange(1, 111), make2D=True)
        assert model.G.griddim == 2 and tuple(model.G.cartDims) == (60, 110)
        assert [w.name for w in schedule.controls[0].W] == ["P1", "P2", "P3", "P4", "I1"]
        assert model.minimumPressure == 1000*PSIA
        assert np.allclose(np.asarray(state.s)[0], [0.2, 0.8])
        assert schedule.step_val.size == rampup_timesteps(2000*DAY, 30*DAY, 8).size
        print("ok test_a_2d_layer")

    test_a_2d_layer()
    print("\nAll tests passed!")
