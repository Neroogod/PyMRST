"""Hysteretic relative permeability (MRST flowprops/hysteresis).

Ports MRST's `HystereticRelativePermeability.m` (AC8): relative permeability
with non-wetting (gas) hysteresis. On drainage - gas saturation at its historic
maximum - it is the drainage curve, identical to `BaseRelativePermeability`. On
imbibition - gas retreating below its historic maximum ``sGMax`` - the gas is
partly trapped and its relperm follows a scanning curve *below* the drainage
one, which is what makes residual trapping in a CO2 or WAG process irreversible.

This is `BaseRelativePermeability` with the gas hook (`evaluate_gas_rel_perm`)
overridden to apply PyMRST's `KilloughGasHysteresis` scanning curve against the
state's ``sGMax``. It needed the relperm hierarchy (`BaseRelativePermeability`)
to extend, which is why it lands with it. The scanning curve is evaluated per
cell numerically (as MRST's is), so the hysteretic gas relperm carries no AD
derivative on the imbibition cells; the drainage cells keep theirs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.flowprops.base_relative_permeability import (  # noqa: E501
    BaseRelativePermeability,
)
from pymrst.modules.ad_core.statefunctions.state_function import STATE
from pymrst.utils.adi import value


class HystereticRelativePermeability(BaseRelativePermeability):
    """Relative permeability with Killough gas hysteresis on imbibition."""

    def __init__(self, model: Any = None, hysteresis: Any = None,
                 sat_max_key: str = "sGMax"):
        super().__init__(model)
        self.hysteresis = hysteresis
        self.sat_max_key = sat_max_key
        self.depends_on(sat_max_key, STATE)

    def evaluate_gas_rel_perm(self, model: Any, sg: Any, state: Any) -> Any:
        if self.hysteresis is None or self.sat_max_key not in state:
            return super().evaluate_gas_rel_perm(model, sg, state)
        sg_max = state[self.sat_max_key]
        return self.hysteresis(sg, sg_max)


__all__ = ["HystereticRelativePermeability"]


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel, initial_state_from_deck,
    )
    from pymrst.modules.ad_core.statefunctions.hysteresis import (
        KilloughGasHysteresis,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.params import NORM_GRAVITY, add_well, make_rock

    DECK = Path(__file__).resolve().parents[7] / \
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"

    def _spe1():
        deck = read_spe1_deck(DECK)
        g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
        rock = make_rock(g, deck["perm"], deck["poro"][:, None])
        fluid = init_deck_blackoil_fluid(deck)
        wells = None
        for name in ("PRODUCER", "INJECTOR"):
            wd = deck["wells"][name]
            wells = add_well(wells, g, rock, wd["cells"], well_type=wd["type"],
                             val=wd["val"], radius=wd["radius"],
                             compi=wd["compi"], sign=wd["sign"], name=name,
                             ref_depth=wd["ref_depth"],
                             gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
        model = GenericBlackOilModel(g, rock, fluid, wells, disgas=True,
                                     gravity=NORM_GRAVITY)
        return model, deck, g

    def _killough(model):
        raw = model.raw_fluid
        # a real trapped-gas endpoint (sg_tmax < sg_max_primary) and a distinct
        # imbibition curve, so the scanning curve genuinely traps gas
        return KilloughGasHysteresis(
            sg_max_primary=1.0, sg_min_primary=0.0,
            krg_drainage=raw.kr_g,
            krg_imbibition=lambda s: 0.7 * np.asarray(value(raw.kr_g(s)),
                                                      dtype=float),
            sg_tmax=0.6)

    def _state(model, g, sg, sg_max):
        nc = g.cells.num
        st = State(pressure=np.full(nc, 200 * 1e5), sW=np.full(nc, 0.2),
                   sG=np.full(nc, 0.0), rs=np.zeros(nc),
                   status=np.full(nc, 2, int),
                   qWs=np.zeros(1), qOs=np.zeros(1), qGs=np.zeros(1),
                   bhp=np.array([200.0 * 1e5]))
        st = model.init_state_ad(st)
        st["sG"] = np.full(nc, sg)               # override with a controlled sG
        st["sGMax"] = np.full(nc, sg_max)
        return st

    def test_drainage_equals_base():
        """Where sG is at its historic max, the hysteretic curve is the
        drainage curve - identical to the base relperm."""
        if not DECK.exists():
            print("ok test_drainage_equals_base (skipped: no deck)")
            return
        model, deck, g = _spe1()
        h = HystereticRelativePermeability(model, _killough(model))
        base = BaseRelativePermeability(model)
        st = _state(model, g, sg=0.4, sg_max=0.4)     # sG == sGMax -> drainage
        krg_h = h.evaluate_gas_rel_perm(model, st["sG"], st)
        krg_b = base.evaluate_gas_rel_perm(model, st["sG"], st)
        assert np.allclose(value(krg_h), value(krg_b))
        print("ok test_drainage_equals_base")

    def test_imbibition_traps_gas():
        """Where sG has retreated below sGMax, the gas relperm is <= drainage
        (trapping reduces gas mobility)."""
        if not DECK.exists():
            print("ok test_imbibition_traps_gas (skipped: no deck)")
            return
        model, deck, g = _spe1()
        h = HystereticRelativePermeability(model, _killough(model))
        base = BaseRelativePermeability(model)
        st = _state(model, g, sg=0.3, sg_max=0.7)     # imbibition
        krg_h = np.asarray(value(h.evaluate_gas_rel_perm(model, st["sG"], st)))
        krg_b = np.asarray(value(base.evaluate_gas_rel_perm(model, st["sG"], st)))
        assert np.all(krg_h <= krg_b + 1e-12)
        assert np.any(krg_h < krg_b - 1e-9), "hysteresis should trap some gas"
        print("ok test_imbibition_traps_gas")

    test_drainage_equals_base()
    test_imbibition_traps_gas()
    print("\nAll tests passed!")
