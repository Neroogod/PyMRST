"""Base relative permeability (MRST ad-core/statefunctions/flowprops).

Ports MRST's `BaseRelativePermeability.m`: relative permeability as a state
function, dispatched on the phase system, with a per-phase evaluation hook that
hysteresis and endpoint scaling extend.

PyMRST's pre-hierarchy `RelativePermeability` simply calls the fluid's combined
`kr(saturations)` closure. This class instead reproduces MRST's phase-by-phase
logic on the *raw* fluid's individual saturation functions (`kr_w`, `kr_g`,
`kr_ow`, `kr_og`, and the connate water `swcon`):

- two-phase water-oil: ``krW = kr_w(sw)``, ``krO = kr_o(so)``;
- three-phase water-oil-gas: ``krW = kr_w(sw)``, ``krG = kr_g(sg)``, and the
  oil relperm is the connate-water-weighted blend of the oil-water and oil-gas
  curves, ``krO = (1-ww) krog(so) + ww krow(so)`` with
  ``ww = (sw - swcon)/(sg + sw - swcon)`` - exactly MRST's `relPermWOG`.

That blend is what the fluid's own `kr_o` computes, so the class reproduces the
validated `RelativePermeability` bit-for-bit (`validate_phase345.py`) while
exposing the `evaluate_phase_relative_permeability` hook the relperm hierarchy
is built on. When ``rock.krscale`` and ``fluid.krPts`` are present, the same
hook applies MRST's two- or three-point endpoint scaling.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.flowprops.virtual.saturation_property import (  # noqa: E501
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class BaseRelativePermeability(StateFunction, SaturationProperty):
    """Phase relative permeability, dispatched on the phase system."""

    def __init__(self, model: Any = None):
        StateFunction.__init__(self, model, label="k_{r\\alpha}")
        SaturationProperty.__init__(self, model)
        self.depends_on("Saturations")
        if model is not None:
            self.store_connate_water(model)
            if self.scaling_active:
                self.store_scalers(model)

    def _raw(self, model: Any) -> Any:
        return getattr(model, "raw_fluid", model.fluid)

    def evaluate_phase_relative_permeability(self, model: Any, phase: str,
                                             s: Any) -> Any:
        """kr of one phase from its own saturation function ``kr<PHASE>``.

        The hook hysteresis / endpoint scaling override; here it is the plain
        unscaled saturation function.
        """
        fn = getattr(self._raw(model), f"kr_{phase.lower()}")
        if self.scaling_active:
            scaler_phase = phase.lower()
            drainage = getattr(model.rock, "krscale", None)
            if isinstance(drainage, dict):
                drainage = drainage.get("drainage", {})
            if scaler_phase == "o" and isinstance(drainage, dict):
                scaler_phase = "ow" if "ow" in drainage else "og"
            scaled, maximum_ratio = self.scale_saturation(
                model, s, scaler_phase
            )
            return maximum_ratio * fn(scaled)
        if self.regions is not None and isinstance(fn, (list, tuple)):
            from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis import (
                evaluate_function_cell_subset_reg,
            )

            return evaluate_function_cell_subset_reg(
                self, fn, self.regions, s
            )
        return fn(s)

    def scale_saturation(
        self, model: Any, saturation: Any, phase: str,
        cells: Any = None, type: str = "drainage",
    ):
        """Map scaled endpoints back to the table saturation coordinate."""
        scaler = self.get_scalers(model, phase, cells=cells, type=type)
        points, intercepts, slopes = (
            scaler["p"], scaler["c"], scaler["m"]
        )
        saturation_value = np.asarray(value(saturation), dtype=float)
        if len(points) == 2:
            below = saturation_value < points[0]
            above = saturation_value >= points[1]
            middle = ~(below | above)
            scaled = middle.astype(float) * slopes[0] * saturation
            scaled = scaled + middle.astype(float) * intercepts[0]
            scaled = scaled + above.astype(float)
        elif len(points) == 3:
            first = (
                (saturation_value >= points[0])
                & (saturation_value < points[1])
            )
            second = (
                (saturation_value >= points[1])
                & (saturation_value < points[2])
            )
            above = saturation_value >= points[2]
            slope = (
                first.astype(float) * slopes[0]
                + second.astype(float) * slopes[1]
            )
            scaled = slope * saturation
            scaled = scaled + first.astype(float) * intercepts[0]
            scaled = scaled + second.astype(float) * intercepts[1]
            scaled = scaled + above.astype(float)
        else:
            raise ValueError(f"unknown number of scaling points: {len(points)}")
        return scaled, scaler["k_max_m"]

    def evaluate_gas_rel_perm(self, model: Any, sg: Any, state: Any) -> Any:
        """Gas relperm (MRST's `evaluateGasRelPerm`) - the hook where gas
        hysteresis, which needs the state's ``sGMax``, is applied. The base is
        the plain drainage curve."""
        return self.evaluate_phase_relative_permeability(model, "g", sg)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        nph = model.n_phases
        s = self.get_evaluated(state, "Saturations")
        if nph == 1:
            return [np.ones(model.G.cells.num)]
        if nph == 2:
            return self._rel_perm_wo(model, state, s)
        return self._rel_perm_wog(model, state, s)

    def _rel_perm_wo(self, model: Any, state: Any, s: List[Any]) -> List[Any]:
        sw, so = s[0], s[1]
        kr_w = self.evaluate_phase_relative_permeability(model, "w", sw)
        kr_o = self.evaluate_phase_relative_permeability(model, "o", so)
        return [kr_w, kr_o]

    def _rel_perm_wog(self, model: Any, state: Any, s: List[Any]) -> List[Any]:
        sw, so, sg = s[0], s[1], s[2]
        raw = self._raw(model)
        kr_w = self.evaluate_phase_relative_permeability(model, "w", sw)
        kr_g = self.evaluate_gas_rel_perm(model, sg, state)
        # three-phase oil relperm: connate-water-weighted blend (MRST relPermWOG)
        swcon = self.get_connate_water(model, state)
        swcon = np.minimum(swcon, value(sw) - 1e-5)
        d = sg + sw - swcon
        ww = (sw - swcon) / d
        krow = raw.kr_ow(so)
        krog = raw.kr_og(so)
        kr_o = (1.0 - ww) * krog + ww * krow
        return [kr_w, kr_o, kr_g]


__all__ = ["BaseRelativePermeability"]


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import (
        cart_grid, compute_geometry, tensor_grid,
    )
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel, initial_state_from_deck,
    )
    from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
        RelativePermeability as LegacyRelPerm,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.params import NORM_GRAVITY, add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    DECK = Path(__file__).resolve().parents[6] / \
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"

    def _ow():
        g = compute_geometry(cart_grid([5, 4, 2], [250.0, 200.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w)
        st = model.init_state_ad(State(
            pressure=np.linspace(210, 190, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
            qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def _compare(model, st):
        model.FlowPropertyFunctions.set_state_function(
            "BaseRelativePermeability", BaseRelativePermeability(model))
        new = model.get_props(st, "BaseRelativePermeability")
        model.FlowPropertyFunctions.set_state_function(
            "RelativePermeability", LegacyRelPerm(model))
        old = model.get_props(st, "RelativePermeability")
        worst = 0.0
        for ph in range(model.n_phases):
            worst = max(worst, np.max(np.abs(value(new[ph]) - value(old[ph]))))
            assert np.allclose((new[ph] - old[ph]).jac[0].toarray(), 0.0)
        return worst

    def test_matches_legacy_oil_water():
        model, st = _ow()
        worst = _compare(model, st)
        assert worst == 0.0, worst
        print(f"ok test_matches_legacy_oil_water (worst {worst:.1e})")

    def test_matches_legacy_black_oil_three_phase():
        if not DECK.exists():
            print("ok test_matches_legacy_black_oil_three_phase (skipped: no deck)")
            return
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
        st0 = initial_state_from_deck(model, deck)
        st = dict(st0)
        nc = g.cells.num
        st["sG"] = np.where(np.arange(nc) % 2 == 0, 0.2, 0.0)
        st["status"] = np.where(np.arange(nc) % 2 == 0, 2, 1)
        st = model.init_state_ad(State(st))
        worst = _compare(model, st)
        assert worst < 1e-14, worst
        print(f"ok test_matches_legacy_black_oil_three_phase (worst {worst:.1e})")

    test_matches_legacy_oil_water()
    test_matches_legacy_black_oil_three_phase()
    print("\nAll tests passed!")
