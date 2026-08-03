"""AdaptiveImplicitFlowStateBuilder (MRST ad-core/statefunctions/flux/flow_state).

Ports MRST's `AdaptiveImplicitFlowStateBuilder.m` - AIM. Rather than choosing
implicit or explicit for the whole domain, AIM chooses *per cell*: cells where
the step is stable are taken explicitly (cheap), cells where it is not - and,
always, the cells a well perforates - are taken implicitly (stable). The blend
is linear per cell:

    X_hybrid = implicit .* X(state) + explicit .* X(state0)

applied to the flow (cell) properties `Mobility` and `ComponentMobility`
(AIM freezes cell quantities, not face quantities - hence `explicit_flux = {}`).
A cell marked implicit keeps its full AD derivative; a cell marked explicit is
frozen at the old value.

Which cells are implicit is a CFL decision (`estimateSaturationCFL` /
`estimateCompositionCFL`). As in MRST, a state without accepted fluxes is
handled before CFL estimation and uses `first_step_implicit`; later states use
the injected estimator. Well cells are always forced implicit.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.flux.flow_state.explicit_flow_state_builder import (  # noqa: E501
    ExplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.statefunctions.flux.flow_state.flow_state_builder import (  # noqa: E501
    owning_grouping,
    seed_property,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value


class AdaptiveImplicitFlowStateBuilder(ExplicitFlowStateBuilder):
    """AIM: per-cell implicit/explicit blend of the flow (cell) mobilities."""

    def __init__(self, first_step_implicit: bool = True,
                 implicit_wells: bool = True,
                 cfl_estimator: Optional[Callable] = None, **kw):
        super().__init__(cfl_estimator=cfl_estimator, **kw)
        self.first_step_implicit = first_step_implicit
        self.implicit_wells = implicit_wells
        self.explicit_flux = []                       # AIM does not do flux terms
        self.explicit_flow = ["Mobility", "ComponentMobility"]

    def get_maximum_timestep(self, model: Any, state: State, state0: State,
                             dt: float, forces: Any = None) -> float:
        """AIM adapts the cell partition, not the step - no CFL cap."""
        return float("inf")

    def prepare_timestep(self, model: Any, state: State, state0: State,
                         dt: float, forces: Any = None, mask: Any = None):
        """Set ``state.implicit`` (per-cell) before the nonlinear loop."""
        nc = model.G.cells.num
        if mask is not None:
            implicit = np.asarray(mask, bool).copy()
        elif self.is_first_timestep(state):
            implicit = np.full(nc, self.first_step_implicit, bool)
        elif self.cfl_estimator is not None:
            cfl_s, cfl_c = self.cfl_estimator(model, state, dt, forces)
            sat = np.asarray(cfl_s, dtype=float)
            if sat.ndim > 1:
                sat = np.max(sat, axis=1)
            comp = np.asarray(cfl_c, dtype=float)
            if comp.ndim > 1:
                comp = np.max(comp, axis=1)
            implicit = sat >= self.saturation_cfl
            implicit |= comp >= self.composition_cfl
        else:
            # A non-initial state without an estimator cannot be partitioned
            # safely. This branch is retained for direct builder use; the
            # public set_time_discretization helper installs the estimator.
            implicit = np.ones(nc, bool)
        if self.implicit_wells and forces is not None:
            for w in _iter_wells(forces):
                implicit[np.asarray(w, int)] = True
        state["implicit"] = implicit
        return self, state

    def build(self, model: Any, state: State, state0: State, dt: float,
              fd: Any = None) -> State:
        implicit = np.asarray(state["implicit"], bool)
        flow_state = state.copy()
        if implicit.all():
            return flow_state                          # nothing to freeze
        explicit = (~implicit).astype(float)
        impl = implicit.astype(float)
        for prop in self.explicit_flow:
            try:
                grp = owning_grouping(model, prop)
            except KeyError:
                continue
            x = grp.get(model, state, prop)            # implicit (AD)
            x0 = grp.get(model, state0, prop)          # explicit (numeric)
            seed_property(flow_state, grp, prop, _blend(x, x0, impl, explicit))
        return flow_state


def _blend(x: Any, x0: Any, impl: np.ndarray, expl: np.ndarray) -> Any:
    """implicit .* x + explicit .* x0, over lists (phases/components) too."""
    if isinstance(x, list):
        out = []
        for xi, x0i in zip(x, x0):
            out.append(None if xi is None else _blend(xi, x0i, impl, expl))
        return out
    return impl * x + expl * np.asarray(value(x0))


def _iter_wells(forces: Any):
    wells = forces.get("W") if isinstance(forces, dict) else getattr(forces, "W", None)
    for w in (wells or []):
        cells = w.get("cells") if isinstance(w, dict) else getattr(w, "cells", None)
        if cells is not None:
            yield cells


__all__ = ["AdaptiveImplicitFlowStateBuilder"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model_and_states():
        g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w, gravity=9.80665)
        st0 = State(pressure=np.linspace(230, 180, g.cells.num) * BARSA,
                    sW=np.full(g.cells.num, 0.4),
                    qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                    bhp=np.array([150 * BARSA]))
        st_new = State(pressure=np.linspace(228, 178, g.cells.num) * BARSA,
                       sW=np.full(g.cells.num, 0.6),
                       qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                       bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st_new), st0, g.cells.num

    def test_all_implicit_is_fully_implicit():
        """An all-implicit mask leaves Mobility untouched (full AD)."""
        model, st, st0, nc = _model_and_states()
        base = model.get_props(st, "Mobility")
        b = AdaptiveImplicitFlowStateBuilder()
        st["implicit"] = np.ones(nc, bool)
        fs = b.build(model, st, st0, dt=30.0)
        mob = model.get_props(fs, "Mobility")
        for ph in range(model.n_phases):
            assert np.allclose(value(mob[ph]), value(base[ph]))
            assert np.allclose((mob[ph] - base[ph]).jac[0].toarray(), 0.0)
        print("ok test_all_implicit_is_fully_implicit")

    def test_all_explicit_freezes_mobility():
        """An all-explicit mask freezes Mobility at state0 (no Jacobian)."""
        model, st, st0, nc = _model_and_states()
        old = model.get_props(st0, "Mobility")
        b = AdaptiveImplicitFlowStateBuilder()
        st["implicit"] = np.zeros(nc, bool)
        fs = b.build(model, st, st0, dt=30.0)
        mob = model.get_props(fs, "Mobility")
        for ph in range(model.n_phases):
            assert np.allclose(value(mob[ph]), value(old[ph]))
            frozen = not hasattr(mob[ph], "jac") or all(
                np.abs(b.toarray()).max() == 0.0 for b in mob[ph].jac)
            assert frozen, "explicit mobility must carry no derivative"
        print("ok test_all_explicit_freezes_mobility")

    def test_mixed_mask_blends_per_cell():
        """A mixed mask blends cell by cell: implicit cells keep the new
        value, explicit cells take the old."""
        model, st, st0, nc = _model_and_states()
        new = model.get_props(st, "Mobility")
        old = model.get_props(st0, "Mobility")
        mask = np.zeros(nc, bool)
        mask[::2] = True                                   # every other cell
        b = AdaptiveImplicitFlowStateBuilder()
        st["implicit"] = mask
        fs = b.build(model, st, st0, dt=30.0)
        mob = model.get_props(fs, "Mobility")

        def total_jac_rows(adi):
            """|derivative| per cell summed over every primary-variable block."""
            return sum(np.abs(b.toarray()).sum(axis=1) for b in adi.jac)

        for ph in range(model.n_phases):
            expect = np.where(mask, value(new[ph]), value(old[ph]))
            assert np.allclose(value(mob[ph]), expect)
            # explicit cells contribute no derivative; implicit cells do
            jac_rows = total_jac_rows(mob[ph])
            assert np.all(jac_rows[~mask] == 0.0)
            assert np.any(jac_rows[mask] > 0.0)
        print("ok test_mixed_mask_blends_per_cell")

    def test_prepare_forces_well_cells_implicit():
        """Well perforations are implicit even under an all-explicit request."""
        model, st, st0, nc = _model_and_states()
        b = AdaptiveImplicitFlowStateBuilder()
        forces = {"W": [{"cells": np.array([0])},
                        {"cells": np.array([nc - 1])}]}
        _, st = b.prepare_timestep(model, st, st0, dt=30.0, forces=forces,
                                   mask=np.zeros(nc, bool))
        assert st["implicit"][0] and st["implicit"][nc - 1]
        assert not st["implicit"][nc // 2]
        print("ok test_prepare_forces_well_cells_implicit")

    def test_aim_has_no_timestep_limit():
        model, st, st0, nc = _model_and_states()
        b = AdaptiveImplicitFlowStateBuilder()
        assert b.get_maximum_timestep(model, st, st0, dt=30.0) == float("inf")
        print("ok test_aim_has_no_timestep_limit")

    test_all_implicit_is_fully_implicit()
    test_all_explicit_freezes_mobility()
    test_mixed_mask_blends_per_cell()
    test_prepare_forces_well_cells_implicit()
    test_aim_has_no_timestep_limit()
    print("\nAll tests passed!")
