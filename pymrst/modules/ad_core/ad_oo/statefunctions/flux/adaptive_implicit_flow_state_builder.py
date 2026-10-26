"""MRST ``ad-core/statefunctions/flux/flow_state/AdaptiveImplicitFlowStateBuilder.m``.

Adaptive implicit (AIM): each step, the cells whose saturation or
composition CFL for that step reaches the target (0.9) are implicit, and
the rest explicit; well cells are always implicit (`implicitWells`), and on
the first step, with no fluxes yet, every cell is (`firstStepImplicit`).
The choice is stored as `state.implicit`. The flow state blends the
explicit flow properties, by default `Mobility` and `ComponentMobility`:
`implicit*X + explicit*X0`, with `X0` from the start of the step. Flux
state functions cannot be explicit here (`explicitFlux` must stay empty),
and AIM sets no timestep limit.

As in the explicit builder, the grouping caches are shared with `state`
(MRST's `HandleStruct` handles), so the blended values are what `state`
itself sees afterwards: the boundary conditions, evaluated on `state`, take
the explicit mobility in explicit cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.explicit_flow_state_builder import (
    ExplicitFlowStateBuilder,
)


class AdaptiveImplicitFlowStateBuilder(ExplicitFlowStateBuilder):
    """MRST `AdaptiveImplicitFlowStateBuilder`."""

    def __init__(self, **options: Any):
        self.firstStepImplicit = True      # The first step is implicit if fluxes are missing
        self.implicitWells = True          # Well cells are always implicit
        super().__init__(**options)
        self.explicitFlux = []             # Not supported
        self.explicitFlow = ["Mobility", "ComponentMobility"]

    def get_maximum_timestep(self, fd: Any, model: Any, state: Any, state0: Any,
                             dt: float, forces: Any) -> float:
        return float("inf")

    def build(self, fd: Any, model: Any, state: Any, state0: Any, dt: float) -> Any:
        flow_state = state
        implicit = np.asarray(state["implicit"], dtype=bool)
        if np.all(implicit):
            return flow_state
        flow_state = copy_state(state)
        state0 = copy_state(state0)
        explicit = ~implicit
        imp, exp = implicit.astype(float), explicit.astype(float)
        groups, names = self.get_explicit_groups(model)
        for grp, props in zip(groups, names):
            name = grp.get_state_function_container_name()
            if name not in flow_state.containers:
                flow_state = model.init_state_function_containers(flow_state)
            for prop in props:
                if name in state0.containers:
                            state0.containers[name][prop] = None
                X = model.get_props(state, prop)
                X0 = model.get_props(state0, prop)
                flow_state.containers[name][prop] = _blend(X, X0, imp, exp)
            # MRST applies `explicitProps` inside this loop, once per group.
            for p in self.explicitProps:
                v0 = model.get_prop(state0, p)
                v = model.get_prop(state, p)
                if isinstance(v, list):
                    v = [v[j]*imp + exp*(v0[j] if isinstance(v0, list) else np.asarray(v0)[:, j])
                         for j in range(len(v))]
                else:
                    v = v*imp + exp*v0
                flow_state = model.set_prop(flow_state, p, v)
        return flow_state

    def prepare_timestep(self, fd: Any, model: Any, state: Any, state0: Any, dt: float,
                         driving_forces: Any):
        from pymrst.modules.ad_core.utils.estimate_composition_cfl import (
            estimate_composition_cfl,
        )
        from pymrst.modules.ad_core.utils.estimate_saturation_cfl import (
            estimate_saturation_cfl,
        )
        builder, state = super().prepare_timestep(fd, model, state, state0, dt, driving_forces)
        nc = model.G.cells.num
        if builder.is_first_time_step(state):
            impl_sat = impl_comp = np.full(nc, bool(builder.firstStepImplicit))
        else:
            iflow = builder.useInflowForEstimate
            cfl = estimate_saturation_cfl(model, state, dt, forces=driving_forces, useInflow=iflow)
            impl_sat = cfl >= builder.saturationCFL
            cfl_c = estimate_composition_cfl(model, state, dt, forces=driving_forces,
                                             useInflow=iflow)
            impl_comp = np.max(cfl_c, axis=1) >= builder.compositionCFL
        implicit = impl_sat | impl_comp
        W = _forces_wells(driving_forces)
        if builder.implicitWells and W:
            wc = np.concatenate([np.asarray(w.cells, dtype=np.int64).ravel() for w in W])
            implicit[wc] = True
        else:
            wc = np.zeros(0, dtype=np.int64)
        if builder.verbose:
            ni = int(implicit.sum())
            print(f"Adaptive implicit: {ni} of {nc} cells are implicit ({100*ni/nc:2.2f}%).\n"
                  f"{int(np.sum(impl_comp))} limited by composition, {int(np.sum(impl_sat))} "
                  f"limited by saturation, {wc.size} belong to wells.")
        state["implicit"] = implicit
        assert not builder.explicitFlux, (
            "AIM does not support flux terms. You should specify PVT or Flow propertes instead.")
        return builder, state

    getMaximumTimestep = get_maximum_timestep
    prepareTimestep = prepare_timestep


def _blend(X: Any, X0: Any, imp: np.ndarray, exp: np.ndarray) -> Any:
    """`implicit.*X + explicit.*X0`, entry by entry through (nested) cell
    arrays such as `ComponentMobility`; empty entries stay empty."""
    if isinstance(X, (list, tuple)):
        return [_blend(x, x0, imp, exp) for x, x0 in zip(X, X0)]
    if X is None or (not hasattr(X, "jac") and np.size(X) == 0):
        return X
    return imp*X + exp*X0


def _forces_wells(forces: Any) -> list:
    if forces is None:
        return []
    W = forces.get("W") if isinstance(forces, dict) else getattr(forces, "W", None)
    return list(W) if W is not None and len(W) else []



__all__ = ["AdaptiveImplicitFlowStateBuilder"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.solvers.init_res_sol import init_res_sol
    from pymrst.utils.units import DAY

    def _case():
        G = compute_geometry(cart_grid([4], [4.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.5), fluid, gas=False)
        return model.validate_model(), G

    def test_the_first_step_is_all_implicit():
        model, G = _case()
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        builder, state = AdaptiveImplicitFlowStateBuilder().prepare_timestep(
            None, model, state, state, DAY, {"W": []})
        assert np.all(state["implicit"])
        assert builder.get_maximum_timestep(None, model, state, state, DAY, None) == float("inf")
        print("ok test_the_first_step_is_all_implicit")

    def test_explicit_cells_take_the_start_of_step_mobility():
        model, G = _case()
        state0 = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.9, 0.1])))
        state["implicit"] = np.array([True, False, True, False])
        flow_state = AdaptiveImplicitFlowStateBuilder().build(None, model, state, state0, DAY)
        name = model.FlowPropertyFunctions.get_state_function_container_name()
        mob = np.asarray(flow_state.containers[name]["Mobility"][0])
        mob_now = np.asarray(model.get_props(state, "Mobility")[0])
        mob0 = np.asarray(model.get_props(state0, "Mobility")[0])
        np.testing.assert_allclose(mob, np.where(state["implicit"], mob_now, mob0))
        print("ok test_explicit_cells_take_the_start_of_step_mobility")

    test_the_first_step_is_all_implicit()
    test_explicit_cells_take_the_start_of_step_mobility()
    print("\nAll tests passed!")
