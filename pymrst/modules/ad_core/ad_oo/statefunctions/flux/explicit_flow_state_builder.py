"""MRST ``ad-core/statefunctions/flux/flow_state/ExplicitFlowStateBuilder.m``.

Explicit (IMPES-type) fluxes: the flow state is the current iterate with
selected state functions replaced by their values at the start of the step.
By default those are `FaceMobility`, `FaceComponentMobility` and
`GravityPotentialDifference` (`explicitFlux`); `explicitFlow`, `explicitPVT`
and `explicitProps` add flow properties, PVT properties and stored
variables. The step is limited to the target saturation and composition
CFL numbers (0.9 each) from `estimateSaturationCFL`/`estimateCompositionCFL`,
using the fluxes stored with the previous step; with no fluxes yet (the
first step) the limit is `initialStep` (one day). `changeTimestep = False`
removes the limit. The model must store fluxes (`outputFluxes`).

MRST's grouping caches are handle objects (`HandleStruct`): the flow state
is a copy of `state` that shares its caches, so the explicit values written
into the flow state are also in `state`'s caches, and clearing an entry of
`state0`'s cache clears it for the caller too. `copy_state` shares the
cache dicts in the same way, and this port writes into them as MRST does.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.flow_state_builder import (
    FlowStateBuilder,
)
from pymrst.utils.units import DAY


class ExplicitFlowStateBuilder(FlowStateBuilder):
    """MRST `ExplicitFlowStateBuilder`."""

    def __init__(self, **options: Any):
        self.saturationCFL = 0.9          # Target saturation CFL. Should be <= 1 for stability.
        self.compositionCFL = 0.9         # Target composition CFL. Should be <= 1 for stability.
        self.explicitFlux = ["FaceMobility", "FaceComponentMobility",
                             "GravityPotentialDifference"]
        self.explicitFlow: List[str] = []
        self.explicitPVT: List[str] = []
        self.explicitProps: List[str] = []
        self.initialStep = 1*DAY          # Timestep used if no fluxes are present
        self.useInflowForEstimate = False
        self.changeTimestep = True        # Estimate stable step.
        super().__init__(**options)

    def get_maximum_timestep(self, fd: Any, model: Any, state: Any, state0: Any,
                             dt: float, forces: Any) -> float:
        from pymrst.modules.ad_core.utils.estimate_composition_cfl import (
            estimate_composition_cfl,
        )
        from pymrst.modules.ad_core.utils.estimate_saturation_cfl import (
            estimate_saturation_cfl,
        )
        if not self.changeTimestep:
            return float("inf")
        if self.is_first_time_step(state):
            return self.initialStep
        state = model.reduce_state(state, True)
        iflow = self.useInflowForEstimate
        cfl_s = estimate_saturation_cfl(model, state, 1/self.saturationCFL, forces=forces,
                                        useInflow=iflow)
        cfl_c = estimate_composition_cfl(model, state, 1/self.compositionCFL, forces=forces,
                                         useInflow=iflow)
        with np.errstate(divide="ignore"):
            dt_max_z = np.min(1/np.max(np.atleast_2d(cfl_c.T).T, axis=1))
            dt_max_s = np.min(1/np.atleast_1d(cfl_s))
        dt_max = float(min(dt_max_z, dt_max_s))
        if self.verbose and dt_max < dt:
            from pymrst.utils.format_time_range import format_time_range
            kind = "saturation" if dt_max_z > dt_max_s else "composition"
            print(f"Time-step limited by {kind} CFL: {format_time_range(dt, 0)} reduced to "
                  f"{format_time_range(dt_max, 0)} ({100*(dt - dt_max)/dt:1.2f}% reduction)")
        return dt_max

    def build(self, fd: Any, model: Any, state: Any, state0: Any, dt: float) -> Any:
        flow_state = copy_state(state)
        state0 = copy_state(state0)
        for p in self.explicitProps:
            flow_state = model.set_prop(flow_state, p, model.get_prop(state0, p))
        groups, names = self.get_explicit_groups(model)
        for grp, props in zip(groups, names):
            name = grp.get_state_function_container_name()
            if name not in state0.containers:
                state0 = model.init_state_function_containers(state0)
            if name not in flow_state.containers:
                flow_state = model.init_state_function_containers(flow_state)
            for prop in props:
                state0.containers[name][prop] = None
                flow_state.containers[name][prop] = model.get_props(state0, prop)
        return flow_state

    def get_explicit_groups(self, model: Any) -> Tuple[list, list]:
        groups, names = [], []
        if self.explicitPVT:
            groups.append(model.PVTPropertyFunctions)
            names.append(self.explicitPVT)
        if self.explicitFlow:
            groups.append(model.FlowPropertyFunctions)
            names.append(self.explicitFlow)
        if self.explicitFlux:
            groups.append(model.FlowDiscretization)
            names.append(self.explicitFlux)
        return groups, names

    def is_first_time_step(self, state: Any) -> bool:
        """No stored fluxes, or all of them zero."""
        flux = state.get("flux") if hasattr(state, "get") else None
        return flux is None or not np.any(np.max(np.abs(np.asarray(flux, dtype=float)), axis=0))

    def prepare_timestep(self, fd: Any, model: Any, state: Any, state0: Any, dt: float,
                         driving_forces: Any):
        assert model.outputFluxes, "model.outputFluxes must be true for explicit solver"
        return self, state

    getMaximumTimestep = get_maximum_timestep
    prepareTimestep = prepare_timestep
    getExplicitGroups = get_explicit_groups
    isFirstTimeStep = is_first_time_step



__all__ = ["ExplicitFlowStateBuilder"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.solvers.init_res_sol import init_res_sol

    def _case():
        G = compute_geometry(cart_grid([4], [4.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.5), fluid, gas=False)
        model = model.validate_model()
        return model, G

    def test_the_first_step_is_the_initial_step():
        model, G = _case()
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        builder = ExplicitFlowStateBuilder()
        assert builder.is_first_time_step(state)
        assert builder.get_maximum_timestep(None, model, state, state, 10*DAY, None) == DAY
        assert ExplicitFlowStateBuilder(changeTimestep=False).get_maximum_timestep(
            None, model, state, state, DAY, None) == float("inf")
        print("ok test_the_first_step_is_the_initial_step")

    def test_the_flux_mobilities_come_from_the_start_of_the_step():
        """The flow state's face mobility is state0's, and - the caches being
        shared, as MRST's handles are - so is `state`'s."""
        model, G = _case()
        state0 = model.validate_state(init_res_sol(G, 1e7, np.array([0.5, 0.5])))
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.9, 0.1])))
        state = model.init_state_function_containers(state)
        builder = ExplicitFlowStateBuilder()
        flow_state = builder.build(model.FlowDiscretization, model, state, state0, DAY)
        fm = model.get_props(flow_state, "FaceMobility")
        fm0 = model.get_props(state0, "FaceMobility")
        for a, b in zip(fm, fm0):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))
        name = model.FlowDiscretization.get_state_function_container_name()
        assert state.containers[name]["FaceMobility"] is flow_state.containers[name]["FaceMobility"]
        print("ok test_the_flux_mobilities_come_from_the_start_of_the_step")

    test_the_first_step_is_the_initial_step()
    test_the_flux_mobilities_come_from_the_start_of_the_step()
    print("\nAll tests passed!")
