"""MRST ``sequential/models/TransportModel.m``.

The transport step of a sequential split: a `WrapperModel` around a fully
implicit reservoir model, solved at the total flux the pressure step left in
`state.flux`. Pressure is not a variable. In the default 'totalSaturation'
formulation its place is taken by the total saturation `sT` (the sum of the
saturations, which transport does not keep at one), and the component masses,
densities and mobilities are scaled by it; with 'missingPhase' the last
component's equation is dropped instead. The flux discretization is replaced
by fractional-flow fluxes at fixed total velocity, with phase-potential
upwinding, and the wells by fixed total well fluxes split by fractional flow
(unless `implicitType` is 'wells', which keeps pressure implicit in well
cells). `dsMaxTotal` limits the change of `sT` per update.

The parent model given is copied before its flux discretization is removed,
as MATLAB's value semantics do.
"""

from __future__ import annotations

from copy import copy
from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel, merge_options
from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel


class TransportModel(WrapperModel):
    """MRST `TransportModel`."""

    def __init__(self, parent: Any, **options: Any):
        parent = copy(parent)
        parent.FlowDiscretization = None       # Remove flux discretization (if setup)
        super().__init__(parent)
        self.formulation = "totalSaturation"
        self.dsMaxTotal = np.inf
        self.implicitType = "none"
        merge_options(self, options)
        self.AutoDiffBackend = parent.AutoDiffBackend

    def get_state_ad(self, state: Any, init: bool = True):
        parent = self.parentModel
        impl_cells = self.get_implicit_cells(state)
        wells_active = impl_cells.size > 0
        basevars, basenames, baseorigin = self.get_primary_variables(state)
        basevars, basenames, baseorigin = list(basevars), list(basenames), list(baseorigin)
        is_parent = [o == type(parent).__name__ for o in baseorigin]
        is_primary = [p or wells_active for p in is_parent]
        isS = [False]*len(basevars)
        nph = parent.get_number_of_phases()
        for i, bn in enumerate(basenames):
            if not is_parent[i]:
                continue
            if bn == "x":
                ss = True
            else:
                f, _ = self.get_variable_field(bn)
                ss = f == "s"
            if ss:
                isS[i] = True
        isP = [n == "pressure" for n in basenames]
        p_index = isP.index(True)
        variables, names, origin = list(basevars), list(basenames), list(baseorigin)
        use_total_saturation = (self.formulation.lower() == "totalsaturation"
                                and sum(isS) < nph)
        if use_total_saturation:
            replacement = "sT"
            sT = self.get_prop(state, replacement)
            if wells_active:
                p = basevars[p_index]
                sT = np.array(sT, dtype=float, copy=True)
                sT[impl_cells] = np.asarray(p)[impl_cells]
            variables[p_index] = sT
            names[p_index] = replacement
            origin[p_index] = type(self).__name__
        else:
            keep = [not x for x in isP]
            variables = [v for v, k in zip(variables, keep) if k]
            names = [n for n, k in zip(names, keep) if k]
            origin = [o for o, k in zip(origin, keep) if k]
            is_primary = [x for x, k in zip(is_primary, keep) if k]
        if init:
            chosen = [v for v, k in zip(variables, is_primary) if k]
            chosen = list(self.AutoDiffBackend.init_variables_ad(*chosen))
            it = iter(chosen)
            variables = [next(it) if k else v for v, k in zip(variables, is_primary)]
        if use_total_saturation:
            for i in range(len(basevars)):
                if not isP[i]:
                    basevars[i] = variables[i]
            sT = variables[p_index]
            if wells_active:
                p = self.AutoDiffBackend.convert_to_ad(basevars[p_index], sT)
                p[impl_cells] = sT[impl_cells]
                sT[impl_cells] = 1                    # Total saturation = 1 in these cells
                basevars[p_index] = p
            state = self.set_prop(state, replacement, sT)
        else:
            it = iter(variables)
            basevars = [v if isP[i] else next(it) for i, v in enumerate(basevars)]
        state = self.init_state_ad(state, basevars, basenames, baseorigin)
        names = [n for n, k in zip(names, is_primary) if k]
        origin = [o for o, k in zip(origin, is_primary) if k]
        return state, names, origin

    def get_model_equations(self, state0: Any, state: Any, dt: float, driving_forces: Any):
        parent = self.parentModel
        eqs, names, types, state = parent.get_model_equations(state0, state, dt, driving_forces)
        if self.formulation.lower() == "missingphase":
            cnames = parent.get_component_names()
            subs = [n.lower() != cnames[-1].lower() for n in names]
            eqs = [e for e, k in zip(eqs, subs) if k]
            names = [n for n, k in zip(names, subs) if k]
            types = [t for t, k in zip(types, subs) if k]
        return eqs, names, types, state

    def validate_state(self, state: Any) -> Any:
        state = super().validate_state(state)
        fn, _ = self.get_variable_field("sT")
        if "s" in state and fn not in state:
            state = self.set_prop(state, "sT", np.sum(np.asarray(state["s"], dtype=float), axis=1))
        return state

    def setup_state_function_groupings(self, set_defaults: bool = True):
        from pymrst.modules.sequential.statefunctions.facility.well_phase_flux_total_fixed import (
            WellPhaseFluxTotalFixed,
        )
        from pymrst.modules.sequential.statefunctions.flux import (
            component_mobility_total_saturation as cmts,
            component_phase_density_total_saturation as cpdts,
            component_phase_flux_fractional_flow as cpfff,
            component_phase_mass_total_saturation as cpmts,
            face_total_mobility as ftm,
            fixed_total_flux as ftf,
            phase_flux_fixed_total_velocity as pfftv,
            phase_interface_pressure_differences as pipd,
            phase_potential_upwind_flag as ppuf,
            total_saturation as ts,
        )
        model = copy(self)
        pmodel = copy(model.parentModel)
        has_facility = getattr(pmodel, "FacilityModel", None) is not None
        is_total_sat = model.formulation.lower() == "totalsaturation"
        if pmodel.FlowDiscretization is None:
            pmodel = pmodel.setup_state_function_groupings()
        elif not set_defaults:
            import warnings
            warnings.warn("State function models for parent model is already set up.")
            return model
        fd = _copy_grouping(pmodel.FlowDiscretization)
        fp = _copy_grouping(pmodel.FlowPropertyFunctions)
        pvt = pmodel.PVTPropertyFunctions
        fd.set_state_function("PhaseFlux", pfftv.PhaseFluxFixedTotalVelocity(pmodel))
        fd.set_state_function("PhaseUpwindFlag", ppuf.PhasePotentialUpwindFlag(pmodel))
        fd.set_state_function("ComponentPhaseFlux", cpfff.ComponentPhaseFluxFractionalFlow(pmodel))
        fd.set_state_function("PhaseInterfacePressureDifferences",
                              pipd.PhaseInterfacePressureDifferences(pmodel))
        fd.set_state_function("TotalFlux", ftf.FixedTotalFlux(pmodel))
        fd.set_state_function("FaceTotalMobility", ftm.FaceTotalMobility(pmodel))
        if is_total_sat:
            fp.set_state_function("TotalSaturation", ts.TotalSaturation(pmodel))
            fp.set_state_function("ComponentPhaseDensity",
                                  cpdts.ComponentPhaseDensityTotalSaturation(pmodel))
            fp.set_state_function("ComponentMobility", cmts.ComponentMobilityTotalSaturation(pmodel))
            fp.set_state_function("ComponentPhaseMass", cpmts.ComponentPhaseMassTotalSaturation(pmodel))
        pmodel.FlowDiscretization = fd
        pmodel.FlowPropertyFunctions = fp
        pmodel.PVTPropertyFunctions = pvt
        if has_facility and model.implicitType.lower() != "wells":
            facility = copy(pmodel.FacilityModel)
            facility.primaryVariableSet = "none"
            fdp = _copy_grouping(facility.FacilityFlowDiscretization)
            fdp.set_state_function("PhaseFlux", WellPhaseFluxTotalFixed(pmodel))
            facility.FacilityFlowDiscretization = fdp
            facility.doPostUpdate = True
            pmodel.FacilityModel = facility
        model.parentModel = pmodel
        return model

    def get_equations(self, state0: Any, state: Any, dt: float, forces: Any, **options: Any):
        return PhysicalModel.get_equations(self, state0, state, dt, forces, **options)

    def get_implicit_cells(self, state: Any) -> np.ndarray:
        if self.implicitType.lower() == "wells":
            fm = self.parentModel.FacilityModel
            mapping = fm.get_prop(state, "FacilityWellMapping")
            return np.asarray(mapping.cells, dtype=np.int64)
        assert self.implicitType.lower() == "none"
        return np.zeros(0, dtype=np.int64)

    def update_state(self, state: Any, problem: Any, dx: List[Any], driving_forces: Any):
        dst, dx, problem = problem.pop_increment(dx, "sT", nargout=3)
        if dst is not None:
            impl_cells = self.get_implicit_cells(state)
            dst = np.array(dst, dtype=float, copy=True)
            if impl_cells.size:
                dp = np.zeros(dst.shape)
                dp[impl_cells] = dst[impl_cells]
                pmodel = self.parentModel
                state, _, _ = self.update_state_from_increment(state, dp, problem, "pressure",
                                                               pmodel.dpMaxRel, pmodel.dpMaxAbs)
                dst[impl_cells] = 0
            state, _, _ = self.update_state_from_increment(state, dst, problem, "sT", np.inf,
                                                           self.dsMaxTotal)
            state = self.cap_property(state, "sT", 1e-8)
        return self.parentModel.update_state(state, problem, dx, driving_forces)

    def check_convergence(self, problem: Any, *args: Any):
        return self.parentModel.check_convergence(problem)

    def get_variable_field(self, name: str, *args: Any, **kwargs: Any):
        if name.lower() == "st":
            from pymrst.modules.ad_core.ad_oo.physical_model import WHOLE
            return "sT", WHOLE
        return self.parentModel.get_variable_field(name, *args, **kwargs)

    def prepare_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        return super().prepare_timestep(state, state0, dt, driving_forces)

    def update_after_convergence(self, *args: Any):
        return super().update_after_convergence(*args)

    getStateAD = get_state_ad
    getModelEquations = get_model_equations
    validateState = validate_state
    setupStateFunctionGroupings = setup_state_function_groupings
    getEquations = get_equations
    getImplicitCells = get_implicit_cells
    updateState = update_state
    checkConvergence = check_convergence
    getVariableField = get_variable_field
    prepareTimestep = prepare_timestep
    updateAfterConvergence = update_after_convergence


def _copy_grouping(grouping: Any) -> Any:
    """MATLAB's value copy of a state function grouping: its own table of
    functions, so setting one does not change the original."""
    out = copy(grouping)
    out.functions = dict(grouping.functions)
    return out


__all__ = ["TransportModel"]


if __name__ == "__main__":
    import warnings

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_total_saturation_replaces_pressure():
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        _, names, origin = T.model.get_state_ad(copy_state(T.state))
        assert names == ["sT", "sW"], names
        assert origin[0] == "TransportModel"
        assert T.model.get_variable_field("sT") == ("sT", ":")
        print("ok test_total_saturation_replaces_pressure")

    def test_the_transport_functions_are_installed():
        fd = T.parent.FlowDiscretization
        assert type(fd.get_state_function("PhaseFlux")).__name__ == "PhaseFluxFixedTotalVelocity"
        assert type(T.parent.FlowPropertyFunctions.get_state_function(
            "ComponentPhaseMass")).__name__ == "ComponentPhaseMassTotalSaturation"
        assert T.parent.FacilityModel.primaryVariableSet == "none"
        print("ok test_the_transport_functions_are_installed")

    def test_the_original_model_keeps_its_discretization():
        case = toy_case()
        assert type(case.model.FlowDiscretization).__name__ in ("NoneType", "FlowDiscretization")
        print("ok test_the_original_model_keeps_its_discretization")

    test_total_saturation_replaces_pressure()
    test_the_transport_functions_are_installed()
    test_the_original_model_keeps_its_discretization()
    print("\nAll tests passed!")
