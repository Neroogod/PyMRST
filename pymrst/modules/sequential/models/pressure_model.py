"""MRST ``sequential/models/PressureModel.m``.

The pressure step of a sequential split: a `WrapperModel` around a fully
implicit reservoir model whose primary variables are pressure plus those not
from the reservoir model (the wells'); everything else stays fixed at its
current value. The component conservation equations are combined into one
pressure equation with the `PressureReductionFactors` weights (analytic for
black-oil models, numerical otherwise). Convergence is the pressure
increment, relative to the pressure range, below `incTolPressure` (1e-3)
after the first iteration, plus the parent's checks on the remaining (well)
equations; with a finite `pressureTol` the pressure equation's residual,
scaled by the weighted total mass, is checked too. `pressureScaling` scales
pressure (and BHP) as variables; `laggedStateFunctions` keeps chosen state
functions at their start-of-step values.
"""

from __future__ import annotations

from copy import copy
from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel, merge_options
from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA


class PressureModel(WrapperModel):
    """MRST `PressureModel`."""

    def __init__(self, parent: Any, **options: Any):
        if hasattr(parent, "useCNVConvergence"):
            parent = copy(parent)
            parent.useCNVConvergence = False
        super().__init__(parent)
        self.incTolPressure = 1e-3
        self.useIncTol = True
        self.reductionStrategy = None
        self.pressureIncTolType = "relative"
        self.pressureTol = np.inf
        self.pressureScaling = None
        self.laggedStateFunctions = None
        merge_options(self, options)
        if self.reductionStrategy is None:
            from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
                ThreePhaseBlackOilModel,
            )
            self.reductionStrategy = ("analytic" if isinstance(parent, ThreePhaseBlackOilModel)
                                      else "numerical")
        self.AutoDiffBackend = parent.AutoDiffBackend

    def get_state_ad(self, state: Any, init: bool = True):
        variables, names, origin = self.get_primary_variables(state)
        if self.reductionStrategy in ("analytic", "numerical"):
            isP = [n == "pressure" for n in names]
            origP = origin[isP.index(True)]
            keep = [p or o != origP for p, o in zip(isP, origin)]
        else:
            raise ValueError(f"Unknown reduction strategy {self.reductionStrategy}")
        pvars = [v for v, k in zip(variables, keep) if k]
        pnames = [n for n, k in zip(names, keep) if k]
        pscale = self.pressureScaling
        has_scaling = pscale is not None and np.size(pscale) > 0
        if init:
            if has_scaling:
                isP = [n.lower() == "pressure" for n in pnames]
                isBHP = [n.lower() == "bhp" for n in pnames]
                pvars[isP.index(True)] = pvars[isP.index(True)]*pscale
                bhpscale = np.mean(pscale)
                if any(isBHP):
                    pvars[isBHP.index(True)] = pvars[isBHP.index(True)]*bhpscale
            pvars = list(self.AutoDiffBackend.init_variables_ad(*pvars))
            if has_scaling:
                pvars[0] = pvars[0]/pscale
                if any(isBHP):
                    pvars[isBHP.index(True)] = pvars[isBHP.index(True)]/bhpscale
        it = iter(pvars)
        variables = [next(it) if k else v for v, k in zip(variables, keep)]
        state = self.init_state_ad(state, variables, names, origin)
        names = pnames
        origin = [o for o, k in zip(origin, keep) if k]
        assert names[0].lower() == "pressure"
        if "laggedStateFunctions" in state:
            for fld, tmp in state["laggedStateFunctions"].items():
                container = dict(state.containers.get(fld, {}))
                container.update(tmp)
                state.containers[fld] = container
        return state, names, origin

    def validate_model(self, *args: Any) -> "PressureModel":
        return super().validate_model(*args)

    def get_model_equations(self, state0: Any, state: Any, dt: float, driving_forces: Any):
        model = self.parentModel
        ncomp = model.get_number_of_components()
        eqs, names, types, state = model.get_model_equations(state0, state, dt, driving_forces)
        ceqs = eqs[:ncomp]
        w = model.get_prop(state, "PressureReductionFactors")
        pressure_equation = 0
        for i in range(len(ceqs)):
            pressure_equation = pressure_equation + w[i]*ceqs[i]
        eqs = [pressure_equation] + list(eqs[ncomp:])
        names = ["pressure"] + list(names[ncomp:])
        types = ["cell"] + list(types[ncomp:])
        state = self.update_reduction_factor_props(state)
        return eqs, names, types, state

    def get_equations(self, state0: Any, state: Any, dt: float, forces: Any, **options: Any):
        return PhysicalModel.get_equations(self, state0, state, dt, forces, **options)

    def prepare_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        model, state = super().prepare_timestep(state, state0, dt, driving_forces)
        state = model.assign_reduction_factor_props(state, state0, dt)
        s = np.asarray(state["s"], dtype=float)
        state["s"] = s/s.sum(axis=1, keepdims=True)
        lfd = model.laggedStateFunctions
        if lfd:
            from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
            tmp = copy_state(state)
            lagged_out = {}
            for f, lagged in lfd.items():
                if not lagged:
                    continue
                sfg = getattr(model.parentModel, f)
                fld = sfg.get_state_function_container_name()
                if fld not in tmp.containers:
                    tmp = sfg.init_state_function_container(tmp)
                for name in lagged:
                    sfg.get(model.parentModel, tmp, name)
                lagged_out[fld] = {n: v for n, v in tmp.containers[fld].items() if n in lagged}
            state["laggedStateFunctions"] = lagged_out
        return model, state

    def validate_state(self, state: Any) -> Any:
        state = super().validate_state(state)
        return self.assign_reduction_factor_props(state, state, 1)

    def update_state(self, state: Any, problem: Any, dx: List[Any], driving_forces: Any):
        pvar = list(problem.primary_variables)
        isP = [n.lower() == "pressure" for n in pvar]
        dx = list(dx)
        if self.pressureScaling is not None and np.size(self.pressureScaling):
            pscale = self.pressureScaling
            dx[isP.index(True)] = dx[isP.index(True)]/pscale
            isBHP = [n == "bhp" for n in pvar]
            if any(isBHP):
                dx[isBHP.index(True)] = dx[isBHP.index(True)]/np.mean(pscale)
        p0 = np.asarray(value(state["pressure"]), dtype=float)
        state, report = self.parentModel.update_state(state, problem, dx, driving_forces)
        if self.verbose and self.verbose > 1:
            print(f"* Pressure: Minimum {p0.min()/BARSA:f} bar, maximum {p0.max()/BARSA:f} bar, "
                  f"mean {p0.mean()/BARSA:f} bar")
            sg = self.parentModel.get_prop(state, "sg")
            print(f"* {int(np.sum(np.asarray(sg) > 0))} two-phase cells")
        state["pressureChange"] = (np.asarray(value(dx[isP.index(True)]), dtype=float)
                                   if any(isP) else np.zeros(p0.shape))
        return state, report

    def check_convergence(self, problem: Any, *args: Any):
        convergence, values, names = self.parentModel.check_convergence(problem)
        convergence, values, names = list(np.ravel(convergence)), list(np.ravel(values)), list(names)
        ptol = self.pressureTol
        itol = self.incTolPressure
        use_ptol = np.isfinite(ptol)
        use_itol = np.isfinite(itol) and self.useIncTol
        assert use_ptol or use_itol, "Pressure model has no valid way of checking convergence!"
        if use_ptol:
            isP = [n.lower() == "pressure" for n in problem.equation_names].index(True)
            mass, w = self.parentModel.get_props(problem.state, "ComponentTotalMass",
                                                 "PressureReductionFactors")
            total = sum(np.asarray(value(m), dtype=float)*np.asarray(value(x), dtype=float)
                        for m, x in zip(mass, w))
            scale = problem.dt/total
            values[0] = float(np.max(np.abs(np.asarray(value(problem.equations[isP]))*scale)))
            convergence[0] = values[0] < ptol
            names[0] = "pressure"
        else:
            names, convergence, values = names[1:], convergence[1:], values[1:]
        if use_itol:
            iteration = problem.iteration_no
            if iteration is not None and iteration > 1 and not np.isnan(iteration):
                inc_type = self.pressureIncTolType.lower()
                if self.parentModel.G.cells.num == 1 and inc_type == "relative":
                    inc_type = "norm"
                dp = np.asarray(problem.state["pressureChange"], dtype=float)
                p0 = np.asarray(value(problem.state["pressure"]), dtype=float)
                if inc_type == "relative":
                    scale = p0.max() - p0.min()
                    if scale == 0:
                        scale = np.max(np.abs(p0))
                elif inc_type == "absolute":
                    scale = 1
                elif inc_type == "norm":
                    scale = np.max(np.abs(p0))
                else:
                    raise ValueError(f"Unknown pressure increment {self.pressureIncTolType}")
                dp = float(np.max(np.abs(dp/scale)))
            else:
                dp = np.inf
            convergence = [dp < itol] + convergence
            names = ["Delta P"] + names
            values = [dp] + values
        return np.array(convergence, dtype=bool), np.array(values, dtype=float), names

    def update_after_convergence(self, *args: Any):
        state, report = super().update_after_convergence(*args)
        for f in ("laggedStateFunctions", "statePressure", "pressureChange",
                  "reductionFactorProps"):
            state.pop(f, None)
        if "sT" in state:
            state["sT"] = np.sum(np.asarray(value(state["s"]), dtype=float), axis=1)
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        state["statePressure"] = copy_state(state)
        return state, report

    def get_surface_densities(self):
        return self.parentModel.get_surface_densities()

    def check_pressure_reduction(self, state0: Any, state: Any, dt: float,
                                 driving_forces: Any) -> None:
        """Print the mean, extremes and mean magnitude of the diagonal of
        the reduced accumulation's Jacobian, per primary variable (with
        pressure scaling undone)."""
        m = self.parentModel
        state = m.reduce_state(state)
        state, pnames, _ = m.get_state_ad(state)
        M, w = m.get_props(state, "ComponentTotalMass", "PressureReductionFactors")
        M0 = m.get_prop(state0, "ComponentTotalMass")
        e = 0
        for i in range(len(M)):
            e = e + (M[i] - M0[i])*w[i]/dt
        for name, J in zip(pnames, e.jac):
            d = np.asarray(J.diagonal() if hasattr(J, "diagonal") else np.diag(J), dtype=float)
            if name.lower() == "pressure" and self.pressureScaling is not None:
                d = d/self.pressureScaling
            if d.size:
                print(f"!! {name}: \n * Mean:  {d.mean():g}\n * Min:   {d.min():g}\n"
                      f" * Max:   {d.max():g}\n *|Min|:  {np.abs(d).min():g}\n"
                      f" *|Mean|: {np.abs(d).mean():g}")

    def setup_state_function_groupings(self, set_defaults: bool = True):
        model = copy(self)
        pmodel = copy(model.parentModel)
        pvt = pmodel.PVTPropertyFunctions
        if model.reductionStrategy == "analytic":
            from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
                ThreePhaseBlackOilModel,
            )
            assert isinstance(pmodel, ThreePhaseBlackOilModel), (
                "Analytical pressure reduction factors currently only implemented for black-oil.")
        elif model.reductionStrategy == "numerical":
            from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.numerical_pressure_reduction_factors import (  # noqa: E501
                NumericalPressureReductionFactors,
            )
            rf = pvt.get_state_function("PressureReductionFactors")
            if not isinstance(rf, NumericalPressureReductionFactors):
                pvt = copy(pvt)
                pvt.functions = dict(pvt.functions)
                pvt = pvt.set_state_function("PressureReductionFactors",
                                             NumericalPressureReductionFactors(pmodel))
        elif model.reductionStrategy != "preselected":
            raise ValueError("Unknown reduction strategy")
        pmodel.PVTPropertyFunctions = pvt
        model.parentModel = pmodel
        return model

    def assign_reduction_factor_props(self, state: Any, state0: Any, dt: float) -> Any:
        if self.reductionStrategy.lower() == "numerical":
            mass0 = self.parentModel.get_prop(state0, "ComponentTotalMass")
            state["reductionFactorProps"] = {"mass0": mass0, "mass": None, "dt": dt,
                                             "pressure": None, "weights": None}
        return state

    def update_reduction_factor_props(self, state: Any) -> Any:
        if "reductionFactorProps" in state:
            props = dict(state["reductionFactorProps"])
            props["pressure"] = value(self.get_prop(state, "pressure"))
            props["weights"] = np.column_stack(
                [value(w) for w in self.parentModel.get_prop(state, "PressureReductionFactors")])
            props["mass"] = np.column_stack(
                [value(m) for m in self.parentModel.get_prop(state, "ComponentTotalMass")])
            state["reductionFactorProps"] = props
        return state

    def set_lagged_terms(self, kind: Any) -> "PressureModel":
        if isinstance(kind, dict):
            lt = kind
        elif kind.lower() == "none":
            lt = {"FlowPropertyFunctions": [], "PVTPropertyFunctions": [],
                  "FlowDiscretization": []}
        elif kind.lower() == "mobilities":
            lt = {"FlowPropertyFunctions": ["Mobility", "ComponentMobility"],
                  "PVTPropertyFunctions": [], "FlowDiscretization": []}
        else:
            raise ValueError(f"Bad type {kind}")
        model = copy(self)
        model.laggedStateFunctions = lt
        return model

    getStateAD = get_state_ad
    getModelEquations = get_model_equations
    getEquations = get_equations
    prepareTimestep = prepare_timestep
    validateState = validate_state
    updateState = update_state
    checkConvergence = check_convergence
    updateAfterConvergence = update_after_convergence
    getSurfaceDensities = get_surface_densities
    checkPressureReduction = check_pressure_reduction
    setupStateFunctionGroupings = setup_state_function_groupings
    assignReductionFactorProps = assign_reduction_factor_props
    updateReductionFactorProps = update_reduction_factor_props
    setLaggedTerms = set_lagged_terms


__all__ = ["PressureModel"]


if __name__ == "__main__":
    import warnings

    from pymrst.modules.sequential.testing import toy_case
    from pymrst.utils.units import DAY

    warnings.simplefilter("ignore")
    CASE = toy_case()
    PM = CASE.seq.pressureModel

    def test_pressure_is_the_only_unknown_without_wells():
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        _, names, _ = PM.get_state_ad(copy_state(CASE.state0))
        assert names == ["pressure"] and PM.reductionStrategy == "analytic"
        print("ok test_pressure_is_the_only_unknown_without_wells")

    def test_the_pressure_step_converges_on_the_pressure_increment():
        state, report, _ = CASE.seq.pressureNonLinearSolver.solve_timestep(
            CASE.state0, DAY, PM, **CASE.forces)
        assert report["Converged"]
        # After convergence MRST drops the increment and keeps the state.
        assert "pressureChange" not in state and "statePressure" in state
        assert np.all(np.isfinite(state["pressure"]))
        print("ok test_the_pressure_step_converges_on_the_pressure_increment")

    def test_the_reduction_report_prints():
        import contextlib
        import io
        from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            PM.check_pressure_reduction(CASE.state0, copy_state(CASE.state0), DAY, CASE.forces)
        assert "!! pressure" in buf.getvalue()
        print("ok test_the_reduction_report_prints")

    test_pressure_is_the_only_unknown_without_wells()
    test_the_pressure_step_converges_on_the_pressure_increment()
    test_the_reduction_report_prints()
    print("\nAll tests passed!")
