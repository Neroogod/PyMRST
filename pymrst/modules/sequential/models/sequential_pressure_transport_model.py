"""MRST ``sequential/models/SequentialPressureTransportModel.m``.

A sequential split of a fully implicit model: each step function solves the
pressure model to convergence with its own `NonLinearSolver` (no timestep
cuts, no error on failure), then the transport model with another (with
relaxation). Pressure boundary conditions become fixed reservoir fluxes
('resflux') from the pressure step's face fluxes for the transport solve. By
default (`stepFunctionIsLinear`) one pressure-transport pass is the step;
with an outer loop, the pass repeats until the volume discrepancy
(`|sT - 1|`), the saturation change across transport and, if a
`parentModel` is given, the parent's fully implicit residual converge. The
pressure and transport models validate, prepare and update together;
driving forces and variables are the pressure model's.

`SequentialOilWaterModel` (in `seq_model.py`) is PyMRST's earlier
sequential model, kept for its callers.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Dict, Tuple

import numpy as np

from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.ad_oo.physical_model import merge_options
from pymrst.modules.ad_core.ad_oo.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel
from pymrst.modules.ad_core.utils.get_reservoir_model import get_reservoir_model
from pymrst.modules.ad_core.utils.set_reservoir_model import set_reservoir_model


class SequentialPressureTransportModel(ReservoirModel):
    """MRST `SequentialPressureTransportModel`."""

    def __init__(self, pressureModel: Any, transportModel: Any, **options: Any):
        super().__init__(transportModel.G)
        self.pressureModel = pressureModel        # Model for computing the pressure
        self.transportModel = transportModel      # Model for the transport subproblem
        self.parentModel = None                   # Used for outer loop convergence if present
        self.pressureNonLinearSolver = None
        self.transportNonLinearSolver = None
        self.reupdatePressure = None
        self.maxOuterIterations = np.inf
        self.volumeDiscrepancyTolerance = 1e-3
        self.incTolSaturation = 1e-3
        self.outerCheckParentConvergence = True
        self.outerCheckPressureConvergence = False
        self.relaxationThreshold = np.inf
        self.stepFunctionIsLinear = True
        self.reupdatePressure = False
        solver_prm = {"transportLinearSolver": options.pop("transportLinearSolver", None),
                      "pressureLinearSolver": options.pop("pressureLinearSolver", None)}
        merge_options(self, options)
        if self.parentModel is None:
            source = transportModel.parentModel if isinstance(transportModel, WrapperModel) \
                else transportModel
        else:
            source = self.parentModel
        self.water, self.oil, self.gas = source.water, source.oil, source.gas
        if self.pressureNonLinearSolver is None:
            self.pressureNonLinearSolver = NonLinearSolver()
        self.pressureNonLinearSolver.identifier = "PRESSURE"
        if self.transportNonLinearSolver is None:
            self.transportNonLinearSolver = NonLinearSolver(useRelaxation=True)
        self.transportNonLinearSolver.identifier = "TRANSPORT"
        if solver_prm["pressureLinearSolver"] is not None:
            self.pressureNonLinearSolver.LinearSolver = solver_prm["pressureLinearSolver"]
        self.pressureNonLinearSolver.maxTimestepCuts = 0
        self.pressureNonLinearSolver.errorOnFailure = False
        if solver_prm["transportLinearSolver"] is not None:
            self.transportNonLinearSolver.LinearSolver = solver_prm["transportLinearSolver"]
        self.outerCheckParentConvergence = (self.outerCheckParentConvergence
                                            and self.parentModel is not None)
        if self.outerCheckParentConvergence:
            self.transportNonLinearSolver.maxTimestepCuts = 0
            self.incTolSaturation = np.inf
        self.transportNonLinearSolver.errorOnFailure = False
        self.FacilityModel = None                 # Handled by subclasses

    def step_function(self, state: Any, state0: Any, dt: float, driving_forces: Any,
                      linsolve: Any, nls: Any, iteration: int, **options: Any):
        state = dict_like_copy(state)
        state["iteration"] = iteration
        (state, pressure_state, pressure_report, transport_report, pressure_ok,
         transport_ok, force_arg) = self.solve_pressure_transport(state, state0, dt,
                                                                  driving_forces, nls, iteration)
        converged_step = pressure_ok and transport_ok
        converged = converged_step
        if converged and not self.stepFunctionIsLinear:
            converged, values, state = self.check_outer_convergence(
                state, state0, dt, driving_forces, iteration, pressure_state)
            if transport_report["Iterations"] == 0 and not self.outerCheckParentConvergence:
                converged = np.asarray(converged) | True
        else:
            values = np.atleast_1d(
                pressure_report["StepReports"][-1]["NonlinearReport"][-1]["Residuals"])[:1]
        failure = (not converged_step) and not self.stepFunctionIsLinear
        failure_msg = "Unable to converge transport for given time-step." if failure else ""
        if not pressure_ok:
            converged = np.asarray(converged) & False
        report = self.make_step_report(Failure=failure, Converged=bool(np.all(converged)),
                                       FailureMsg=failure_msg,
                                       ResidualsConverged=converged, Residuals=values)
        report["PressureSolver"] = pressure_report
        report["TransportSolver"] = transport_report
        if self.reupdatePressure and np.all(converged):
            state, _, _ = self.pressureNonLinearSolver.solve_timestep(
                state0, dt, self.pressureModel, initialGuess=state, **force_arg)
            _, state = self.transportModel.get_equations(state0, state, dt, driving_forces,
                                                         resOnly=True, iteration=np.inf)
            state = self.reduce_state(state)
        state.pop("iteration", None)
        return state, report

    def solve_pressure_transport(self, state: Any, state0: Any, dt: float,
                                 driving_forces: Any, nls: Any, iteration: int):
        psolver = self.pressureNonLinearSolver
        tsolver = self.transportNonLinearSolver
        prmodel = get_reservoir_model(self.pressureModel)
        trmodel = get_reservoir_model(self.transportModel)
        if iteration > 1 and getattr(prmodel, "FacilityModel", None) is not None:
            prmodel, trmodel = copy(prmodel), copy(trmodel)
            prmodel.FacilityModel = copy(prmodel.FacilityModel)
            trmodel.FacilityModel = copy(trmodel.FacilityModel)
            for wm in list(prmodel.FacilityModel.WellModels) + list(trmodel.FacilityModel.WellModels):
                wm.doUpdatePressureDrop = False
            self.pressureModel = set_reservoir_model(self.pressureModel, prmodel)
            self.transportModel = set_reservoir_model(self.transportModel, trmodel)
        force_arg = self.pressureModel.get_driving_forces(driving_forces)
        state, pressure_report, _ = psolver.solve_timestep(state0, dt, self.pressureModel,
                                                           initialGuess=state, **force_arg)
        pressure_state = state
        pressure_ok = bool(pressure_report["Converged"]) or psolver.continueOnFailure
        if pressure_ok:
            bc = _field(driving_forces, "bc")
            if bc is not None:
                kinds = [str(t).lower() for t in bc.type]
                isDir = np.array([t == "pressure" for t in kinds], dtype=bool)
                if np.any(isDir):
                    transport_forces = dict(driving_forces)
                    G = self.pressureModel.G
                    dirFace = np.asarray(bc.face, dtype=np.int64)[isDir]
                    q = np.asarray(state["flux"], dtype=float)[dirFace, :].sum(axis=1)
                    sgn = 1 - 2*(np.asarray(G.faces.neighbors)[dirFace, 1] < 0)
                    new_bc = copy(bc)
                    new_bc.value = np.array(bc.value, dtype=float, copy=True)
                    new_bc.value[isDir] = sgn*q
                    new_bc.type = ["resflux" if d else t for t, d in zip(bc.type, isDir)]
                    transport_forces["bc"] = new_bc
                    force_arg = self.transportModel.get_driving_forces(transport_forces)
            state["timestep"] = dt
            state["pressure_full"] = state["pressure"]
            state, transport_report, _ = tsolver.solve_timestep(state0, dt, self.transportModel,
                                                                initialGuess=state, **force_arg)
            if iteration > self.relaxationThreshold:
                nls.relaxationParameter = nls.minRelaxation
            if not self.stepFunctionIsLinear and nls.relaxationParameter != 1:
                state = self.apply_relaxation(state, pressure_state, nls)
            transport_ok = bool(transport_report["Converged"])
        else:
            transport_ok = False
            transport_report = None
        return (state, pressure_state, pressure_report, transport_report, pressure_ok,
                transport_ok, force_arg)

    def apply_relaxation(self, state: Any, pressure_state: Any, nls: Any) -> Any:
        _, names, origin = self.transportModel.get_state_ad(state, False)
        rclass = type(get_reservoir_model(self.transportModel)).__name__
        names = [n for n, o in zip(names, origin) if o.lower() in (rclass.lower(), "transportmodel")]
        if getattr(self.transportModel, "implicitType", "").lower() == "wells":
            names = names + ["pressure"]
        isS = [n.lower() in ("sw", "so", "sg") for n in names]
        if any(isS):
            names = [n for n, s in zip(names, isS) if not s] + ["s"]
        if any(n.lower() == "x" for n in names):
            names = [n for n in names if n.lower() != "x"] + ["rs", "rv"]
        w = nls.relaxationParameter
        for name in names:
            vp = self.transportModel.get_prop(pressure_state, name)
            vt = self.transportModel.get_prop(state, name)
            state = self.transportModel.set_prop(state, name, (1 - w)*np.asarray(vp)
                                                 + w*np.asarray(vt))
        return state

    def check_outer_convergence(self, state: Any, state0: Any, dt: float, driving_forces: Any,
                                iteration: int, pressure_state: Any):
        resnames, converged, values = [], [], []
        tol_vol = self.volumeDiscrepancyTolerance
        if np.isfinite(tol_vol):
            sT = state["sT"] if "sT" in state else np.sum(np.asarray(state["s"]), axis=1)
            v = float(np.max(np.abs(np.asarray(sT) - 1)))
            values.append(v)
            converged.append(v <= tol_vol)
            resnames.append("Volume error")
        tol_inc = self.incTolSaturation
        if np.isfinite(tol_inc):
            ds = float(np.max(np.abs(np.asarray(state["s"]) - np.asarray(pressure_state["s"]))))
            values.append(ds)
            converged.append(ds <= tol_inc)
            resnames.append("Saturation increment")
        if self.parentModel is not None and self.outerCheckParentConvergence:
            rmodel = get_reservoir_model(self.parentModel)
            if type(rmodel).__name__ == "ThreePhaseCompositionalModel":
                rmodel = copy(rmodel)
                rmodel.incTolPressure = np.inf
                self.parentModel = set_reservoir_model(self.parentModel, rmodel)
            s = np.asarray(state["s"], dtype=float)
            state["s"] = s/s.sum(axis=1, keepdims=True)
            problem, state = self.parentModel.get_equations(state0, state, dt, driving_forces,
                                                            resOnly=True, iteration=np.inf)
            state = self.parentModel.reduce_state(state, False)
            c, v, n = self.parentModel.check_convergence(problem)
            converged += list(np.ravel(c))
            values += list(np.ravel(v))
            resnames += list(n)
        if self.outerCheckPressureConvergence:
            from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
            tmp = copy_state(state)
            s = np.asarray(state["s"], dtype=float)
            tmp["s"] = s/s.sum(axis=1, keepdims=True)
            problem, _ = self.pressureModel.get_equations(state0, tmp, dt, driving_forces,
                                                          resOnly=True, iteration=np.inf)
            c, v, n = self.pressureModel.check_convergence(problem)
            keep = [x.lower() != "delta p" for x in n]
            converged += [x for x, k in zip(np.ravel(c), keep) if k]
            values += [x for x, k in zip(np.ravel(v), keep) if k]
            resnames += [x for x, k in zip(n, keep) if k]
        converged = np.asarray(converged, dtype=bool) | (iteration > self.maxOuterIterations)
        if self.verbose:
            for name, value_, ok in zip(resnames, values, converged):
                print(f"{name:>24s}: {value_:.2e} {'(ok)' if ok else ''}")
        return converged, np.asarray(values, dtype=float), state

    def get_active_phases(self):
        return self.transportModel.get_active_phases()

    def validate_state(self, state: Any) -> Any:
        state = self.pressureModel.validate_state(state)
        return self.transportModel.validate_state(state)

    def update_for_changed_controls(self, state: Any, forces: Any):
        model = copy(self)
        model.pressureModel, state = self.pressureModel.update_for_changed_controls(state, forces)
        model.transportModel, _ = self.transportModel.update_for_changed_controls(state, forces)
        if self.parentModel is not None:
            model.parentModel, _ = self.parentModel.update_for_changed_controls(state, forces)
        return model, state

    def prepare_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        model = copy(self)
        model.pressureModel, state = self.pressureModel.prepare_timestep(state, state0, dt,
                                                                         driving_forces)
        model.transportModel, _ = self.transportModel.prepare_timestep(state, state0, dt,
                                                                       driving_forces)
        if self.parentModel is not None:
            model.parentModel, _ = self.parentModel.prepare_timestep(state, state0, dt,
                                                                     driving_forces)
        return model, state

    def prepare_report_step(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        model = copy(self)
        model.pressureModel, state = self.pressureModel.prepare_report_step(state, state0, dt,
                                                                            driving_forces)
        model.transportModel, _ = self.transportModel.prepare_report_step(state, state0, dt,
                                                                          driving_forces)
        if self.parentModel is not None:
            model.parentModel, _ = self.parentModel.prepare_report_step(state, state0, dt,
                                                                        driving_forces)
        return model, state

    def validate_model(self, *args: Any) -> "SequentialPressureTransportModel":
        model = copy(self)
        if hasattr(model.pressureModel, "extraWellSolOutput"):
            model.pressureModel = copy(model.pressureModel)
            model.pressureModel.extraWellSolOutput = True
        model.pressureModel = model.pressureModel.validate_model(*args)
        model.transportModel = model.transportModel.validate_model(*args)
        if model.parentModel is not None:
            model.parentModel = model.parentModel.validate_model(*args)
        return model

    def get_variable_field(self, name: str, *args: Any, **kwargs: Any):
        return self.pressureModel.get_variable_field(name, *args, **kwargs)

    def get_maximum_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        dt_p = self.pressureModel.get_maximum_timestep(state, state0, dt, driving_forces)
        dt_t = self.transportModel.get_maximum_timestep(state, state0, dt, driving_forces)
        return min(dt_p, dt_t)

    def validate_driving_forces(self, forces: Any, *args: Any):
        return self.pressureModel.validate_driving_forces(forces, *args)

    def update_after_convergence(self, state0: Any, state: Any, dt: float, driving_forces: Any):
        if (self.parentModel is not None and not self.stepFunctionIsLinear
                and self.outerCheckParentConvergence):
            return self.parentModel.update_after_convergence(state0, state, dt, driving_forces)
        return state, None

    stepFunction = step_function
    solvePressureTransport = solve_pressure_transport
    applyRelaxation = apply_relaxation
    checkOuterConvergence = check_outer_convergence
    getActivePhases = get_active_phases
    validateState = validate_state
    updateForChangedControls = update_for_changed_controls
    prepareTimestep = prepare_timestep
    prepareReportstep = prepare_report_step
    validateModel = validate_model
    getVariableField = get_variable_field
    getMaximumTimestep = get_maximum_timestep
    validateDrivingForces = validate_driving_forces
    updateAfterConvergence = update_after_convergence


def dict_like_copy(state: Any) -> Any:
    from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
    return copy_state(state)


def _field(forces: Any, name: str) -> Any:
    return forces.get(name) if isinstance(forces, dict) else getattr(forces, name, None)


__all__ = ["SequentialPressureTransportModel"]


if __name__ == "__main__":
    import warnings

    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.sequential.testing import toy_case

    warnings.simplefilter("ignore")
    CASE = toy_case()

    def test_a_split_run_floods_and_conserves_volume():
        _, states, report = simulate_schedule_ad(CASE.state0, CASE.seq, CASE.schedule)
        sw = [np.asarray(st["s"])[:, 0] for st in states]
        assert sw[-1][0] > sw[0][0] > 0.2, "water enters at the left"
        assert np.all(np.diff(sw[-1]) <= 1e-12), "and decreases along the flood"
        sT = np.asarray(states[-1]["sT"])
        assert np.max(np.abs(sT - 1)) < 1e-3
        assert np.all(np.asarray(report["Converged"]))
        print("ok test_a_split_run_floods_and_conserves_volume")

    def test_each_step_is_one_pressure_transport_pass():
        assert CASE.seq.stepFunctionIsLinear
        assert CASE.seq.pressureNonLinearSolver.maxTimestepCuts == 0
        assert CASE.seq.transportNonLinearSolver.useRelaxation
        print("ok test_each_step_is_one_pressure_transport_pass")

    test_a_split_run_floods_and_conserves_volume()
    test_each_step_is_one_pressure_transport_pass()
    print("\nAll tests passed!")
