"""MRST model/problem wrapper for CNV and material-balance convergence."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.utils.convergence import (
    cnv_mb_values,
    formation_volume_factor_avg,
)
from pymrst.utils.adi import value


def _method(obj, *names):
    for name in names:
        candidate = getattr(obj, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(f"none of {names!r} found")


def cnv_mb_convergence_report(model, problem):
    """Return MRST ``(converged, values, evaluated, names)``."""
    state = problem.state
    pore_volume = np.asarray(model.operators.pv, dtype=float)
    shrinkage = _method(
        model.FlowPropertyFunctions, "getProperty", "get_property"
    )(model, state, "ShrinkageFactors")
    phase_result = _method(model, "getPhaseNames", "get_phase_names")()
    if (
        isinstance(phase_result, tuple)
        and len(phase_result) == 2
        and not isinstance(phase_result[0], str)
    ):
        phases, equation_names = phase_result
    else:
        equation_names = list(phase_result)
        phases = [str(name)[0].upper() for name in equation_names]
    equations = list(problem.equations)
    evaluated = np.zeros(len(equations), dtype=bool)
    residuals = []
    for name in equation_names:
        index = int(_method(
            problem, "indexOfEquationName", "index_of_equation_name"
        )(name))
        residuals.append(np.asarray(value(equations[index]), dtype=float))
        evaluated[index] = True
    b_average = formation_volume_factor_avg(shrinkage)
    cnv, material_balance = cnv_mb_values(
        residuals, b_average, pore_volume, float(problem.dt)
    )
    converged = np.concatenate([
        cnv <= float(model.toleranceCNV),
        material_balance <= float(model.toleranceMB),
    ])
    values = np.concatenate([cnv, material_balance])
    names = (
        [f"CNV_{phase}" for phase in phases]
        + [f"MB_{phase}" for phase in phases]
    )
    return converged, values, evaluated, names


CNV_MBConvergence = cnv_mb_convergence_report

__all__ = ["cnv_mb_convergence_report", "CNV_MBConvergence"]
