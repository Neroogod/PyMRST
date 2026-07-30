"""Wrap a callable ``x = fn(A, b)`` as an AD linear solver."""

from __future__ import annotations

from typing import Callable

import numpy as np

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD


class HandleLinearSolverAD(LinearSolverAD):
    def __init__(self, function: Callable, **options):
        if not callable(function):
            raise TypeError("function must be callable")
        super().__init__(**options)
        self.function = function
        self.fcn_handle = function

    def solve_linear_system(self, A, b):
        result = np.asarray(self.function(A, b), dtype=float).ravel()
        if result.size != A.shape[1]:
            raise ValueError("linear solver handle returned the wrong size")
        self.last_solve_report = {}
        return result

    def get_description(self):
        return "Wrapper for function handle."


__all__ = ["HandleLinearSolverAD"]
