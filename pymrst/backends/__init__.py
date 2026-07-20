"""Optional numeric backends for experimental GPU/tensor execution."""

from pymrst.backends.array_backend import ArrayBackend, get_backend
from pymrst.backends.newton import (
    LinearSolveReport,
    NewtonReport,
    finite_difference_jacobian,
    torch_jvp_newton_step,
    torch_matrix_free_gmres,
    torch_newton,
)
from pymrst.backends.tables import BackendPVTTable, interp_table_backend, pvt_interp_backend

__all__ = [
    "ArrayBackend",
    "BackendPVTTable",
    "LinearSolveReport",
    "NewtonReport",
    "finite_difference_jacobian",
    "get_backend",
    "interp_table_backend",
    "pvt_interp_backend",
    "torch_jvp_newton_step",
    "torch_matrix_free_gmres",
    "torch_newton",
]
