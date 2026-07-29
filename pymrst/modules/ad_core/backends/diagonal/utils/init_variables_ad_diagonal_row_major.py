"""MRST ``initVariablesAD_diagonalRowMajor`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_diagonal_grouped,
)


def init_variables_ad_diagonal_row_major(*values, types=None, use_mex=False):
    return init_variables_diagonal_grouped(*values, types=types, row_major=True)


initVariablesAD_diagonalRowMajor = init_variables_ad_diagonal_row_major
__all__ = [
    "init_variables_ad_diagonal_row_major",
    "initVariablesAD_diagonalRowMajor",
]
