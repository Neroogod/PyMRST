"""MRST ``initVariablesAD_diagonal`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_diagonal_grouped,
)


def init_variables_ad_diagonal(*values, types=None, use_mex=False):
    return init_variables_diagonal_grouped(*values, types=types, row_major=False)


initVariablesAD_diagonal = init_variables_ad_diagonal
__all__ = ["init_variables_ad_diagonal", "initVariablesAD_diagonal"]
