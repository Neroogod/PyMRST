"""MRST ``initVariablesAD_oneBlock`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_one_block,
)

initVariablesAD_oneBlock = init_variables_one_block
__all__ = ["init_variables_one_block", "initVariablesAD_oneBlock"]
