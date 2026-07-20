"""Compatibility alias for MRST ``blackoil/convertRSRVtoKValue.m``."""

from pymrst.modules.compositional.fluids.blackoil.convert_rs_rv_to_k_value import (
    BlackOilPropertyTables,
    KValueTables,
    RSRVKValueProperties,
    convert_rs_rv_to_k_value,
)

convertRSRVtoKValue = convert_rs_rv_to_k_value
convert_rsrv_to_k_value = convert_rs_rv_to_k_value

__all__ = [
    "BlackOilPropertyTables",
    "KValueTables",
    "RSRVKValueProperties",
    "convert_rs_rv_to_k_value",
    "convert_rsrv_to_k_value",
    "convertRSRVtoKValue",
]
