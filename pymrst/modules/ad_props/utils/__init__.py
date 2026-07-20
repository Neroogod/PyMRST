"""MRST-shaped `ad-props/utils` namespace."""

from pymrst.modules.ad_props.tables import PVTTable, extend_tab, interp_table, make_pvt_table, pvt_interp

interp_pvt = pvt_interp

__all__ = [
    "PVTTable",
    "extend_tab",
    "interp_pvt",
    "interp_table",
    "make_pvt_table",
    "pvt_interp",
]
