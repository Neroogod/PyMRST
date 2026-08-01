"""MRST ``ad-props/utils``: table extension, regions, and PVT interpolation."""

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import ALL, get_reg_map
from pymrst.modules.ad_props.utils.handle_regions import Regions, handle_regions
from pymrst.modules.ad_props.utils.interp_2d_pvt import interp_2d_pvt
from pymrst.modules.ad_props.utils.interp_2d_table import Table2D, interp_2d_table
from pymrst.modules.ad_props.utils.interp_pvt import (
    PVTTable,
    make_pvt_table,
    pvt_interp,
)
from pymrst.modules.ad_props.utils.interp_reg import dinterp_reg, interp_reg
from pymrst.modules.ad_props.utils.interp_reg_pvt import interp_reg_pvt
from pymrst.modules.ad_props.utils.interp_regular_2d_table import (
    RegularTable2D,
    interp_regular_2d_table,
)
from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
    Branch,
    PackedPVT,
    preprocess_table_pvt,
)

__all__ = [
    "ALL", "Branch", "PVTTable", "PackedPVT", "Regions", "Table2D",
    "dinterp_reg", "extend_tab", "get_reg_map", "handle_regions",
    "interp_2d_pvt", "interp_2d_table", "interp_reg", "interp_reg_pvt",
    "interp_regular_2d_table", "make_pvt_table", "preprocess_table_pvt",
    "pvt_interp", "RegularTable2D",
]
