"""MRST ``interpPVT.m`` PVT table interpolation entry point."""

from pymrst.modules.ad_props.tables import PVTTable, make_pvt_table, pvt_interp


def interp_pvt(tab: PVTTable, p, rs, sat_flag):
    """Evaluate a PVT table using MRST's validated parallel interpolation path."""
    return pvt_interp(tab, p, rs, sat_flag)

__all__ = ["PVTTable", "interp_pvt", "make_pvt_table", "pvt_interp"]
