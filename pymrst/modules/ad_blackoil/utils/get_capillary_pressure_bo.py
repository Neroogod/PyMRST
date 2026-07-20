"""MRST `ad-blackoil/utils/getCapillaryPressureBO.m`."""


def get_capillary_pressure_bo(fluid, s_w, s_g=None):
    """Return oil-water and oil-gas capillary pressures.

    `pcOW = p_o - p_w` and `pcOG = p_g - p_o`, matching MRST. Missing
    functions evaluate to zero.
    """
    pc_ow = fluid.pc_ow(s_w) if getattr(fluid, "pc_ow", None) else 0.0
    pc_og = (
        fluid.pc_og(s_g)
        if s_g is not None and getattr(fluid, "pc_og", None)
        else 0.0
    )
    return pc_ow, pc_og


__all__ = ["get_capillary_pressure_bo"]
