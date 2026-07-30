"""MRST ``ad-blackoil/utils/getbG_BO.m`` compatibility function."""

from pymrst.modules.ad_blackoil.utils.get_b_o_bo import _raw_fluid


def get_b_g_bo(model, pressure, rv=None, is_liquid=None):
    """Evaluate reciprocal gas formation volume factor.

    MRST calls the wet-gas three-argument PVT function only when vaporized oil
    is enabled. For dry gas, ``rv`` and ``is_liquid`` are ignored.
    """
    fluid = _raw_fluid(model)
    if fluid is None:
        raise AttributeError("model must define fluid or raw_fluid")
    b_g = getattr(fluid, "b_g", getattr(fluid, "bG", None))
    if b_g is None:
        raise AttributeError("model fluid must define b_g or bG")
    if bool(getattr(model, "vapoil", False)):
        return b_g(pressure, rv, is_liquid)
    return b_g(pressure)


getbG_BO = get_b_g_bo

__all__ = ["get_b_g_bo", "getbG_BO"]
