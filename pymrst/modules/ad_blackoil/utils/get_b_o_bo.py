"""MRST ``ad-blackoil/utils/getbO_BO.m`` compatibility function."""


def _raw_fluid(model):
    """Return the MRST-style fluid object owned by ``model``."""
    return getattr(model, "raw_fluid", getattr(model, "fluid", None))


def get_b_o_bo(model, pressure, rs=None, is_saturated=None):
    """Evaluate reciprocal oil formation volume factor.

    MRST calls the live-oil three-argument PVT function only when dissolved
    gas is enabled. For dead oil, ``rs`` and ``is_saturated`` are deliberately
    ignored and the pressure-only function is used.
    """
    fluid = _raw_fluid(model)
    if fluid is None:
        raise AttributeError("model must define fluid or raw_fluid")
    b_o = getattr(fluid, "b_o", getattr(fluid, "bO", None))
    if b_o is None:
        raise AttributeError("model fluid must define b_o or bO")
    if bool(getattr(model, "disgas", False)):
        return b_o(pressure, rs, is_saturated)
    return b_o(pressure)


getbO_BO = get_b_o_bo

__all__ = ["get_b_o_bo", "getbO_BO"]
