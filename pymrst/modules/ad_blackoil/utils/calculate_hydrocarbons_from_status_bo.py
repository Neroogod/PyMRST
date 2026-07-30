"""MRST ``calculateHydrocarbonsFromStatusBO`` variable switching."""

import numpy as np


def _property(fluid, snake_name, matlab_name):
    prop = getattr(fluid, snake_name, getattr(fluid, matlab_name, None))
    if isinstance(prop, (list, tuple)):
        prop = prop[0]
    if prop is None:
        raise AttributeError(f"fluid must define {snake_name} or {matlab_name}")
    return prop


def _status_masks(status):
    if isinstance(status, (list, tuple)) and len(status) == 3:
        return tuple(np.asarray(mask, dtype=bool) for mask in status)
    encoded = np.asarray(status)
    return encoded == 1, encoded == 2, encoded == 3


def calculate_hydrocarbons_from_status_bo(
    fluid, status, oil_space, x, rs, rv, pressure,
    disgas: bool, vapoil: bool,
):
    """Resolve ``sG``, ``rs`` and ``rv`` from MRST's switched variable.

    ``oil_space`` is the hydrocarbon saturation ``1-sW`` used when only gas
    is present. ``status`` may be encoded as 1/2/3 or supplied as MRST's
    three logical masks. Numeric and ADI primary variables are preserved.
    """
    oil, gas, both = _status_masks(status)
    f1, f2, f3 = (mask.astype(float) for mask in (oil, gas, both))
    s_g = f2*oil_space + f3*x

    if disgas:
        rs_sat = _property(fluid, "rs_sat", "rsSat")(pressure)
        rs = (1.0-f1)*rs_sat + f1*x
    else:
        rs_sat = rs

    if vapoil:
        rv_sat = _property(fluid, "rv_sat", "rvSat")(pressure)
        rv = (1.0-f2)*rv_sat + f2*x
    else:
        rv_sat = rv
    return s_g, rs, rv, rs_sat, rv_sat


calculateHydrocarbonsFromStatusBO = calculate_hydrocarbons_from_status_bo

__all__ = [
    "calculate_hydrocarbons_from_status_bo",
    "calculateHydrocarbonsFromStatusBO",
]
