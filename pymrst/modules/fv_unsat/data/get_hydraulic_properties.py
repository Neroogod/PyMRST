"""Soil catalog, ported from MRST ``getHydraulicProperties.m``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HydraulicProperties:
    theta_s: float
    theta_r: float
    K_s: float
    alpha: float
    n: float

    @property
    def m(self):
        return 1.0 - 1.0/self.n


_NAMES = [
    "clay", "clayLoam", "loam", "loamySand", "silt", "siltLoam",
    "siltyClay", "siltyClayLoam", "sand", "sandyClay",
    "sandyClayLoam", "sandyLoam", "newMexSample",
]
_THETA_S = [0.38, 0.41, 0.43, 0.41, 0.46, 0.45, 0.36, 0.43,
            0.43, 0.38, 0.39, 0.41, 0.368]
_THETA_R = [0.068, 0.095, 0.078, 0.057, 0.034, 0.067, 0.070, 0.089,
            0.045, 0.100, 0.100, 0.065, 0.102]
_KS_CM_HOUR = [0.20, 0.26, 1.04, 14.59, 0.25, 0.45, 0.02, 0.07,
               29.70, 0.12, 1.31, 4.42, 33.1920]
_ALPHA_PER_CM = [0.008, 0.019, 0.036, 0.124, 0.016, 0.020, 0.005,
                 0.010, 0.145, 0.027, 0.059, 0.075, 0.0335]
_N = [1.09, 1.31, 1.56, 2.28, 1.37, 1.41, 1.09, 1.23, 2.68, 1.23,
      1.48, 1.89, 2.0]


def get_hydraulic_properties(soil_name):
    """Return SI-valued van Genuchten/Mualem properties for ``soil_name``."""
    try:
        index = _NAMES.index(str(soil_name))
    except ValueError as exc:
        raise ValueError(
            f"unknown soil {soil_name!r}; expected one of {', '.join(_NAMES)}"
        ) from exc
    return HydraulicProperties(
        theta_s=_THETA_S[index],
        theta_r=_THETA_R[index],
        K_s=_KS_CM_HOUR[index]*0.01/3600.0,
        alpha=_ALPHA_PER_CM[index]/0.01,
        n=_N[index],
    )


getHydraulicProperties = get_hydraulic_properties

__all__ = ["HydraulicProperties", "get_hydraulic_properties",
           "getHydraulicProperties"]
