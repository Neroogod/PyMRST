"""CoolProp-backed compositional mixture (MRST ``CoolPropsCompositionalMixture.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.eos import Mixture


def _coolprop_module():
    try:
        import CoolProp.CoolProp as CP  # type: ignore
    except Exception as exc:
        raise ImportError("CoolPropsCompositionalMixture requires the optional CoolProp package") from exc
    return CP


def _as_names(names) -> tuple[str, ...]:
    if isinstance(names, str):
        return (names,)
    return tuple(str(name) for name in names)


class CoolPropsCompositionalMixture(Mixture):
    """Create a compositional mixture by querying the optional CoolProp package."""

    def __init__(self, names):
        names = _as_names(names)
        CP = _coolprop_module()
        valid = self.getFluidList()
        valid_lookup = {name.lower(): name for name in valid}
        missing = [name for name in names if name.lower() not in valid_lookup]
        if missing:
            raise ValueError(
                "Unable to create fluid. The following names were not known to coolprops: "
                + " ".join(missing)
            )
        canonical = tuple(valid_lookup[name.lower()] for name in names)
        Tcrit = np.array([CP.Props1SI(name, "Tcrit") for name in canonical], dtype=float)
        Pcrit = np.array([CP.Props1SI(name, "Pcrit") for name in canonical], dtype=float)
        rhocrit = np.array([CP.Props1SI(name, "rhocrit") for name in canonical], dtype=float)
        acentric = np.array([CP.Props1SI(name, "acentric") for name in canonical], dtype=float)
        molarMass = np.array([CP.Props1SI(name, "molarmass") for name in canonical], dtype=float)
        if not all(np.all(np.isfinite(v)) for v in (Tcrit, Pcrit, rhocrit, acentric, molarMass)):
            raise ValueError("CoolProp returned non-finite fluid properties")
        super().__init__(tuple(canonical), Tcrit, Pcrit, molarMass / rhocrit, acentric, molarMass)

    @staticmethod
    def getFluidList() -> tuple[str, ...]:
        CP = _coolprop_module()
        return tuple(str(CP.get_global_param_string("FluidsList")).split(","))

    @staticmethod
    def get_fluid_list() -> tuple[str, ...]:
        return CoolPropsCompositionalMixture.getFluidList()


def coolprops_compositional_mixture(names) -> CoolPropsCompositionalMixture:
    """Pythonic constructor for :class:`CoolPropsCompositionalMixture`."""
    return CoolPropsCompositionalMixture(names)


__all__ = ["CoolPropsCompositionalMixture", "coolprops_compositional_mixture"]
