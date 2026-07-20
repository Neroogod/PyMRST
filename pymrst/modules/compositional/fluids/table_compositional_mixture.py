"""Table-backed compositional mixture (MRST ``TableCompositionalMixture.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.utils.cool_prop_fluids_structs import coolPropFluidsStructs


def _as_names(names) -> tuple[str, ...]:
    if isinstance(names, str):
        return (names,)
    return tuple(str(name) for name in names)


class TableCompositionalMixture(Mixture):
    """Create a compositional mixture from MRST's stored CoolProp table."""

    def __init__(self, names, outnames=None):
        names = _as_names(names)
        outnames = names if outnames is None else _as_names(outnames)
        if len(names) != len(outnames):
            raise AssertionError("names and outnames must have the same length")
        fluids = coolPropFluidsStructs()
        lookup = {f.name.lower(): f for f in fluids}
        missing = [name for name in names if name.lower() not in lookup]
        if missing:
            raise ValueError(
                "Unable to create fluid. The following names were not known to CoolProps table: "
                + " ".join(missing)
            )
        rows = [lookup[name.lower()] for name in names]
        Tcrit = np.array([r.Tcrit for r in rows], dtype=float)
        Pcrit = np.array([r.Pcrit for r in rows], dtype=float)
        rhocrit = np.array([r.rhocrit for r in rows], dtype=float)
        acentric = np.array([r.acentric for r in rows], dtype=float)
        molarMass = np.array([r.molarmass for r in rows], dtype=float)
        Vcrit = molarMass / rhocrit
        super().__init__(
            names=tuple(outnames),
            Tcrit=Tcrit,
            Pcrit=Pcrit,
            Vcrit=Vcrit,
            acentric=acentric,
            molarMass=molarMass,
            name="CoolProp - Tabulated",
        )

    @staticmethod
    def getFluidList() -> tuple[str, ...]:
        """Return valid names in MRST table order."""
        return tuple(f.name for f in coolPropFluidsStructs())

    @staticmethod
    def get_fluid_list() -> tuple[str, ...]:
        """Pythonic alias for :meth:`getFluidList`."""
        return TableCompositionalMixture.getFluidList()


def table_compositional_mixture(names, outnames=None) -> TableCompositionalMixture:
    """Pythonic constructor for :class:`TableCompositionalMixture`."""
    return TableCompositionalMixture(names, outnames=outnames)


__all__ = ["TableCompositionalMixture", "table_compositional_mixture"]
