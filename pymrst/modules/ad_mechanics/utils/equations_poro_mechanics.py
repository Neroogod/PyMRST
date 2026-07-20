"""Poro-mechanics residual (MRST ``equationsPoroMechanics.m`` layout)."""

from pymrst.modules.ad_mechanics.mechanics import MechanicModel, MechanicOperators


def equations_poro_mechanics(mech: MechanicModel, xd, p):
    """Return ``A*xd - gradP*(alpha*p) - rhs`` with MRST residual scaling."""
    return mech.equations(xd, p)


__all__ = ["MechanicModel", "MechanicOperators", "equations_poro_mechanics"]
