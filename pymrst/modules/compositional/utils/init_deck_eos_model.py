"""Initialize EOS model from parsed deck PROPS (MRST ``initDeckEOSModel.m``)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from pymrst.modules.compositional.eos import R_GAS, Mixture

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from pymrst.modules.compositional.models.eos import EquationOfStateModel

ATM = 101325.0


def _field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_field(obj, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _props(deck):
    props = _field(deck, "PROPS", None)
    if props is None:
        raise ValueError("deck must contain a PROPS section")
    return props


def _vector(value, dtype=float):
    return np.asarray(value, dtype=dtype).reshape(-1)


def _names(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _first_eos_region(eos_type):
    if isinstance(eos_type, (list, tuple)):
        if len(eos_type) > 1:
            warnings.warn("Multiple EOS regions not supported, using region 1", RuntimeWarning, stacklevel=2)
        return eos_type[0] if eos_type else "PR"
    arr = np.asarray(eos_type, dtype=object)
    if arr.ndim and arr.size > 1:
        warnings.warn("Multiple EOS regions not supported, using region 1", RuntimeWarning, stacklevel=2)
        return arr.flat[0]
    return str(arr.item()) if arr.shape == () else str(eos_type)


def _omega_default(eos_type: str, kind: str) -> float:
    key = str(eos_type).upper()
    if kind == "A":
        return 0.457235529 if key == "PR" else 0.4274802
    return 0.077796074 if key == "PR" else 0.08664035


def _omega_value(values, eos_type: str, kind: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim > 1:
        arr = arr[0, :]
    arr = arr.reshape(-1).copy()
    default = _omega_default(eos_type, kind)
    arr[~np.isfinite(arr)] = default
    return float(arr[0])


def initDeckEOSModel(deck) -> "EquationOfStateModel":
    """Set up an :class:`EquationOfStateModel` from a parsed deck-like object."""
    from pymrst.modules.compositional.models.eos import EquationOfStateModel

    props = _props(deck)
    names = _names(_field(props, "CNAMES"))
    Tcrit = _vector(_field(props, "TCRIT"))
    Pcrit = _vector(_field(props, "PCRIT"))
    if _has_field(props, "VCRIT"):
        Vcrit = _vector(_field(props, "VCRIT"))
    else:
        Zcrit = _vector(_field(props, "ZCRIT"))
        Vcrit = R_GAS * Zcrit * Tcrit / Pcrit
    acf = _vector(_field(props, "ACF"))
    mw = _vector(_field(props, "MW"))
    ncomp = len(names)
    for label, arr in (("TCRIT", Tcrit), ("PCRIT", Pcrit), ("VCRIT/ZCRIT", Vcrit), ("ACF", acf), ("MW", mw)):
        if arr.size != ncomp:
            raise ValueError(f"{label} must have {ncomp} entries, got {arr.size}")

    fluid = Mixture(names=names, Tcrit=Tcrit, Pcrit=Pcrit, Vcrit=Vcrit, acentric=acf, molarMass=mw)
    if _has_field(props, "BIC"):
        bic = np.asarray(_field(props, "BIC"), dtype=float)
        if np.any(bic != 0.0):
            fluid.setBinaryInteraction(bic)

    eos_type = _first_eos_region(_field(props, "EOS", "PR"))
    model = EquationOfStateModel(None, fluid)
    if _has_field(props, "PRCORR") and (eos_type in (None, "") or str(eos_type).upper() == "PR"):
        model.setType("PRCORR")
    else:
        model.setType(str(eos_type))

    if _has_field(props, "OMEGAA"):
        model.omegaA = _omega_value(_field(props, "OMEGAA"), str(eos_type), "A")
    if _has_field(props, "OMEGAB"):
        model.omegaB = _omega_value(_field(props, "OMEGAB"), str(eos_type), "B")
    if _has_field(props, "SSHIFT"):
        model.PropertyModel.volumeShift = np.asarray(_field(props, "SSHIFT"), dtype=float).reshape(-1)
    model.reference_temperature = 273.15 + 15.0
    model.reference_pressure = ATM
    return model


def init_deck_eos_model(deck) -> "EquationOfStateModel":
    """Pythonic alias for :func:`initDeckEOSModel`."""
    return initDeckEOSModel(deck)


__all__ = ["initDeckEOSModel", "init_deck_eos_model"]
