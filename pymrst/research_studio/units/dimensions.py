"""Refuse quantities whose dimensions do not match what an argument means.

`pymrst.utils.units` gives a user `300*BARSA`, and that is a plain float. So is
`100*MILLI*DARCY`. Once either reaches a function they are indistinguishable,
and a compressibility argument receiving a permeability produces a plausible
number rather than an error.

**Opt-in, and inert by default.** This is the one capability in the plan that
can break the covenant while looking additive: a check that refuses input
PyMRST accepts today has changed the behaviour of the port, even though it
added no feature. So the rule here is that *the user supplies the dimension*.
A plain float arrives and passes through untouched, exactly as it does now. A
`Quantity` arrives and is verified. Nothing is checked that the caller did not
ask to have checked.

Dimensions are exponents over (length, mass, time), which is enough for every
quantity a reservoir model takes and small enough to compare exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable
import inspect

import numpy as np


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponents of length, mass and time."""

    length: int = 0
    mass: int = 0
    time: int = 0
    name: str = ""

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.length + other.length, self.mass + other.mass,
                         self.time + other.time)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.length - other.length, self.mass - other.mass,
                         self.time - other.time)

    def matches(self, other: "Dimension") -> bool:
        """Exponents only. Two quantities with the same exponents are the same
        kind of thing, whatever either happens to be called."""
        return (self.length, self.mass, self.time) == \
               (other.length, other.mass, other.time)

    def __str__(self) -> str:
        if self.name:
            return self.name
        parts = []
        for symbol, power in (("L", self.length), ("M", self.mass),
                              ("T", self.time)):
            if power:
                parts.append(symbol if power == 1 else f"{symbol}^{power}")
        return "·".join(parts) or "dimensionless"


DIMENSIONLESS = Dimension(name="dimensionless")
LENGTH = Dimension(length=1, name="length")
MASS = Dimension(mass=1, name="mass")
TIME = Dimension(time=1, name="time")
AREA = Dimension(length=2, name="area")
VOLUME = Dimension(length=3, name="volume")
PRESSURE = Dimension(length=-1, mass=1, time=-2, name="pressure")
PERMEABILITY = Dimension(length=2, name="permeability")
VISCOSITY = Dimension(length=-1, mass=1, time=-1, name="viscosity")
DENSITY = Dimension(length=-3, mass=1, name="density")
COMPRESSIBILITY = Dimension(length=1, mass=-1, time=2, name="compressibility")
RATE = Dimension(length=3, time=-1, name="rate")

# Permeability and area share exponents, which is physically true - both are
# L^2 - so a check cannot separate them and must not pretend to.
_ALIASES = {"permeability": AREA, "area": AREA}


class DimensionError(TypeError):
    """A quantity arrived with the wrong dimensions."""


@dataclass(frozen=True, slots=True)
class Quantity:
    """A value that knows what kind of thing it is.

    Constructed by a caller who wants checking. Never produced by PyMRST
    itself, so nothing downstream has to handle it.
    """

    value: Any
    dimension: Dimension

    def __float__(self) -> float:
        return float(self.value)

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.value, dtype=dtype)
        return array.copy() if copy else array

    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, {self.dimension})"


def magnitude(value: Any) -> Any:
    """The plain value, whether or not it was wrapped.

    The single place the wrapper is removed, so a checked call and an
    unchecked one reach the numerics with the identical object.
    """
    return value.value if isinstance(value, Quantity) else value


def expects(**dimensions: Dimension) -> Callable:
    """Declare the dimensions of named arguments.

    Wrapped functions behave exactly as before for callers passing plain
    numbers - the decorator finds no `Quantity` and does nothing but forward.
    A caller who passes one gets it verified and unwrapped, so the function
    body never sees a `Quantity` either way.
    """
    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)
        unknown = set(dimensions) - set(signature.parameters)
        if unknown:
            raise ValueError(
                f"{function.__name__} has no parameter(s) {sorted(unknown)}; "
                f"a declared dimension that names nothing checks nothing"
            )

        @wraps(function)
        def checked(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            for name, expected in dimensions.items():
                if name not in bound.arguments:
                    continue
                given = bound.arguments[name]
                if not isinstance(given, Quantity):
                    continue                       # unchecked, as before
                expected = _ALIASES.get(expected.name, expected)
                actual = _ALIASES.get(given.dimension.name, given.dimension)
                if not actual.matches(expected):
                    raise DimensionError(
                        f"{function.__name__}({name}=...) expects "
                        f"{expected} [{_exponents(expected)}], received "
                        f"{given.dimension} [{_exponents(given.dimension)}]"
                    )
            # Every argument, not only the declared ones: a caller may wrap a
            # value the function never declared, and the body must still never
            # see a Quantity. Doing it here rather than per-check also means
            # there is exactly one place the wrapper comes off.
            for name, value in list(bound.arguments.items()):
                bound.arguments[name] = _unwrap(value)
            return function(*bound.args, **bound.kwargs)

        checked.__pymrst_dimensions__ = dict(dimensions)
        return checked
    return decorate


def _unwrap(value: Any) -> Any:
    if isinstance(value, Quantity):
        return value.value
    return value


def _exponents(dimension: Dimension) -> str:
    return f"L^{dimension.length} M^{dimension.mass} T^{dimension.time}"


def declared_dimensions(function: Callable) -> dict:
    """What `function` declares, or an empty mapping.

    Coverage is measured with this rather than assumed, so the gap between
    "entry points that check" and "entry points that could" stays visible.
    """
    return dict(getattr(function, "__pymrst_dimensions__", {}))


__all__ = [
    "AREA", "COMPRESSIBILITY", "DENSITY", "DIMENSIONLESS", "Dimension",
    "DimensionError", "LENGTH", "MASS", "PERMEABILITY", "PRESSURE",
    "Quantity", "RATE", "TIME", "VISCOSITY", "VOLUME", "declared_dimensions",
    "expects", "magnitude",
]
