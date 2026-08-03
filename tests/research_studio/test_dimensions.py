"""Opt-in dimensional checking (M2.12).

The central property is not that wrong dimensions are caught - it is that
nothing changes for a caller who did not ask. This package can break the
additive covenant while looking additive, because a check that refuses input
PyMRST accepts today has changed the port even though it added no feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.research_studio.units import (
    AREA, COMPRESSIBILITY, DENSITY, Dimension, DimensionError, PERMEABILITY,
    PRESSURE, Quantity, RATE, VISCOSITY, VOLUME, declared_dimensions, expects,
    magnitude,
)
from pymrst.utils.units import BARSA, DARCY, MILLI


@expects(pressure=PRESSURE, compressibility=COMPRESSIBILITY)
def _fluid(pressure, compressibility, note="unchecked"):
    return {"pressure": pressure, "compressibility": compressibility,
            "note": note}


def test_plain_numbers_pass_through_exactly_as_before():
    """The covenant, and the reason this is opt-in.

    A caller who supplies floats - which is every caller today - must reach
    the function body with the identical objects and no exception. If this
    ever fails, the feature has changed PyMRST rather than extended it.
    """
    result = _fluid(300*BARSA, 1e-10)
    assert result["pressure"] == 300*BARSA
    assert result["compressibility"] == 1e-10
    assert result["note"] == "unchecked"


def test_an_undeclared_argument_is_never_touched():
    result = _fluid(1.0, 2.0, note=object())
    assert isinstance(result["note"], object)


def test_a_wrongly_dimensioned_quantity_is_refused_naming_both_sides():
    """The error has to say what was expected *and* what arrived.

    "Invalid units" sends someone to read the source. Naming both makes the
    mistake obvious at the call site.
    """
    with pytest.raises(DimensionError) as raised:
        _fluid(Quantity(300*BARSA, PRESSURE),
               Quantity(100*MILLI*DARCY, PERMEABILITY))
    message = str(raised.value)
    assert "compressibility" in message
    assert "expects" in message and "received" in message
    assert "L^1 M^-1 T^2" in message and "L^2" in message


def test_swapped_arguments_are_caught():
    """The mistake this exists for: both are floats, so nothing else sees it."""
    with pytest.raises(DimensionError, match="pressure"):
        _fluid(Quantity(1e-10, COMPRESSIBILITY), Quantity(300*BARSA, PRESSURE))


def test_a_correctly_dimensioned_quantity_is_unwrapped_before_the_body():
    """The function body must never handle a Quantity.

    Otherwise every function that opts into checking also has to learn a new
    type, and the feature stops being free.
    """
    result = _fluid(Quantity(300*BARSA, PRESSURE), Quantity(1e-10, COMPRESSIBILITY))
    assert type(result["pressure"]) is float
    assert result["pressure"] == 300*BARSA
    assert result["compressibility"] == 1e-10


def test_checked_and_unchecked_calls_produce_identical_values():
    """Bit-identical, on the same evidence standard as the observation hook."""
    plain = _fluid(300*BARSA, 1e-10)
    wrapped = _fluid(Quantity(300*BARSA, PRESSURE),
                     Quantity(1e-10, COMPRESSIBILITY))
    assert plain == wrapped


def test_arrays_survive_the_wrapper_unchanged():
    @expects(permeability=PERMEABILITY)
    def take(permeability):
        return permeability

    values = np.array([1e-13, 2e-13, 3e-13])
    np.testing.assert_array_equal(take(values), values)
    np.testing.assert_array_equal(take(Quantity(values, PERMEABILITY)), values)
    assert type(take(Quantity(values, PERMEABILITY))) is np.ndarray


def test_dimensions_compose_by_multiplication_and_division():
    assert (PRESSURE*COMPRESSIBILITY).matches(Dimension())
    assert (VOLUME/Dimension(time=1)).matches(RATE)
    assert not PRESSURE.matches(VISCOSITY)


def test_permeability_and_area_are_not_separated_because_they_cannot_be():
    """Both are L^2. A check that claimed to tell them apart would be lying.

    Stated as a test so the limit is recorded rather than discovered by
    someone who expected more.
    """
    assert PERMEABILITY.matches(AREA)

    @expects(k=PERMEABILITY)
    def take(k):
        return k

    assert take(Quantity(4.0, AREA)) == 4.0


def test_declaring_a_dimension_for_a_parameter_that_does_not_exist_is_refused():
    """A declared dimension naming nothing checks nothing, silently."""
    with pytest.raises(ValueError, match="no parameter"):
        @expects(nonexistent=PRESSURE)
        def take(pressure):
            return pressure


def test_declared_dimensions_are_introspectable_for_coverage():
    """Coverage is measured, not assumed.

    The gap between entry points that check and entry points that could stays
    visible only if it can be counted.
    """
    assert declared_dimensions(_fluid) == {"pressure": PRESSURE,
                                           "compressibility": COMPRESSIBILITY}

    def undecorated(x):
        return x

    assert declared_dimensions(undecorated) == {}


def test_magnitude_unwraps_either_form():
    assert magnitude(3.0) == 3.0
    assert magnitude(Quantity(3.0, PRESSURE)) == 3.0


def test_keyword_and_positional_calls_are_checked_alike():
    with pytest.raises(DimensionError):
        _fluid(pressure=Quantity(1e-10, COMPRESSIBILITY), compressibility=1e-10)
    with pytest.raises(DimensionError):
        _fluid(Quantity(1e-10, COMPRESSIBILITY), 1e-10)


def test_a_partially_applied_call_does_not_break_the_wrapper():
    @expects(a=PRESSURE, b=VISCOSITY)
    def take(a, b=None):
        return a, b

    assert take(1.0) == (1.0, None)
    assert take(Quantity(1.0, PRESSURE)) == (1.0, None)


def test_nothing_in_the_replica_layer_imports_this():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]/"pymrst"
    offenders = [
        path for path in (root/"modules").rglob("*.py")
        if "research_studio" in path.read_text(errors="ignore")
    ]
    assert not offenders, f"replica modules importing the extension: {offenders}"


def test_a_wrapped_value_on_an_undeclared_argument_is_still_unwrapped():
    """Unwrapping covers every argument, not only the declared ones.

    A caller may wrap something the function never declared - reasonably, if
    they are being careful - and the body must still never handle a Quantity.
    Doing this per-check instead left the unwrap redundant with the general
    loop, which a mutation exposed by surviving.
    """
    @expects(declared=PRESSURE)
    def take(declared, undeclared):
        return type(declared), type(undeclared)

    assert take(Quantity(1.0, PRESSURE), Quantity(2.0, VISCOSITY)) == (float, float)
