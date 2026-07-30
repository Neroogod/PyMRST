from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.models.facilities import (
    nozzle_valve,
    pack_perforation_properties,
    unpack_perforation_properties,
)


def test_nozzle_valve_flow_conversions_and_sign():
    rho = np.array([800.0, 900.0])
    diameter = np.array([0.1, 0.2])
    coeff = np.array([0.8, 0.7])
    velocity = np.array([2.0, -3.0])
    area = np.pi * (diameter / 2.0) ** 2
    expected = -(np.sign(velocity) * rho * velocity**2) / (2 * coeff**2)
    assert np.allclose(
        nozzle_valve(velocity, rho, diameter, coeff, "velocity"), expected
    )
    assert np.allclose(
        nozzle_valve(velocity * area, rho, diameter, coeff, "volumeRate"),
        expected,
    )
    assert np.allclose(
        nozzle_valve(velocity * rho * area, rho, diameter, coeff), expected
    )
    with pytest.raises(ValueError, match="unknown flow type"):
        nozzle_valve(velocity, rho, diameter, coeff, "molarRate")


def test_pack_and_unpack_perforation_properties_recursively():
    well = SimpleNamespace(cells=np.array([1, 3]))
    packed = pack_perforation_properties(
        well,
        pressure=np.array([100.0, 110.0, 120.0, 130.0]),
        mobility=[
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 7.0, 8.0]),
        ],
        density=[np.array([700.0, 710.0, 720.0, 730.0])],
        dissolved=[[np.array([0.1, 0.2, 0.3, 0.4])], None],
        components=[np.array([0.9, 0.8, 0.7, 0.6])],
        wellvars=[np.array([10.0, 20.0]), np.array([30.0, 40.0, 50.0])],
        wellvarnames=["a", "b"],
        varmaps=[np.array([0, 1]), np.array([1, 0, 1])],
        wellmap=np.array([[1, 2], [2, 0]]),
        well_index=0,
    )
    assert np.array_equal(packed.pressure, [110.0, 130.0])
    assert np.array_equal(packed.mob[0], [2.0, 4.0])
    assert np.array_equal(packed.dissolved[0][0], [0.2, 0.4])
    assert packed.dissolved[1] is None
    assert np.array_equal(packed.extravars[0], [10.0])
    assert np.array_equal(packed.extravars[1], [40.0])
    unpacked = unpack_perforation_properties(packed)
    assert unpacked[0] is packed.pressure
    assert unpacked[5] is packed.extravars
