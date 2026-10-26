"""Tests for MRST ``wellBoreFriction`` and control-management helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.models.facilities.well_bore_friction import (
    well_bore_friction_ad,
)
from pymrst.modules.ad_core.models.wells.update_switched_controls import (
    update_switched_controls,
)
from pymrst.utils.adi import init_variables_adi, value


def _well(name, control, sign, limits):
    return SimpleNamespace(
        name=name, type=control, val=0.0, sign=sign, lims=limits, status=True
    )


def test_update_switched_well_controls_handles_injector_and_producer():
    injector = _well("I", "rate", 1, {"bhp": 300e5})
    producer = _well("P", "bhp", -1, {"orat": -1e-3})
    ok, switches = update_switched_controls(
        [injector, producer],
        np.array([320e5, 150e5]),
        np.array([[3e-3, 0.0], [0.0, -2e-3]]),
    )
    assert not ok
    assert switches == [(0, "rate", "bhp"), (1, "bhp", "orat")]
    assert (injector.type, injector.val) == ("bhp", 300e5)
    assert (producer.type, producer.val) == ("orat", -1e-3)


def test_well_bore_friction_sign_flow_types_and_ad_derivative():
    common = dict(
        rho=np.full(3, 1000.0),
        mu=np.full(3, 1e-3),
        diameter=np.full(3, 0.2),
        length=np.full(3, 10.0),
        roughness=np.full(3, 1e-5),
    )
    mass_rate = np.array([0.0, 2.0, -2.0])
    pressure_drop = well_bore_friction_ad(mass_rate, **common)
    assert pressure_drop[0] == 0.0
    assert pressure_drop[1] < 0.0 < pressure_drop[2]
    assert np.isclose(pressure_drop[1], -pressure_drop[2])

    area = np.pi * (0.2 / 2.0) ** 2
    velocity = mass_rate / (common["rho"] * area)
    from_velocity = well_bore_friction_ad(
        velocity, **common, flowtype="velocity"
    )
    assert np.allclose(from_velocity, pressure_drop)

    (rate_ad,) = init_variables_adi(np.array([2.0]))
    ad_drop = well_bore_friction_ad(
        rate_ad,
        rho=np.array([1000.0]),
        mu=np.array([1e-3]),
        diameter=np.array([0.2]),
        length=np.array([10.0]),
        roughness=np.array([1e-5]),
    )
    assert value(ad_drop)[0] < 0.0
    assert np.isfinite(ad_drop.jac[0].toarray()).all()
    assert ad_drop.jac[0][0, 0] < 0.0


def test_well_bore_friction_rejects_unknown_flow_type():
    with pytest.raises(ValueError, match="unknown flow type"):
        well_bore_friction_ad(
            np.array([1.0]), np.array([1000.0]), np.array([1e-3]),
            np.array([0.2]), np.array([10.0]), np.array([1e-5]),
            flowtype="unknown",
        )
