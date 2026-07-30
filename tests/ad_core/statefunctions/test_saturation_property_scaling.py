"""Endpoint-scaling tests for MRST ``SaturationProperty``."""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.statefunctions.flowprops.virtual import (
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.flowprops.base_relative_permeability import (
    BaseRelativePermeability,
)
from pymrst.utils.adi import init_variables_adi, value


def _model():
    fluid_points = {
        "w": np.array([[0.10, 0.20, 0.90, 0.80],
                       [0.15, 0.25, 0.85, 0.70]]),
        "g": np.array([[0.00, 0.05, 0.70, 0.60],
                       [0.02, 0.08, 0.65, 0.55]]),
        "ow": np.array([[0.10, 0.20, 0.90, 0.90],
                        [0.15, 0.25, 0.85, 0.85]]),
        "og": np.array([[0.00, 0.10, 0.90, 0.88],
                        [0.02, 0.12, 0.83, 0.80]]),
    }
    drainage = {
        name: values.copy() for name, values in {
            "w": np.array([[np.nan, 0.25, 0.80, 0.60],
                           [0.12, np.nan, 0.90, np.nan]]),
            "g": np.array([[0.01, 0.06, 0.68, 0.50],
                           [np.nan, 0.09, 0.62, 0.45]]),
            "ow": np.array([[0.12, 0.24, 0.86, 0.75],
                            [0.18, 0.28, 0.80, 0.70]]),
            "og": np.array([[0.01, 0.11, 0.86, 0.72],
                            [0.03, 0.14, 0.78, 0.68]]),
        }.items()
    }
    return SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=2)),
        rock=SimpleNamespace(krscale={"drainage": drainage}),
        fluid=SimpleNamespace(),
        raw_fluid=SimpleNamespace(
            krPts=fluid_points,
            kr_w=lambda saturation: saturation**2,
        ),
        n_phases=3,
        get_phase_names=lambda: ["W", "O", "G"],
    )


def test_connate_water_uses_scaled_values_and_table_fallback():
    model = _model()
    prop = SaturationProperty(model)
    prop.regions = np.array([1, 2])
    assert prop.scaling_active
    assert np.allclose(prop.get_connate_water(model), [0.10, 0.12])


def test_two_point_scalers_and_stored_subset():
    model = _model()
    prop = SaturationProperty(model)
    prop.regions = np.array([1, 2])
    scaler = prop.get_scalers(model, "w")
    expected_m = (
        np.array([0.90, 0.85]) - np.array([0.20, 0.25])
    ) / (
        np.array([0.80, 0.90]) - np.array([0.25, 0.25])
    )
    assert np.allclose(scaler["m"][0], expected_m)
    assert np.allclose(scaler["k"][0], [0.80, 0.70])
    assert np.allclose(scaler["k"][1], [0.60, 0.70])

    prop.store_scalers(model)
    subset = prop.get_scalers(model, "w", cells=np.array([1]))
    assert subset["m"][0].shape == (1,)
    assert np.allclose(subset["m"][0], scaler["m"][0][1])


def test_three_point_scalers_and_critical_saturations():
    model = _model()
    prop = SaturationProperty(model)
    prop.regions = np.array([1, 1])
    prop.relperm_points = 3
    scaler = prop.get_scalers(model, "w")
    assert scaler["points"] == 3
    assert len(scaler["m"]) == 2
    assert len(scaler["p"]) == 3
    assert all(np.isfinite(item).all() for item in scaler["m"])

    critical = prop.get_critical_phase_saturations(model)
    assert np.allclose(critical, [0.20, max(0.20, 0.10), 0.05])


def test_base_relative_permeability_applies_scaling_with_ad():
    model = _model()
    prop = BaseRelativePermeability(model)
    prop.regions = np.array([1, 2])
    # Re-store now that the synthetic regional map has been assigned.
    prop.scalers = {}
    prop.store_scalers(model)
    (saturation,) = init_variables_adi(np.array([0.50, 0.60]))
    scaled, ratio = prop.scale_saturation(model, saturation, "w")
    result = prop.evaluate_phase_relative_permeability(
        model, "w", saturation
    )
    assert np.allclose(value(result), ratio * value(scaled) ** 2)
    assert np.isfinite(result.jac[0].toarray()).all()
