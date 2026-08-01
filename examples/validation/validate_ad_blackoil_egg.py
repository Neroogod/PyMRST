"""Validate ``eggExample`` setup and production against MATLAB MRST."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.egg import (
    compare_egg_production_history, setup_egg, run_egg_example,
)

HERE = Path(__file__).resolve().parent


def validate():
    ref = loadmat(HERE/"ref_ad_blackoil_egg.mat", squeeze_me=True)
    _, model, schedule, state0 = setup_egg()
    assert model.G.cells.num == int(ref["nc"]) == 18553
    np.testing.assert_allclose(model.G.cells.centroids, ref["centroids"], atol=2e-12)
    np.testing.assert_allclose(model.rock.poro.ravel(), ref["poro"], rtol=1e-14)
    np.testing.assert_allclose(model.rock.perm, ref["perm"], rtol=2e-12)
    np.testing.assert_allclose(state0["pressure"], ref["pressure0"], rtol=2e-6)
    np.testing.assert_allclose(schedule.step_val, ref["step_val"], rtol=1e-14)
    result = run_egg_example(1, verbose=False)
    final = result.states[0]
    np.testing.assert_allclose(final["pressure"], ref["pressure1"], rtol=8e-3)
    np.testing.assert_allclose(final["sW"], ref["sw1"], rtol=3.5e-2, atol=3e-4)
    np.testing.assert_allclose(final["qWs"], ref["qWs1"], rtol=2e-2, atol=1e-7)
    np.testing.assert_allclose(final["qOs"], ref["qOs1"], rtol=2e-2, atol=1e-7)
    errors = compare_egg_production_history(result, HERE/"ref_ad_blackoil_egg.mat")
    assert errors["qWs"] == 0.0
    assert errors["qOs"] < 0.012
    assert errors["bhp"] < 7e-4
    print("EGG MRST/PyMRST setup and production regression passed")


if __name__ == "__main__":
    validate()
