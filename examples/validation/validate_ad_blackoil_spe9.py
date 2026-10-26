"""Validate ``simulateSPE9`` and ``blackoilTutorialSPE9`` against MRST."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.spe9 import setup_spe9_ad, simulate_spe9_ad

HERE = Path(__file__).resolve().parent


def validate():
    ref = loadmat(HERE/"ref_ad_blackoil_spe9.mat", squeeze_me=True)
    deck, model, schedule, state = setup_spe9_ad()
    assert model.G.cells.num == int(ref["nc"]) == 9000
    assert tuple(deck["cartDims"]) == tuple(np.asarray(ref["cartDims"], int))
    np.testing.assert_allclose(state["pressure"], ref["pressure"], rtol=2e-7)
    np.testing.assert_allclose(state["sW"], ref["sw"], rtol=2e-7)
    np.testing.assert_allclose(state["sG"], ref["sg"], atol=1e-14)
    np.testing.assert_allclose(state["rs"], ref["rs"], rtol=2e-7)
    np.testing.assert_allclose(model.rock.poro.ravel(), ref["poro"], rtol=2e-7)
    np.testing.assert_allclose(model.rock.perm, ref["perm"], rtol=2e-7)
    np.testing.assert_allclose(schedule.step_val, ref["step_val"], rtol=2e-14)
    assert len(schedule.controls) == int(ref["ncontrols"]) == 3
    assert all(len(control.W) == 26 for control in schedule.controls)
    producer = next(i for i, w in enumerate(schedule.controls[0].W) if w.name == "PROD2")
    values = [control.W[producer].val for control in schedule.controls]
    reference_values = [np.asarray(v).ravel()[producer] for v in np.atleast_1d(ref["well_val"])]
    np.testing.assert_allclose(values, reference_values, rtol=2e-12)
    result = simulate_spe9_ad(1, verbose=False)
    final = result.states[0]
    # Different nonlinear stopping rules and direct solvers yield a small
    # first-step spread; phase/state trends remain tightly aligned.
    np.testing.assert_allclose(final["pressure"], ref["state1_pressure"], rtol=1e-2)
    np.testing.assert_allclose(final["sW"], ref["state1_sw"], rtol=3e-3, atol=2e-5)
    np.testing.assert_allclose(final["sG"], ref["state1_sg"], rtol=3e-3, atol=2e-3)
    np.testing.assert_allclose(final["rs"], ref["state1_rs"], rtol=3e-3, atol=1e-7)
    assert result.report.converged
    assert "FOPR" in result.summary and "PROD2:WBHP" in result.summary
    print("SPE9 MRST/PyMRST setup regression passed")


if __name__ == "__main__":
    validate()
