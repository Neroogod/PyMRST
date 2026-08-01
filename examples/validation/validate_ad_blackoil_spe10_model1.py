"""Validate PyMRST ``tenthCSP_Model_I`` setup and first report against MRST."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.spe10 import (
    run_tenth_csp_model_i, setup_tenth_csp_model_i,
)

HERE = Path(__file__).resolve().parent


def validate():
    ref = loadmat(HERE/"ref_ad_blackoil_spe10_model1.mat", squeeze_me=True)
    dataset, model, schedule, state0 = setup_tenth_csp_model_i()
    assert model.G.cells.num == 2000
    assert schedule.n_steps == 131
    np.testing.assert_allclose(model.G.cells.centroids, ref["centroids"], atol=3e-13)
    np.testing.assert_array_equal(model.G.cells.indexMap, np.arange(2000))
    np.testing.assert_allclose(model.rock.perm, ref["perm"], rtol=0, atol=0)
    np.testing.assert_allclose(model.rock.poro.ravel(), ref["poro"], rtol=0, atol=0)
    np.testing.assert_allclose(dataset["sgof"], ref["sgof"], rtol=0, atol=0)
    np.testing.assert_allclose(state0["pressure"], ref["pressure0"], rtol=2e-15)
    np.testing.assert_allclose(model.wells[0].WI, ref["WI_I"], rtol=4e-15)
    np.testing.assert_allclose(model.wells[1].WI, ref["WI_P"], rtol=4e-15)

    result = run_tenth_csp_model_i(1, verbose=False)
    state = result.states[0]
    np.testing.assert_allclose(state["pressure"], ref["pressure1"], rtol=2.3e-3)
    np.testing.assert_allclose(state["sW"], ref["sO1"], rtol=0, atol=1.7e-2)
    np.testing.assert_allclose(state["bhp"], ref["bhp1"], rtol=1.7e-3)
    # Aggregate rate control is invariant even where PyMRST and MRST allocate
    # a marginal crossflow completion differently between oil and gas.
    np.testing.assert_allclose(
        state["qWs"][0]+state["qOs"][0],
        ref["qOs1"][0]+ref["qGs1"][0], rtol=2e-15,
    )
    np.testing.assert_allclose(state["qWs"][1], ref["qOs1"][1], rtol=1e-5)
    print("SPE10 Model I MRST/PyMRST setup and first-report regression passed")


if __name__ == "__main__":
    validate()
