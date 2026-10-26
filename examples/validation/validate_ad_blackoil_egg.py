"""Validate ``eggExample`` setup and production against MATLAB MRST.

`ref_ad_blackoil_egg.mat` is MRST's own output: the grid, rock, equilibrated
state, the well set, and the state after the first report step.

The tolerances split into two groups, and the split is the finding:

* the **setup** quantities agree to machine precision. They are the same
  numbers read a different way - the deck's own - and nothing between the file
  and the model iterates. Anything looser here would hide a real difference;
* the **post-step** quantities agree to the tolerance two different nonlinear
  solvers reach on the same problem, which is measured rather than assumed.

`setup_egg` lives in `pymrst.modules.test_suite.dataset_functions.egg`, where
MRST keeps `setupEGG.m` - which is why `eggExample.m` opens with
`mrstModule add ... test-suite`.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.egg import run_egg_example
from pymrst.modules.test_suite.dataset_functions.egg import setup_egg

HERE = Path(__file__).resolve().parent


def validate():
    ref = loadmat(HERE/"ref_ad_blackoil_egg.mat", squeeze_me=True)

    # -- setup: the deck's own numbers, to machine precision ---------------
    G, rock, _fluid, _deck, _state = setup_egg()
    assert G.cells.num == int(ref["nc"]) == 18553
    np.testing.assert_array_equal(np.asarray(G.cells.indexMap) + 1,
                                  np.asarray(ref["indexMap"]))
    np.testing.assert_allclose(G.cells.centroids,
                               np.asarray(ref["centroids"], dtype=float),
                               atol=2e-12)
    np.testing.assert_allclose(np.asarray(rock.poro).ravel(), ref["poro"],
                               rtol=1e-14)
    np.testing.assert_allclose(rock.perm, ref["perm"], rtol=2e-12)

    result = run_egg_example(1, verbose=False)
    np.testing.assert_allclose(result.state0["pressure"], ref["pressure0"],
                               rtol=1e-12)
    np.testing.assert_allclose(result.state0["sW"], ref["sw0"], rtol=1e-14)
    # The run above is truncated to one report step; the schedule to compare
    # is the deck's own, all 123 of them.
    from pymrst.modules.test_suite.dataset_functions.egg import get_deck_egg
    np.testing.assert_allclose(
        np.asarray(get_deck_egg()["SCHEDULE"].step["val"], dtype=float),
        np.asarray(ref["step_val"], dtype=float), rtol=1e-14)

    wells = list(result.schedule.controls[0].W)
    assert [w.name for w in wells] == list(ref["well_names"])
    assert [int(w.sign) for w in wells] == list(
        np.asarray(ref["well_sign"], dtype=int))
    np.testing.assert_allclose([w.val for w in wells][:8],
                               np.asarray(ref["well_val"], float)[:8],
                               rtol=1e-12)

    # -- after one report step: two solvers on one problem -----------------
    first = result.states[0]
    np.testing.assert_allclose(first["pressure"], ref["pressure1"], rtol=1e-3)
    np.testing.assert_allclose(first["sW"], ref["sw1"], rtol=3.5e-2, atol=3e-4)
    np.testing.assert_allclose(first["qWs"], ref["qWs1"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(first["qOs"], ref["qOs1"], rtol=1.2e-2,
                               atol=1e-7)
    np.testing.assert_allclose(first["bhp"], ref["bhp1"], rtol=1e-3)

    print("EGG MRST/PyMRST setup and production regression passed")


if __name__ == "__main__":
    validate()
