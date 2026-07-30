"""Compare the runnable blackoilSectorModelExample with MRST 2024a."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.blackoil_sector_model_example import (
    build_blackoil_sector_model_example,
    run_blackoil_sector_model_example,
)
from pymrst.utils.units import DAY


HERE = Path(__file__).resolve().parent


def validate() -> None:
    ref = loadmat(HERE / "ref_ad_blackoil_sector_model.mat", squeeze_me=True)
    kwargs = dict(
        dims=(4, 4, 1), physical_dims=(80.0, 80.0, 8.0),
        total_time=2.0 * DAY, max_timestep=DAY, rampup_steps=1,
    )
    case = build_blackoil_sector_model_example(**kwargs)
    result = run_blackoil_sector_model_example(**kwargs)

    np.testing.assert_allclose(case.timesteps, np.ravel(ref["dt"]), rtol=0, atol=0)
    assert case.producer_cell == int(ref["producerCell"])
    np.testing.assert_allclose(
        case.open_model.facility.wells[0].WI, np.ravel(ref["WI"]),
        rtol=2e-14, atol=0,
    )

    comparisons = (
        (result.open_boundary, "pressure", "pOpen", 6e-7, 0.0),
        (result.closed_boundary, "pressure", "pClosed", 6e-7, 0.0),
        (result.open_boundary, "sW", "swOpen", 2e-5, 2e-9),
        (result.closed_boundary, "sW", "swClosed", 2e-5, 2e-9),
        (result.open_boundary, "sG", "sgOpen", 0.0, 0.0),
        (result.closed_boundary, "sG", "sgClosed", 0.0, 0.0),
        (result.open_boundary, "rs", "rsOpen", 2e-14, 0.0),
        (result.closed_boundary, "rs", "rsClosed", 2e-14, 0.0),
    )
    for simulation, py_key, matlab_key, rtol, atol in comparisons:
        np.testing.assert_allclose(
            simulation.history[py_key], np.atleast_2d(ref[matlab_key]),
            rtol=rtol, atol=atol,
        )

    for simulation, q_key, p_key in (
        (result.open_boundary, "qOsOpen", "bhpOpen"),
        (result.closed_boundary, "qOsClosed", "bhpClosed"),
    ):
        np.testing.assert_allclose(
            np.ravel(simulation.history["qOs"]), np.ravel(ref[q_key]),
            rtol=2e-14, atol=0,
        )
        # Reservoir states agree substantially more closely than BHP because
        # the two facility assemblers use slightly different well-equation
        # scaling and stopping tolerances.
        np.testing.assert_allclose(
            np.ravel(simulation.history["bhp"]), np.ravel(ref[p_key]),
            rtol=4e-3, atol=0,
        )

    assert result.open_boundary.history["pressure"][-1].mean() > (
        result.closed_boundary.history["pressure"][-1].mean()
    )
    print("blackoilSectorModelExample matches MRST reference")


if __name__ == "__main__":
    validate()
