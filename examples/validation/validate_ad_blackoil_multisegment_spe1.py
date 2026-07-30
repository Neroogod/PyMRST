"""Validate the full multisegmentWellExample against MRST 2024a."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.multisegment_well_example import (
    build_multisegment_well_example,
    run_multisegment_well_example,
)


REF = Path(__file__).with_name("ref_ad_blackoil_multisegment_spe1.mat")


def validate() -> None:
    ref = loadmat(REF, squeeze_me=True)
    case = build_multisegment_well_example(1)
    result = run_multisegment_well_example(1)
    well = case.multisegment_model.wells[1]
    ms = well.multisegment

    topology = np.array([[s.parent + 1, s.index + 1] for s in ms.segments[1:]])
    np.testing.assert_array_equal(topology, ref["topo"])
    np.testing.assert_array_equal(case.completion_cells, ref["completionCells"])
    np.testing.assert_array_equal(ms.perforation_segment, ref["perforationNodes"])
    np.testing.assert_allclose(well.WI, ref["WI"], rtol=4e-15, atol=0)
    assert case.timesteps[0] == float(ref["dt"])

    np.testing.assert_allclose(
        result.node_pressure[-1], ref["nodePressure"], rtol=1.8e-3, atol=0,
    )
    np.testing.assert_allclose(
        result.multisegment.history["ms_vmix"][-1], ref["segmentFlux"],
        rtol=3.5e-4, atol=0,
    )
    np.testing.assert_allclose(
        result.multisegment.history["pressure"][-1], ref["reservoirPressure"],
        rtol=5.6e-6, atol=0,
    )
    np.testing.assert_allclose(
        result.simple.history["pressure"][-1], ref["simplePressure"],
        rtol=7e-9, atol=0,
    )
    for simulation, key in ((result.simple, "qSimple"),
                            (result.multisegment, "qMS")):
        rates = np.array([
            simulation.history[name][-1, 1]
            for name in ("qWs", "qOs", "qGs")
        ])
        np.testing.assert_allclose(rates, ref[key], rtol=3e-3, atol=7e-10)
    print("full SPE1 multisegmentWellExample matches MRST reference")


if __name__ == "__main__":
    validate()
