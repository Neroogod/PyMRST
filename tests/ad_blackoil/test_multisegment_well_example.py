import numpy as np

from pymrst.modules.ad_blackoil.examples.multisegment_well_example import (
    build_multisegment_well_example,
    run_multisegment_well_example,
)


def test_multisegmentWellExample_has_full_spe1_topology_and_controls():
    case = build_multisegment_well_example(1)
    well = case.multisegment_model.wells[1]
    ms = well.multisegment
    assert len(ms.segments) == 13
    assert np.array_equal(ms.perforation_segment, np.arange(7, 13))
    assert [s.flow_model for s in ms.segments[1:7]] == ["tubing"] * 6
    assert [s.flow_model for s in ms.segments[7:]] == ["valve"] * 6
    assert all(s.valve_count == 30 for s in ms.segments[7:])
    assert case.multisegment_model.wells[0].type == "rate"
    assert well.type == "rate" and well.sign == -1


def test_multisegmentWellExample_runs_simple_and_full_ms_models():
    result = run_multisegment_well_example(1)
    assert result.simple.report.converged
    assert result.multisegment.report.converged
    assert result.node_pressure.shape == (1, 13)
    assert result.multisegment.history["ms_vmix"].shape == (1, 12)
    # A producing well loses pressure toward the wellhead.
    assert result.node_pressure[0, 0] < result.node_pressure[0, -1]
