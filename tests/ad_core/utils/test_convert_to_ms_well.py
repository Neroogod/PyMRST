from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import convert_to_ms_well
from pymrst.params.wells_and_bc import Well


def _well():
    return Well(
        cells=np.array([0, 1, 2]),
        type="rate",
        val=1.0,
        r=np.array([0.1, 0.2, 0.3]),
        dir="z",
        WI=np.ones(3),
        dZ=np.array([0.0, 5.0, 10.0]),
        name="MS",
        compi=np.array([1.0, 0.0]),
        refDepth=100.0,
    )


def test_convert_to_ms_well_default_geometry():
    grid = SimpleNamespace(cells=SimpleNamespace(
        centroids=np.array([[0.0, 0, 100], [3.0, 0, 104], [3.0, 4, 110]])
    ))
    converted = convert_to_ms_well(_well(), grid=grid)
    assert converted.isMS
    assert np.array_equal(converted.cell2node.toarray(), np.eye(3))
    assert np.array_equal(converted.nodes.depth, [100.0, 105.0, 110.0])
    assert np.allclose(converted.segments.length, [5.0, np.sqrt(52.0)])
    assert np.allclose(converted.segments.diam, [0.3, 0.5])
    assert np.array_equal(converted.connDZ, [0.0, 0.0, 0.0])


def test_convert_to_ms_well_grouped_nodes_and_explicit_segments():
    converted = convert_to_ms_well(
        _well(),
        cell2node=np.array([0, 0, 1]),
        topology=np.array([[0, 1]]),
        segment_length=[12.0],
        segment_diameter=[0.4],
        segment_roughness=[1e-5],
        segment_flow_model=[2],
        conn_dz=[1, 2, 3],
        volume=[7, 8],
    )
    assert np.allclose(converted.nodes.depth, [102.5, 110.0])
    assert np.array_equal(converted.nodes.vol, [7, 8])
    assert np.array_equal(converted.segments.length, [12.0])
    assert np.array_equal(converted.segments.diam, [0.4])
    assert np.array_equal(converted.connDZ, [1, 2, 3])
