from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils import convert_multi_segment_wells_deck


def _case():
    grid = compute_geometry(cart_grid([1, 1, 3], [1.0, 1.0, 30.0]))
    well = SimpleNamespace(
        name="MS1",
        refDepth=0.0,
        cells=np.array([0, 1, 2]),
        WI=np.array([11.0, 22.0, 33.0]),
        dZ=np.array([5.0, 15.0, 25.0]),
        r=np.array([0.1, 0.2, 0.3]),
        cstatus=np.ones(3, dtype=bool),
    )
    compsegs = [
        [1, 1, 1, 1, 9.0, 11.0],
        [1, 1, 2, 1, 19.0, 21.0],
        [1, 1, 3, 1, 29.0, 31.0],
    ]
    welsegs = SimpleNamespace(
        header=[0.0, 0.0, 2.0, "INC"],
        segments=[[2, 4, 1, 1, 10.0, 10.0, 0.2, 1e-5, 3.0]],
    )
    control = {
        "COMPSEGS": [["MS1", compsegs]],
        "WELSEGS": [["MS1", welsegs]],
    }
    return grid, well, control


def test_compact_deck_conversion():
    grid, well, control = _case()
    output = convert_multi_segment_wells_deck([well], grid, control)[0]
    assert output.isMS
    assert np.array_equal(output.cells, [0, 1, 2])
    assert np.array_equal(output.perf_map, [0, 1, 2])
    assert np.array_equal(output.cells_to_nodes, [[0, 1], [1, 2], [2, 3]])
    assert np.array_equal(output.topo, [[0, 1], [1, 2], [2, 3]])
    assert np.allclose(output.nodes.depth, [0.0, 10.0, 20.0, 30.0])
    assert np.allclose(output.nodes.vol, [2.0, 3.0, 3.0, 3.0])
    assert np.allclose(output.segments.length, 10.0)
    assert np.allclose(output.segments.diameter, 0.2)


def test_noncompact_preserves_branch_data():
    grid, well, control = _case()
    output = convert_multi_segment_wells_deck(
        [well], grid, control, compact=False
    )[0]
    assert len(output.branches) == 1
    branch = output.branches[0]
    assert np.array_equal(branch.index, [2, 3, 4])
    assert np.allclose(branch.tube_depths, [10.0, 20.0, 30.0])
    assert np.allclose(branch.z_depths, [10.0, 20.0, 30.0])
