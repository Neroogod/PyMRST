"""Phase 432: compare convertMultiSegmentWellsDeck with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils import convert_multi_segment_wells_deck

REF = Path(__file__).with_name("ref_p432_ms_well_deck.mat")


def _convert():
    grid = compute_geometry(cart_grid([1, 1, 3], [1.0, 1.0, 30.0]))
    well = SimpleNamespace(
        name="MS1", refDepth=0.0, cells=np.arange(3),
        WI=np.array([11.0, 22.0, 33.0]), dZ=np.array([5.0, 15.0, 25.0]),
        r=np.array([0.1, 0.2, 0.3]), cstatus=np.ones(3, dtype=bool),
    )
    comp = [
        [1, 1, 1, 1, 9, 11],
        [1, 1, 2, 1, 19, 21],
        [1, 1, 3, 1, 29, 31],
    ]
    segments = SimpleNamespace(
        header=[0, 0, 2, "INC"],
        segments=[
            [2, 2, 1, 1, 10, 10, 0.2, 1e-5, 3],
            [3, 3, 1, 2, 10, 10, 0.2, 1e-5, 3],
            [4, 4, 1, 3, 10, 10, 0.2, 1e-5, 3],
        ],
    )
    control = {
        "COMPSEGS": [["MS1", comp]],
        "WELSEGS": [["MS1", segments]],
    }
    return convert_multi_segment_wells_deck([well], grid, control)[0]


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase432_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    well = _convert()
    checks = (
        ("isMS", [[float(well.isMS)]], ref["out_is_ms"]),
        ("cells", well.cells + 1, ref["out_cells"].ravel()),
        ("WI", well.WI, ref["out_wi"].ravel()),
        ("dZ", well.dZ, ref["out_dz"].ravel()),
        ("r", well.r, ref["out_r"].ravel()),
        ("perf map", well.perf_map + 1, ref["out_perf_map"].ravel()),
        ("cells to nodes", well.cells_to_nodes + 1, ref["out_cells_to_nodes"]),
        ("topology", well.topo + 1, ref["out_topo"]),
        ("node depth", well.nodes.depth, ref["out_node_depth"].ravel()),
        ("node volume", well.nodes.vol, ref["out_node_vol"].ravel()),
        ("node branch", well.nodes.branch_id, ref["out_node_branch"].ravel()),
        ("segment length", well.segments.length,
         ref["out_seg_length"].ravel()),
        ("segment diameter", well.segments.diameter,
         ref["out_seg_diameter"].ravel()),
        ("segment roughness", well.segments.roughness,
         ref["out_seg_roughness"].ravel()),
    )
    failed = 0
    for name, actual, expected in checks:
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = np.max(np.abs(actual - expected)) if actual.size else 0.0
        ok = actual.shape == expected.shape and error < 1e-12
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={error:.3e}")
    print("\nconvertMultiSegmentWellsDeck matches MRST!" if not failed
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
