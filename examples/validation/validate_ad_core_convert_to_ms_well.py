"""Validate ``convert2MSWell`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

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


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_convert_to_ms_well.mat",
        squeeze_me=True,
    )
    grid = SimpleNamespace(cells=SimpleNamespace(centroids=np.array([
        [0.0, 0, 100], [3.0, 0, 104], [3.0, 4, 110]
    ])))
    default = convert_to_ms_well(_well(), grid=grid)
    grouped = convert_to_ms_well(
        _well(),
        cell2node=np.array([0, 0, 1]),
        topology=np.array([[0, 1]]),
        segment_length=[12.0],
        segment_diameter=[0.4],
        conn_dz=[1, 2, 3],
        volume=[7, 8],
    )
    checks = {
        "default cell2node": (
            default.cell2node.toarray(),
            ref["default_cell2node"],
        ),
        "default depth": (default.nodes.depth, ref["default_depth"]),
        "default volume": (default.nodes.vol, ref["default_volume"]),
        "default topology": (
            default.segments.topo + 1,
            ref["default_topology"],
        ),
        "default length": (
            default.segments.length,
            ref["default_length"],
        ),
        "default diameter": (
            default.segments.diam,
            ref["default_diameter"],
        ),
        "default delta": (
            default.segments.deltaDistance,
            ref["default_delta"],
        ),
        "default connDZ": (default.connDZ, ref["default_conn_dz"]),
        "grouped cell2node": (
            grouped.cell2node.toarray(),
            ref["grouped_cell2node"],
        ),
        "grouped depth": (grouped.nodes.depth, ref["grouped_depth"]),
        "grouped volume": (grouped.nodes.vol, ref["grouped_volume"]),
        "grouped topology": (
            grouped.segments.topo + 1,
            ref["grouped_topology"],
        ),
        "grouped length": (
            grouped.segments.length,
            ref["grouped_length"],
        ),
        "grouped diameter": (
            grouped.segments.diam,
            ref["grouped_diameter"],
        ),
        "grouped connDZ": (grouped.connDZ, ref["grouped_conn_dz"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
