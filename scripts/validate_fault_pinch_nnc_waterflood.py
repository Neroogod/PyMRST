#!/usr/bin/env python3
"""Validate the PyMRST fault/pinch/NNC example against saved MRST output."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.examples import run_fault_pinch_nnc_waterflood


def validate(reference: Path) -> None:
    mrst = loadmat(reference, squeeze_me=True)
    pymrst = run_fault_pinch_nnc_waterflood()

    np.testing.assert_array_equal(
        pymrst.grid.cells.indexMap + 1, np.ravel(mrst["active_index_map"])
    )
    np.testing.assert_array_equal(
        pymrst.grid.nnc["cells"] + 1, np.asarray(mrst["nnc_cells"])
    )
    np.testing.assert_allclose(
        pymrst.grid.nnc["trans"], np.ravel(mrst["nnc_trans"]), rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        [fault.numf for fault in pymrst.faults],
        np.ravel(mrst["fault_face_counts"]),
    )
    np.testing.assert_allclose(
        [fault.multiplier for fault in pymrst.faults],
        np.ravel(mrst["fault_multipliers"]), rtol=0, atol=0,
    )

    comparisons = {
        "pressure": (pymrst.pressure, 0.1),
        "sw": (pymrst.saturation, 1.1e-3),
        "bhp": (pymrst.bhp, 0.1),
        "qws": (pymrst.qws, 2e-13),
        "qos": (pymrst.qos, 2e-13),
    }
    for name, (actual, tolerance) in comparisons.items():
        expected = np.asarray(mrst[name])
        maximum = float(np.max(np.abs(actual - expected)))
        if maximum > tolerance:
            raise AssertionError(
                f"{name}: maximum absolute error {maximum:g} exceeds {tolerance:g}"
            )
        print(f"{name:8s} maximum absolute error: {maximum:.9g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="MRST .mat result file")
    args = parser.parse_args()
    validate(args.reference)
    print("fault/pinch/NNC MRST validation passed")


if __name__ == "__main__":
    main()
