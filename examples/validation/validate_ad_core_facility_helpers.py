"""Validate ``packPerforationProperties``, ``unpackPerforationProperties``,
and ``nozzleValve`` against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.models.facilities import (
    nozzle_valve,
    pack_perforation_properties,
    unpack_perforation_properties,
)


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_facility_helpers.mat",
        squeeze_me=True,
    )
    rho_n = np.array([800.0, 900.0])
    diameter = 0.1
    coeff = 0.8
    velocity = np.array([2.0, -3.0])
    area = np.pi * (diameter / 2.0) ** 2
    packed = pack_perforation_properties(
        SimpleNamespace(cells=np.array([1, 3])),
        np.array([100.0, 110.0, 120.0, 130.0]),
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])],
        [np.array([700, 710, 720, 730])],
        [[np.array([0.1, 0.2, 0.3, 0.4])], None],
        [np.array([0.9, 0.8, 0.7, 0.6])],
        [np.array([10, 20]), np.array([30, 40, 50])],
        ["a", "b"],
        [np.array([0, 1]), np.array([1, 0, 1])],
        np.array([[1, 2], [2, 0]]),
        0,
    )
    pressure, mob, density, dissolved, components, extra = (
        unpack_perforation_properties(packed)
    )
    checks = {
        "nozzle velocity": (
            nozzle_valve(velocity, rho_n, diameter, coeff, "velocity"),
            ref["dp_velocity"],
        ),
        "nozzle volume rate": (
            nozzle_valve(velocity * area, rho_n, diameter, coeff, "volumeRate"),
            ref["dp_volume"],
        ),
        "nozzle mass rate": (
            nozzle_valve(velocity * rho_n * area, rho_n, diameter, coeff),
            ref["dp_mass"],
        ),
        "pressure": (pressure, ref["p_out"]),
        "mobility 1": (mob[0], ref["mob_1"]),
        "mobility 2": (mob[1], ref["mob_2"]),
        "density": (density[0], ref["rho_1"]),
        "dissolved": (dissolved[0][0], ref["dissolved_1"]),
        "components": (components[0], ref["comp_1"]),
        "extra 1": (extra[0], ref["extra_1"]),
        "extra 2": (extra[1], ref["extra_2"]),
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
