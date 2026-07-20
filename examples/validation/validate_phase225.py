"""Phase 225: MRST psideh hydrostatic boundary-condition parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.params import psideh  # noqa: E402
from pymrst.utils.gridtools import boundary_face_indices  # noqa: E402


class TwoPhaseFluid:
    def properties(self, state):
        return np.array([2.0, 4.0]), np.array([1000.0, 700.0])

    def relperm(self, sat):
        return np.asarray(sat, dtype=float)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _omega(mu, rho, sat) -> float:
    mob = np.asarray(sat, dtype=float) / np.asarray(mu, dtype=float)
    return float(np.sum(mob * rho) / np.sum(mob))


def validate_hydrostatic_formula() -> int:
    g = compute_geometry(cart_grid([3, 2, 4], [30.0, 20.0, 40.0]))
    sat = np.array([0.25, 0.75])
    fluid = TwoPhaseFluid()
    grav = np.array([0.0, 0.0, 10.0])
    ref_position = np.array([0.0, 0.0, 5.0])
    ref_pressure = 1.0e5

    bc = psideh(
        None,
        g,
        "zmax",
        fluid,
        sat=sat,
        ref_position=ref_position,
        ref_pressure=ref_pressure,
        gravity=grav,
    )

    expected_faces = boundary_face_indices(g, "zmax")
    expected_pressure = (
        ref_pressure
        + _omega([2.0, 4.0], [1000.0, 700.0], sat)
        * ((g.faces.centroids[expected_faces] - ref_position) @ grav)
    )

    failures = 0
    failures += _check("psideh selects the requested side",
                       np.array_equal(np.sort(bc.face), np.sort(expected_faces)))
    failures += _check("psideh stores pressure BCs",
                       set(bc.type) == {"pressure"})
    failures += _check("hydrostatic pressure matches MRST formula",
                       np.allclose(bc.value, expected_pressure, rtol=1e-13, atol=1e-10))
    failures += _check("uniform saturation is stored on every face",
                       bc.sat.shape == (expected_faces.size, 2)
                       and np.allclose(bc.sat, sat))
    return failures


def validate_range_and_aliases() -> int:
    g = compute_geometry(cart_grid([3, 2, 4], [30.0, 20.0, 40.0]))
    fluid = {"mu": [1.0e-3], "rho": [1000.0]}

    bc = psideh(
        None,
        g,
        "bottom",
        fluid,
        i1=[1],
        search_range=[3],
        ref_depth=0.0,
        ref_pressure=2.0e5,
        gravity=[0.0, 0.0, 9.80665],
    )
    cells = np.where(g.faces.neighbors[bc.face, 0] >= 0,
                     g.faces.neighbors[bc.face, 0],
                     g.faces.neighbors[bc.face, 1])
    gids = g.cells.indexMap[cells]
    nx, ny, _ = g.cartDims
    ijk = np.column_stack([gids % nx, (gids // nx) % ny, gids // (nx * ny)])

    expected = 2.0e5 + 1000.0 * (g.faces.centroids[bc.face, 2] * 9.80665)

    failures = 0
    failures += _check("side alias bottom maps to zmax",
                       bc.face.size == 2 and set(bc.type) == {"pressure"})
    failures += _check("i1 restricts the first in-plane axis",
                       np.all(ijk[:, 0] == 1))
    failures += _check("search_range restricts the perpendicular axis",
                       np.all(ijk[:, 2] == 3))
    failures += _check("mapping fluid fallback uses rho/mu",
                       np.allclose(bc.value, expected, rtol=1e-13, atol=1e-10))
    return failures


def validate_error_paths() -> int:
    g = compute_geometry(cart_grid([2, 2, 1], [20.0, 20.0, 10.0]))
    failures = 0
    try:
        psideh(None, g, "xmax", TwoPhaseFluid(), sat=[[0.2, 0.8], [0.3, 0.7]])
    except ValueError:
        failures += _check("nonuniform saturation is rejected", True)
    else:
        failures += _check("nonuniform saturation is rejected", False)

    try:
        psideh(None, g, "xmax", {"mu": [0.0], "rho": [1000.0]})
    except ValueError:
        failures += _check("zero viscosity is rejected", True)
    else:
        failures += _check("zero viscosity is rejected", False)
    return failures


def main() -> int:
    print("\nPhase 225: MRST psideh hydrostatic boundary-condition parity")
    failures = validate_hydrostatic_formula()
    failures += validate_range_and_aliases()
    failures += validate_error_paths()
    if failures:
        print(f"\nPhase 225 failed ({failures} checks).")
        return 1
    print("\npsideh now computes hydrostatic side pressures with MRST's mobility-weighted density formula.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
