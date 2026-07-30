"""Validate ``pressureBCContrib``, ``pressureBCContribADI``, and
``combineEquations`` against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import (
    bc_to_ad_bc,
    combine_equations,
    pressure_bc_contrib,
    pressure_bc_contrib_adi,
)
from pymrst.utils.adi import init_variables_adi


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_pressure_bc_equations.mat",
        squeeze_me=True,
    )
    grid = SimpleNamespace(
        faces=SimpleNamespace(
            neighbors=np.array([[-1, 0], [0, 1], [1, -1]]),
            centroids=np.array([[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0]]),
        ),
        cells=SimpleNamespace(
            num=2, centroids=np.array([[0, 0, 1.0], [0, 0, 2.0]])
        ),
    )
    operators = SimpleNamespace(T_all=np.array([2.0, 4.0, 3.0]))
    boundary = bc_to_ad_bc(grid, SimpleNamespace(
        face=np.array([0, 2]),
        type=np.array(["pressure", "pressure"]),
        value=np.array([120.0, 80.0]),
    ))
    pressure = [np.array([100.0, 100.0])]
    density = [np.array([2.0, 3.0])]
    mobility = [np.array([0.5, 0.25])]
    bfactor = [np.array([1.2, 1.5])]
    legacy, legacy_cells = pressure_bc_contrib(
        grid, operators, pressure, density, mobility, bfactor, boundary, gravity=10
    )
    (pressure_ad,) = init_variables_adi(pressure[0])
    adi, adi_cells = pressure_bc_contrib_adi(
        grid,
        operators,
        [pressure_ad],
        density,
        mobility,
        bfactor,
        boundary,
        gravity=10,
    )
    p_ad, bhp_ad = init_variables_adi(np.array([1.0, 2.0]), 5.0)
    combined = combine_equations(3 * p_ad + bhp_ad, 2 * bhp_ad)
    checks = {
        "legacy values": (legacy[0], ref["legacy_value"]),
        "legacy cells": (legacy_cells + 1, ref["legacy_cells"]),
        "ADI values": (adi[0].val, ref["adi_value"]),
        "ADI cells": (adi_cells + 1, ref["adi_cells"]),
        "ADI Jacobian": (adi[0].jac[0].toarray(), ref["adi_jacobian"]),
        "numeric equations": (
            combine_equations([1, 2], 3),
            ref["numeric_combined"],
        ),
        "combined values": (combined.val, ref["combined_value"]),
        "combined Jacobian": (
            combined.jac[0].toarray(),
            ref["combined_jacobian"],
        ),
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
