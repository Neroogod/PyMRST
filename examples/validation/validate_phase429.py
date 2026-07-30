"""Phase 429: compare setMPFADiscretization sparse operators with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.utils import set_mpfa_discretization
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params import make_rock
from pymrst.utils.units import DARCY, MILLI

REF = Path(__file__).with_name("ref_p429_mpfa_discretization.mat")


def _model():
    grid = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    perm = np.tile(
        np.array([100.0, 35.0, 55.0]) * MILLI * DARCY,
        (grid.cells.num, 1),
    )
    rock = make_rock(grid, perm, 0.2)
    model = SimpleNamespace(
        G=grid,
        rock=rock,
        operators=setup_operators_tpfa(grid, rock),
        fluid=SimpleNamespace(has_pc=False),
        component_state_keys=lambda: [],
    )
    model.FlowDiscretization = FlowDiscretization(model)
    return model


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase429_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    model = set_mpfa_discretization(_model())
    gravity = model.FlowDiscretization.get_state_function(
        "GravityPotentialDifference"
    )
    values = (
        ("pressure gradient", model.operators.mpfagrad.toarray(),
         ref["out_gradient"]),
        ("gravity weight", gravity.weight.toarray(),
         ref["out_gravity_weight"]),
        ("neighbors", model.operators.N + 1, ref["out_neighbors"]),
    )
    failed = 0
    for name, actual, expected in values:
        error = np.max(np.abs(actual - expected))
        scale = max(1.0, np.max(np.abs(expected)))
        ok = actual.shape == expected.shape and error < 5e-11 * scale
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={error:.3e}")
    print("\nsetMPFADiscretization matches MRST!" if not failed
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
