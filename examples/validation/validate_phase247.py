"""Phase 247: shared separator allocation diagnostics parity.

MRST's facility separator tree is a group-level object. When several producers
feed the same separator group, the combined component stream is flashed through
one tree and the resulting stock-tank rates are allocated back to wells. PyMRST
now exposes that allocation on the returned state so the facility behavior is
inspectable instead of only being applied to qOs/qGs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import ATM, BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _multi_producer_model(nx: int = 8) -> OverallCompositionModel:
    grid = compute_geometry(cart_grid([nx, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    wells = add_well(
        None,
        grid,
        rock,
        np.array([0]),
        well_type="rate",
        val=5.0e-5,
        compi=np.array([0.0, 1.0]),
        sign=1,
        name="Injector",
    )
    wells[0].components = np.array([0.9, 0.05, 0.05])
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([nx - 2]),
        well_type="bhp",
        val=55 * BARSA,
        compi=np.array([0.0, 1.0]),
        sign=-1,
        name="ProducerA",
    )
    wells[1].components = np.array([0.5, 0.3, 0.2])
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([nx - 1]),
        well_type="bhp",
        val=54 * BARSA,
        compi=np.array([0.0, 1.0]),
        sign=-1,
        name="ProducerB",
    )
    wells[2].components = np.array([0.45, 0.35, 0.2])
    model = OverallCompositionModel(grid, rock, simple_mixture(), temperature=350.0, wells=wells)
    model.separator_group = SeparatorGroup(EOSSeparator(pressure=ATM, T=300.0))
    return model


def main() -> int:
    model = _multi_producer_model()
    state = model.initial_state(70 * BARSA, [0.5, 0.3, 0.2])
    state.qOs[1:] = np.array([-2.0e-5, -1.0e-5])
    state.qGs[1:] = np.array([-4.0e-6, -2.0e-6])

    out = model._apply_separator_group_to_state(state)
    diag = out.separator_group_diagnostics

    print("\nPhase 247: shared separator allocation diagnostics parity")
    failures = 0
    failures += _check("separator diagnostics are attached to routed producer state", diag is not None)
    if diag is None:
        print("\nPhase 247 failed (missing diagnostics).")
        return 1

    producer_streams = diag["producer_component_mass_streams"]
    total_stream = diag["total_component_mass_stream"]
    shares = diag["allocation_shares"]
    allocated = diag["allocated_surface_rates"]
    totals = diag["positive_surface_rates_total"]

    failures += _check("only producer wells are listed in MRST well order",
                       diag["producer_indices"] == [1, 2]
                       and diag["producer_names"] == ["ProducerA", "ProducerB"])
    failures += _check("per-producer component streams sum to the group stream",
                       producer_streams.shape == (2, model.mix.n)
                       and np.allclose(producer_streams.sum(axis=0), total_stream))
    failures += _check("allocation shares form a normalized producer split",
                       shares.shape == (2,)
                       and np.all(shares >= 0.0)
                       and np.isclose(shares.sum(), 1.0))
    failures += _check("allocated rates match signed state oil/gas rates",
                       allocated.shape == (2, 2)
                       and np.allclose(out.qOs[1:3], allocated[:, 0])
                       and np.allclose(out.qGs[1:3], allocated[:, 1]))
    failures += _check("positive group totals equal the negative allocated sum",
                       totals.shape == (2,)
                       and np.all(totals >= 0.0)
                       and np.allclose(totals, -allocated.sum(axis=0)))
    failures += _check("state copy deep-copies separator allocation arrays",
                       _copy_is_independent(out))

    if failures:
        print(f"\nPhase 247 failed ({failures} checks).")
        return 1
    print("\nShared separator allocation is now exposed on CompositionalState.")
    return 0


def _copy_is_independent(state) -> bool:
    copied = state.copy()
    copied.separator_group_diagnostics["allocation_shares"][0] = 0.0
    return not np.allclose(
        copied.separator_group_diagnostics["allocation_shares"],
        state.separator_group_diagnostics["allocation_shares"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
