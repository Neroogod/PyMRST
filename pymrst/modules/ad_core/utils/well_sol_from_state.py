"""MRST's `wellSol` rows, rebuilt from a solved state (PyMRST bridge).

No MRST counterpart, because MRST has no gap to bridge: `simulateScheduleAD`
returns `wellSols` as its first output and every well quantity lives there.

PyMRST's black-oil models keep the well unknowns on the **state** instead - as
`bhp`, `qWs`, `qOs`, `qGs` - because that is where `init_state_ad` seeds their
derivatives. The `wellSol` a state carries is the one `init_well_sol_ad`
seeded before the run and *the solve does not update it*. Anything reading
those rows therefore gets each well's initial guess for the whole run: every
pressure-controlled producer reports a rate of exactly zero, which looks like a
shut-in well rather than like a bug.

This assembles the rows MRST would have returned, so `get_well_output` and the
plotting hooks read what they expect.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def well_sol_from_state(model: Any, state: Any) -> List[Dict[str, Any]]:
    """One `wellSol` row per well, from the solved state."""
    wells = list(getattr(getattr(model, "facility", model), "wells", ()) or ())
    phases = [str(name).upper()[0] for name in model.get_phase_names()]
    rows: List[Dict[str, Any]] = []
    for index, well in enumerate(wells):
        row: Dict[str, Any] = {
            "name": well.name,
            "sign": int(well.sign),
            "status": bool(getattr(well, "status", True)),
            "type": str(well.type),
            "bhp": _at(state, "bhp", index),
        }
        for phase in phases:
            row[f"q{phase}s"] = _at(state, f"q{phase}s", index)
        rows.append(row)
    return rows


def well_sols_from_states(model: Any, states) -> List[List[Dict[str, Any]]]:
    """`wellSols` for a whole run - MRST's first `simulateScheduleAD` output."""
    return [well_sol_from_state(model, state) for state in states]


wellSolFromState = well_sol_from_state
wellSolsFromStates = well_sols_from_states


def _at(state: Any, key: str, index: int) -> float:
    values = state.get(key) if hasattr(state, "get") else None
    if values is None:
        return float("nan")
    flat = np.ravel(np.asarray(values, dtype=float))
    return float(flat[index]) if index < flat.size else float("nan")


__all__ = ["wellSolFromState", "wellSolsFromStates", "well_sol_from_state",
           "well_sols_from_states"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core import State
    from pymrst.modules.ad_core.simulators.nonlinear_solver import (
        NonLinearSolver,
    )
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _case(n=6):
        g = compute_geometry(cart_grid([n, n, 1], [100.0, 100.0, 10.0]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=200*BARSA)
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0, 0.0]), sign=1, name="I")
        W = add_well(W, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150*BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     name="P")
        model = GenericBlackOilModelAD(g, rock, fluid, W)
        state = State(pressure=np.full(g.cells.num, 200*BARSA),
                      sW=np.full(g.cells.num, 0.2),
                      qWs=np.array([1e-4, 0.0]), qOs=np.array([0.0, 0.0]),
                      bhp=np.array([210*BARSA, 150*BARSA]))
        solver = NonLinearSolver()
        solver.verbose = False
        states, _ = solver.simulate_schedule(
            model, state, simple_schedule(np.full(3, 10.0*DAY), W=W))
        return model, states

    MODEL, STATES = _case()

    def test_a_row_per_well_with_mrsts_field_names():
        rows = well_sol_from_state(MODEL, STATES[-1])
        assert [row["name"] for row in rows] == ["I", "P"]
        assert set(rows[0]) >= {"name", "sign", "bhp", "qWs", "qOs"}
        print("ok test_a_row_per_well_with_mrsts_field_names")

    def test_it_reports_the_solved_rate_not_the_seed():
        """The whole reason this exists. The `wellSol` rows a state carries
        are `init_well_sol_ad`'s seeds; a pressure-controlled producer's rate
        there is a machine epsilon, which reads as a shut-in well."""
        from pymrst.modules.ad_core.utils.init_well_sol_ad import (
            init_well_sol_ad,
        )
        rows = well_sol_from_state(MODEL, STATES[-1])
        producer = next(row for row in rows if row["sign"] < 0)
        assert producer["qOs"] < -1e-9, producer

        seeds = init_well_sol_ad(MODEL.facility.wells, MODEL, STATES[-1])
        stale = float(np.ravel(np.asarray(seeds[1].qs[1], dtype=float))[0])
        assert abs(stale) < 1e-12, ("the seed carries a real rate after all; "
                                    "this bridge may no longer be needed")
        print(f"ok test_it_reports_the_solved_rate_not_the_seed "
              f"(solved {producer['qOs']:.3e}, seed {stale:.1e})")

    def test_the_bottom_hole_pressure_is_the_states():
        rows = well_sol_from_state(MODEL, STATES[-1])
        assert np.allclose([row["bhp"] for row in rows],
                           np.asarray(STATES[-1]["bhp"], dtype=float))
        print("ok test_the_bottom_hole_pressure_is_the_states")

    def test_it_feeds_get_well_output():
        """The consumer it exists for: MRST's own well-curve extractor."""
        from pymrst.modules.ad_core.utils.get_well_output import (
            get_well_output_ad,
        )
        sols = well_sols_from_states(MODEL, STATES)
        data, names = get_well_output_ad(sols, "qOs", [1])
        assert names == ["P"] and data.shape == (3, 1, 1)
        assert np.all(data < 0.0), data
        print("ok test_it_feeds_get_well_output")

    def test_a_missing_field_is_nan_not_zero():
        """A phase the state does not carry has no rate; zero would read as a
        well that is open and producing nothing."""
        stripped = STATES[-1].copy()
        del stripped["qOs"]
        row = well_sol_from_state(MODEL, stripped)[0]
        assert np.isnan(row["qOs"]), row
        print("ok test_a_missing_field_is_nan_not_zero")

    test_a_row_per_well_with_mrsts_field_names()
    test_it_reports_the_solved_rate_not_the_seed()
    test_the_bottom_hole_pressure_is_the_states()
    test_it_feeds_get_well_output()
    test_a_missing_field_is_nan_not_zero()
    print("\nAll tests passed!")
