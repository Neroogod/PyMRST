"""Phase 24: the state-function graph, validated against MATLAB ad-blackoil.

The graph has no reference of its own - MRST's `ad-core` is machinery, not
physics, and there is no array to dump from it. What it has instead is a
much stronger check: **it must reproduce the physics that is already
validated**.

So this replays the Phases 5a and 5b comparisons - the same cases, the same
MATLAB reference files (`ref_p5_ow.mat`, `ref_p5b_spe1.mat`) - but assembles
the residual through the graph (`GenericBlackOilModel`) instead of through
the hand-rolled models. If the graph is right, both cases still match MATLAB
to the tolerances they matched at before, and the graph inherits their
validation. If a state function is wrong, a dependency is missing, or the
cache returns a stale value, one of these numbers moves.

The graph model is also compared directly against the hand-rolled models,
which is a tighter check than either is against MATLAB - two Python codes
solving the identical discrete system should agree to round-off, not merely
to solver tolerance.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    initial_state_from_deck,
)
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.state_functions import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
SPE1_STEPS = 30


# -- Phase 5a's case, rebuilt on the graph ------------------------------------

def build_ow():
    """The exact case of validate_phase5.py (15x15 heterogeneous quarter spot)."""
    g = compute_geometry(cart_grid([15, 15, 1], [300.0, 300.0, 10.0]))
    kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
    )
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    return g, rock, fluid, w


def ow_initial(g, wells, p0=200 * BARSA, sw0=0.2) -> State:
    return State(
        pressure=np.full(g.cells.num, float(p0)),
        sW=np.full(g.cells.num, float(sw0)),
        qWs=np.array([w.val * w.compi[0] if w.type == "rate" else 0.0
                      for w in wells]),
        qOs=np.array([w.val * w.compi[1] if w.type == "rate" else 0.0
                      for w in wells]),
        bhp=np.array([w.val if w.type == "bhp" else float(p0) for w in wells]),
    )


def run_ow_graph() -> Tuple[Any, Dict[str, np.ndarray]]:
    """Phase 5a's schedule, assembled through the state-function graph."""
    g, rock, fluid, wells = build_ow()
    model = GenericBlackOilModel(g, rock, fluid, wells)
    solver = NonLinearSolver()
    st = ow_initial(g, wells)

    hist: Dict[str, List[np.ndarray]] = {k: [] for k in
                                         ("p", "sw", "qWs", "qOs", "bhp")}
    for i, dt in enumerate(np.full(8, 30.0) * DAY):
        st, rep = solver.solve_ministep(model, st, st.copy(), float(dt))
        assert rep.converged, f"oil-water step {i} failed: {rep.residuals[-1]}"
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("bhp", st["bhp"])):
            hist[key].append(val.copy())
    return wells, {k: np.vstack(v) for k, v in hist.items()}


def run_ow_hand() -> Dict[str, np.ndarray]:
    """The same schedule through the hand-rolled model, for the direct check."""
    from pymrst.modules.ad_core import simulate_schedule_ad

    g, rock, fluid, wells = build_ow()
    model = TwoPhaseOilWaterModelAD(g, rock, fluid, wells)
    states, _ = simulate_schedule_ad(model.initial_state(200 * BARSA, 0.2),
                                     model, np.full(8, 30.0) * DAY)
    return {
        "p": np.vstack([s.pressure for s in states]),
        "sw": np.vstack([s.sW for s in states]),
        "qWs": np.vstack([s.qWs for s in states]),
        "qOs": np.vstack([s.qOs for s in states]),
        "bhp": np.vstack([s.bhp for s in states]),
    }


# -- Phase 5b's case (SPE 1), rebuilt on the graph -----------------------------

def build_spe1(order):
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name in order:
        wd = deck["wells"][name]
        wells = add_well(
            wells, g, rock, wd["cells"], well_type=wd["type"], val=wd["val"],
            radius=wd["radius"], compi=wd["compi"], sign=wd["sign"],
            name=name, ref_depth=wd["ref_depth"],
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )
    return deck, g, rock, fluid, wells


def run_spe1_graph(order) -> Tuple[Any, Dict[str, np.ndarray]]:
    """SPE 1's first 30 report steps, assembled through the graph."""
    deck, g, rock, fluid, wells = build_spe1(order)
    model = GenericBlackOilModel(g, rock, fluid, wells, disgas=True,
                                 gravity=NORM_GRAVITY, tol_well=1e-8)
    solver = NonLinearSolver()
    st = initial_state_from_deck(model, deck)

    hist: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs", "bhp")
    }
    for i, dt in enumerate(deck["tsteps"][:SPE1_STEPS]):
        st, rep = solver.solve_ministep(model, st, st.copy(), float(dt))
        assert rep.converged, f"SPE1 step {i} failed: {rep.residuals[-1]}"
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("sg", st["sG"]), ("rs", st["rs"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("qGs", st["qGs"]), ("bhp", st["bhp"])):
            hist[key].append(val.copy())
    return model, {k: np.vstack(v) for k, v in hist.items()}


def run_spe1_hand(order) -> Dict[str, np.ndarray]:
    deck, g, rock, fluid, wells = build_spe1(order)
    model = ThreePhaseBlackOilModelAD(g, rock, fluid, wells)
    st = initial_state_spe1(model, deck)
    hist: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs", "bhp")
    }
    for dt in deck["tsteps"][:SPE1_STEPS]:
        st, ok, _ = model.step(st, dt)
        assert ok
        for key, val in (("p", st.pressure), ("sw", st.s[:, 0]),
                         ("sg", st.s[:, 2]), ("rs", st.rs), ("qWs", st.qWs),
                         ("qOs", st.qOs), ("qGs", st.qGs), ("bhp", st.bhp)):
            hist[key].append(val.copy())
    return {k: np.vstack(v) for k, v in hist.items()}


# -- comparison ---------------------------------------------------------------

def compare(title: str, checks) -> int:
    print(title)
    failed = 0
    for name, py, ref, rtol in checks:
        py = np.asarray(py, dtype=float).ravel()
        ref = np.asarray(ref, dtype=float).ravel()
        scale = max(np.abs(ref).max(), 1e-300)
        diff = float(np.abs(py - ref).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
              f"{diff:.3e} (rel {diff / scale:.2e}, rtol {rtol:.0e})")
        failed += 0 if ok else 1
    return failed


def main() -> int:
    ow_ref = HERE / "ref_p5_ow.mat"
    spe1_ref = HERE / "ref_p5b_spe1.mat"
    if not ow_ref.exists() or not spe1_ref.exists():
        print("ref_p5_ow.mat / ref_p5b_spe1.mat missing (run "
              "dump_phase5_reference.m and dump_phase5b_reference.m)")
        return 1

    failed = 0

    # 1. Oil-water through the graph, against MATLAB (Phase 5a's reference).
    ref = scipy.io.loadmat(ow_ref, squeeze_me=True)
    wells, g_ow = run_ow_graph()
    failed += compare(
        "Oil-water through the state-function graph, vs MATLAB ad-blackoil:",
        [("WI", np.concatenate([wells[0].WI, wells[1].WI]),
          np.concatenate([np.atleast_1d(ref["WI1"]),
                          np.atleast_1d(ref["WI2"])]), 1e-12),
         ("pressure history", g_ow["p"], ref["p_hist"], 5e-6),
         ("sW history", g_ow["sw"], ref["sw_hist"], 5e-6),
         ("qWs history", g_ow["qWs"], ref["qWs"], 5e-6),
         ("qOs history", g_ow["qOs"], ref["qOs"], 5e-6),
         ("bhp history", g_ow["bhp"], ref["bhp"], 5e-6)],
    )

    # 2. The graph against the hand-rolled model: two codes, one system.
    h_ow = run_ow_hand()
    failed += compare(
        "\nOil-water: graph vs hand-rolled model (must agree to round-off):",
        [(k, g_ow[k], h_ow[k], 1e-11) for k in ("p", "sw", "qWs", "qOs", "bhp")],
    )

    # 3. SPE 1 through the graph, against MATLAB (Phase 5b's reference).
    ref = scipy.io.loadmat(spe1_ref, squeeze_me=True)
    order = ("PRODUCER", "INJECTOR") if bool(ref["isProdFirst"]) \
        else ("INJECTOR", "PRODUCER")
    model, g_spe1 = run_spe1_graph(order)
    wi = np.concatenate([w.WI for w in model.wells])
    failed += compare(
        "\nSPE 1 (three-phase, live oil) through the graph, vs MATLAB:",
        [("WI", wi, ref["WI"], 1e-10),
         ("pressure history", g_spe1["p"], ref["p_hist"], 2e-5),
         ("sW history", g_spe1["sw"], ref["sw_hist"], 2e-5),
         ("sG history", g_spe1["sg"], ref["sg_hist"], 2e-5),
         ("rs history", g_spe1["rs"], ref["rs_hist"], 2e-5),
         ("qWs history", g_spe1["qWs"], ref["qWs"], 2e-5),
         ("qOs history", g_spe1["qOs"], ref["qOs"], 2e-5),
         ("qGs history", g_spe1["qGs"], ref["qGs"], 2e-5),
         ("bhp history", g_spe1["bhp"], ref["bhp"], 2e-5)],
    )

    # 4. SPE 1: graph against hand-rolled, flash and all.
    h_spe1 = run_spe1_hand(order)
    failed += compare(
        "\nSPE 1: graph vs hand-rolled model (must agree to round-off):",
        [(k, g_spe1[k], h_spe1[k], 1e-11)
         for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs", "bhp")],
    )

    print("\nThe state-function graph reproduces every validated result!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
