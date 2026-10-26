"""MRST ``ad-core/initialization/initEclipseProblemAD.m``.

`init_eclipse_problem_ad` is the port: from a deck (or its file name) it
builds the model the way MRST does: active cells from ACTNUM, pore volume
and permeability; the grid; the fluid; the rock; `selectModelFromDeck` with
`dpMaxRel = 0.2`; the diagonal AD backend for black-oil and compositional
models above 1000 cells. It then returns `initStateDeck`'s initial state,
`convertDeckScheduleToMRST`'s schedule and `getNonLinearSolver`'s solver.

`init_eclipse_problem_ad_ad`, below, is PyMRST's earlier version for the
legacy models, kept for its callers. Its notes follow.

End-to-end construction of a PyMRST simulation problem from a deck (legacy).

This mirrors MRST's ``initEclipseProblemAD`` orchestration: given a deck it
returns the initial state, the model, the schedule and a configured nonlinear
solver, which is the whole of what an MRST deck example does before calling
``simulateScheduleAD``.

Two deck representations reach here and both are handled:

* a **full deck** with RUNSPEC/GRID/PROPS/SOLUTION/SCHEDULE, as
  `read_eclipse_deck` now returns. The grid, rock and fluid come from
  `init_eclipse_grid`, `init_eclipse_rock` and `init_deck_adi_fluid`, exactly
  as MRST builds them;
* PyMRST's **SPE1-style parsed subset**, the flat `x`/`y`/`z`/`poro`/`perm`
  dictionary that predates the full reader.

Supplying ``model`` bypasses model construction, as in MRST. Supplying ``G``
alone - which is what `eggExample` does, because it wants the grid `setupEGG`
already built - takes the rock and fluid from the deck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from pymrst.modules.ad_core.initialization.deck import init_state_deck_ad
from pymrst.modules.ad_core.solvers.get_non_linear_solver import get_non_linear_solver_ad
from pymrst.modules.ad_core.utils.schedule import Schedule, simple_schedule
from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck_ad


def _get(mapping: Mapping[str, Any], *names, default=None):
    for name in names:
        for key in (name, name.lower(), name.upper()):
            if key in mapping:
                return mapping[key]
    return default


def _read_deck(deck):
    if not isinstance(deck, (str, Path)):
        return deck
    from pymrst.modules.deckformat import read_spe1_deck
    return read_spe1_deck(Path(deck))


def _is_full_deck(deck) -> bool:
    """Whether ``deck`` carries the GRID section a full ECLIPSE deck has."""
    grid = _get(deck, "GRID")
    return isinstance(grid, Mapping) and bool(
        {"PORO", "PERMX"} & {str(key).upper() for key in grid})


def _full_deck_inputs(deck, G=None):
    """Grid, rock and fluid from a full deck, as MRST's own chain builds them."""
    from pymrst.gridprocessing import compute_geometry
    from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
        init_deck_adi_fluid,
    )
    from pymrst.modules.deckformat.grid import init_eclipse_grid
    from pymrst.modules.deckformat.params.rock.compress_rock import (
        compress_rock,
    )
    from pymrst.modules.deckformat.params.rock.init_eclipse_rock import (
        init_eclipse_rock,
    )

    if G is None:
        G = compute_geometry(init_eclipse_grid(deck))
    rock = init_eclipse_rock(deck)
    index_map = getattr(getattr(G, "cells", None), "indexMap", None)
    if index_map is not None:
        # `initEclipseRock` returns one row per *Cartesian* cell. Against an
        # ACTNUM grid that is 25200 rows for the Egg model's 18553 cells, and
        # every cell past the first inactive one takes another cell's rock.
        rock = compress_rock(rock, index_map)
    # `init_deck_adi_fluid` returns MRST's struct; the models are written
    # against the `BlackOilFluid` object. Same property functions, one naming.
    from pymrst.modules.ad_props.props.deck_fluid import deck_fluid_object
    return G, rock, deck_fluid_object(init_deck_adi_fluid(deck)), None


def _automatic_inputs(deck, G=None):
    if _is_full_deck(deck):
        return _full_deck_inputs(deck, G)

    required = ("x", "y", "z", "poro", "perm")
    if not all(_get(deck, key) is not None for key in required):
        raise NotImplementedError(
            "automatic grid/rock/fluid construction requires either a full "
            "deck with a GRID section or the PyMRST SPE1-style "
            "x/y/z/poro/perm parsed-deck subset; provide model=... for "
            "another deck representation"
        )
    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
    from pymrst.params import add_well, make_rock

    G = compute_geometry(tensor_grid(
        np.asarray(_get(deck, "x")), np.asarray(_get(deck, "y")),
        np.asarray(_get(deck, "z")),
    ))
    rock = make_rock(G, np.asarray(_get(deck, "perm")),
                     np.asarray(_get(deck, "poro"))[:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name, spec in (_get(deck, "wells", default={}) or {}).items():
        wells = add_well(
            wells, G, rock, spec["cells"], well_type=spec["type"],
            val=spec["val"], radius=spec["radius"], compi=spec["compi"],
            sign=spec["sign"], name=name, ref_depth=spec["ref_depth"],
            gravity=np.array([0.0, 0.0, 9.80665]),
        )
    return G, rock, fluid, wells


def _schedule_from_deck(deck, model) -> Optional[Schedule]:
    schedule = _get(deck, "schedule")
    if isinstance(schedule, Schedule):
        return schedule
    if hasattr(schedule, "control") and hasattr(schedule, "step"):
        # A full deck's SCHEDULE section: control periods of raw well records
        # plus the step lengths that point at them.
        from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
            convert_deck_schedule_to_mrst_ad,
        )
        return convert_deck_schedule_to_mrst_ad(
            model, {"step": schedule.step, "control": schedule.control})
    if isinstance(schedule, Mapping):
        from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
            convert_deck_schedule_to_mrst_ad,
        )
        return convert_deck_schedule_to_mrst_ad(model, schedule)
    timesteps = _get(deck, "tsteps", "dt")
    if timesteps is None:
        return None
    wells = getattr(model, "wells", getattr(model, "W", None))
    return simple_schedule(timesteps, W=wells)


def init_eclipse_problem_ad_ad(
    deck,
    *,
    model=None,
    G=None,
    rock=None,
    fluid=None,
    schedule=None,
    get_schedule: bool = True,
    get_initial_state: bool = True,
    timestep_strategy: str = "iteration",
    use_cpr: bool = True,
    max_iterations: int = 12,
    use_relaxation: bool = True,
    gravity: float = 9.80665,
):
    """Return ``(state0, model, schedule, nonlinear_solver)``.

    Supplying ``model`` bypasses physical-model construction, matching MRST.
    Supplying ``G``, ``rock`` and ``fluid`` selects a model from ``RUNSPEC``.
    """
    deck = _read_deck(deck)
    if not isinstance(deck, Mapping):
        raise TypeError("deck must be a path or parsed mapping")

    if model is None:
        wells = None
        if G is None or rock is None or fluid is None:
            G_auto, rock_auto, fluid_auto, wells = _automatic_inputs(deck, G)
            G = G if G is not None else G_auto
            rock = rock if rock is not None else rock_auto
            fluid = fluid if fluid is not None else fluid_auto
        selection_deck = deck
        if not _get(deck, "RUNSPEC"):
            selection_deck = dict(deck)
            selection_deck["RUNSPEC"] = {
                "WATER": True, "OIL": True, "GAS": True,
                "BLACKOIL": True, "DISGAS": _get(deck, "pvto") is not None,
            }
            selection_deck["PROPS"] = {
                "PVTO": _get(deck, "pvto"), "PVTG": _get(deck, "pvtg")
            }
        model = select_model_from_deck_ad(
            G, rock, fluid, selection_deck, gravity=gravity, wells=wells or ()
        )

    state0 = init_state_deck_ad(model, deck, gravity=gravity) if get_initial_state else None
    schedule_out = (
        schedule if schedule is not None else _schedule_from_deck(deck, model)
    ) if get_schedule else None
    if state0 is not None:
        from pymrst.modules.ad_core.utils.init_well_sol_ad import init_well_sol_ad
        wells = (
            schedule_out.controls[0].W
            if schedule_out is not None and schedule_out.controls
            else getattr(model, "wells", ())
        )
        state0["wellSol"] = init_well_sol_ad(wells, model, state0)
        _seed_well_unknowns(model, state0, wells)
    nonlinear = get_non_linear_solver_ad(
        model, timestep_strategy=timestep_strategy, use_cpr=use_cpr,
        max_iterations=max_iterations, use_relaxation=use_relaxation,
    )
    return state0, model, schedule_out, nonlinear


def _seed_well_unknowns(model, state, wells) -> None:
    """Put the well primary variables on the state.

    MRST keeps a well's unknowns inside `wellSol`; PyMRST's black-oil models
    read them off the state as `qWs`/`qOs`/`qGs` and `bhp`, because that is
    where `init_state_ad` seeds their derivatives. Building the `wellSol` and
    stopping there leaves the model with no well unknowns at all, and the
    first assembly raises on the missing key.

    The seed itself matters for how the first Newton step behaves:

    * a **rate-controlled** well starts at its own target, split by `compi`.
      Starting it at zero asks the first iteration to find the whole rate,
      which for an injector at 79.5 m3/day is the entire solution;
    * a **pressure-controlled** well starts at zero rate, because its rate is
      genuinely unknown until the drawdown is evaluated.

    The bottom-hole pressures come from `init_well_sol_ad`, which offsets each
    well from *its own* first perforation rather than from the field mean - so
    a deep well does not start below its own formation pressure.
    """
    wells = list(wells or ())
    if not wells:
        return
    phases = tuple(str(name).upper()[0] for name in model.get_phase_names())
    n = len(wells)
    rates = {f"q{phase}s": np.zeros(n) for phase in phases}
    bhp = np.zeros(n)

    rows = list(state.get("wellSol") or ())
    for k, well in enumerate(wells):
        row = rows[k] if k < len(rows) else None
        composition = np.atleast_1d(np.asarray(well.compi, dtype=float))
        if str(well.type).lower() == "bhp":
            bhp[k] = float(well.val)
        else:
            bhp[k] = (float(getattr(row, "bhp", 0.0)) if row is not None
                      else float(np.mean(state["pressure"])))
            for index, phase in enumerate(phases):
                if index < composition.size:
                    rates[f"q{phase}s"][k] = float(well.val)*composition[index]

    for name, values in rates.items():
        state[name] = values
    state["bhp"] = bhp


def init_eclipse_problem_ad(deck, *, useMexGeometry=None, useMexProcessGrid=None,
                            TimestepStrategy: str = "iteration", useCPR: bool = True,
                            useMex=None, pvtMethodOil: str = "parallel",
                            pvtMethodGas: str = "linshift", optimizeTables: bool = False,
                            rowMajorAD: bool = False, AutoDiffBackend=None,
                            UniformFacilityModel: bool = False, model=None, G=None,
                            permTolerance: float = 0.0, maxIterations: int = 12,
                            useRelaxation: bool = True, deckfn=None,
                            getSchedule: bool = True, getInitialState: bool = True,
                            SplitDisconnected: bool = False, PreserveCpNodes: bool = True,
                            useLegacyModels: bool = False, **extra):
    """MRST `initEclipseProblemAD`: `(state0, model, schedule, nonlinear)`."""
    from pymrst.modules.ad_core.initialization.init_state_deck import init_state_deck
    from pymrst.modules.ad_core.solvers.get_non_linear_solver import (
        get_non_linear_solver,
    )
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
    from pymrst.params.gravity.gravity import gravity
    from pymrst.utils.settings_manager.mrst_settings import mrst_settings

    gravity("reset", "on")
    if isinstance(deck, (str, Path)):
        deck = read_eclipse_deck(deck)
    deck = convert_deck_units(deck)
    opt = dict(useMexGeometry=useMexGeometry, useMexProcessGrid=useMexProcessGrid,
               useMex=mrst_settings("get", "useMEX") if useMex is None else useMex,
               pvtMethodOil=pvtMethodOil, pvtMethodGas=pvtMethodGas,
               optimizeTables=optimizeTables, G=G, permTolerance=permTolerance,
               SplitDisconnected=SplitDisconnected, PreserveCpNodes=PreserveCpNodes,
               useLegacyModels=useLegacyModels)
    if deckfn is not None:
        deck = deckfn(deck)
    if opt["useMexGeometry"] is None:
        opt["useMexGeometry"] = opt["useMex"]
    if opt["useMexProcessGrid"] is None:
        opt["useMexProcessGrid"] = opt["useMex"]
    model = _initialize_model(deck, opt) if model is None else model
    if AutoDiffBackend is None:
        heavy = any(c.__name__ in ("NaturalVariablesCompositionalModel",
                                   "ThreePhaseBlackOilModel")
                    for c in type(model).__mro__)
        if heavy and model.G.cells.num > 1000:
            AutoDiffBackend = "diagonal-mex" if opt["useMex"] else "diagonal"
        else:
            AutoDiffBackend = "sparse"
    backend = AutoDiffBackend.lower()
    if backend in ("diagonal-mex", "diagonal"):
        from pymrst.modules.ad_core.backends.diagonal.diagonal_auto_diff_backend import (
            DiagonalAutoDiffBackend,
        )
        model.AutoDiffBackend = DiagonalAutoDiffBackend(
            use_mex=True if backend == "diagonal-mex" else bool(opt["useMex"]),
            row_major=rowMajorAD)
    elif backend != "sparse":
        raise ValueError(f"Unknown backend string {AutoDiffBackend}")
    cnames = model.get_component_names()
    if not cnames and UniformFacilityModel and not getattr(model, "isGenericReservoirModel", False):
        raise NotImplementedError("UniformFacilityModel is not ported onto the AD-OO base.")
    state0 = init_state_deck(model, deck) if getInitialState else None
    schedule = convert_deck_schedule_to_mrst(model, deck, **extra) if getSchedule else None
    nonlinear = get_non_linear_solver(model, TimestepStrategy=TimestepStrategy,
                                      useCPR=useCPR, maxIterations=maxIterations,
                                      useRelaxation=useRelaxation)
    return state0, model, schedule, nonlinear


def _initialize_model(deck, opt):
    """MRST's `initializeModel`: grid, rock and fluid from the deck, then
    `selectModelFromDeck`."""
    from pymrst.gridprocessing import compute_geometry
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
        init_deck_adi_fluid,
    )
    from pymrst.modules.deckformat.grid import init_eclipse_grid
    from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
    from pymrst.modules.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
    import warnings

    rock = init_eclipse_rock(deck)
    G = opt["G"]
    if G is None:
        grid = deck["GRID"]
        if "ACTNUM" not in grid:
            grid["ACTNUM"] = np.ones(int(np.prod(grid["cartDims"])))
        pv = np.asarray(rock.poro, dtype=float).ravel()
        if rock.ntg is not None:
            pv = pv*np.asarray(rock.ntg, dtype=float).ravel()
        perm = np.asarray(rock.perm, dtype=float)
        perm_ok = np.any(perm.reshape(perm.shape[0], -1) > opt["permTolerance"], axis=1)
        grid["ACTNUM"] = ((np.asarray(grid["ACTNUM"]).ravel() > 0) & (pv > 0)
                          & perm_ok).astype(float)
        G = init_eclipse_grid(deck, split_disconnected=opt["SplitDisconnected"],
                              preserve_cp_nodes=(not opt["useMexProcessGrid"]
                                                 and opt["PreserveCpNodes"]))
        if isinstance(G, (list, tuple)):
            if len(G) > 1:
                warnings.warn("Multiple disconnected grids found. Picking largest.")
            G = G[0]
    if getattr(G.cells, "centroids", None) is None:
        G = compute_geometry(G)
    fluid = init_deck_adi_fluid(deck, grid=G, optimize=opt["optimizeTables"],
                                pvt_method_gas=opt["pvtMethodGas"],
                                pvt_method_oil=opt["pvtMethodOil"])
    rock = compress_rock(rock, G.cells.indexMap)
    model = select_model_from_deck(G, rock, fluid, deck,
                                   useLegacyModels=opt["useLegacyModels"])
    model.dpMaxRel = 0.2
    return model


initEclipseProblemAD = init_eclipse_problem_ad

__all__ = ["init_eclipse_problem_ad", "init_eclipse_problem_ad_ad",
           "initEclipseProblemAD"]
