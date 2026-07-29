"""End-to-end construction of a PyMRST simulation problem from a deck.

This mirrors MRST's ``initEclipseProblemAD`` orchestration.  PyMRST currently
has a complete automatic reader for its SPE1-style parsed-deck subset; callers
with other parsed decks can provide ``model`` (or ``G``, ``rock`` and
``fluid``) and still use the common state, schedule and solver construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from pymrst.modules.ad_core.initialization.deck import init_state_deck
from pymrst.modules.ad_core.solvers.get_non_linear_solver import get_non_linear_solver
from pymrst.modules.ad_core.utils.schedule import Control, Schedule, simple_schedule
from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck


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


def _automatic_inputs(deck):
    required = ("x", "y", "z", "poro", "perm")
    if not all(_get(deck, key) is not None for key in required):
        raise NotImplementedError(
            "automatic grid/rock/fluid construction currently requires the "
            "PyMRST SPE1-style x/y/z/poro/perm parsed-deck subset; provide "
            "model=... for another deck representation"
        )
    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
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
    if isinstance(schedule, Mapping):
        step = _get(schedule, "step", default=schedule)
        values = _get(step, "val", "step_val")
        indices = _get(step, "control", "step_control")
        controls_raw = _get(schedule, "controls", "control", default=[])
        controls = [
            c if isinstance(c, Control) else Control(
                W=_get(c, "W", "wells"), bc=_get(c, "bc"), src=_get(c, "src")
            )
            for c in controls_raw
        ]
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size and indices.min() == 1:
            indices = indices - 1
        return Schedule(values, indices, controls)
    timesteps = _get(deck, "tsteps", "dt")
    if timesteps is None:
        return None
    wells = getattr(model, "wells", getattr(model, "W", None))
    return simple_schedule(timesteps, W=wells)


def init_eclipse_problem_ad(
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
            G_auto, rock_auto, fluid_auto, wells = _automatic_inputs(deck)
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
        model = select_model_from_deck(
            G, rock, fluid, selection_deck, gravity=gravity, wells=wells or ()
        )

    state0 = init_state_deck(model, deck, gravity=gravity) if get_initial_state else None
    schedule_out = (
        schedule if schedule is not None else _schedule_from_deck(deck, model)
    ) if get_schedule else None
    nonlinear = get_non_linear_solver(
        model, timestep_strategy=timestep_strategy, use_cpr=use_cpr,
        max_iterations=max_iterations, use_relaxation=use_relaxation,
    )
    return state0, model, schedule_out, nonlinear


initEclipseProblemAD = init_eclipse_problem_ad

__all__ = ["init_eclipse_problem_ad", "initEclipseProblemAD"]
