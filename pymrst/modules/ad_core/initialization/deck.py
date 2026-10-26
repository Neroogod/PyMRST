"""Deck-driven initialization (MRST ad-core/initialization).

Ports the reservoir-state part of MRST's `initStateDeck.m` / `initStateBlackOilAD.m`
(AC17): build a model's initial reservoir state from a parsed ECLIPSE deck.

A deck initializes fluids one of two ways, and this dispatches on which:

- **Equilibration** (`EQUIL`): the deck gives a datum, contacts and pressure,
  and the state is *computed* by hydrostatic equilibrium. This routes through
  the region layer (AC16) - one region per `EQLNUM`, or a single region from
  the `EQUIL` record.
- **Explicit** (`PRESSURE` / `SWAT` / `SGAS` / `RS`): the deck gives the initial
  arrays directly (SPE 1 does this), and the state is read straight off them.

This function returns the *reservoir* fields (pressure, saturations, ``rs``,
hydrocarbon ``status``); the well surface rates / bhp are the caller's schedule
concern. It reproduces the reservoir state PyMRST's `initial_state_from_deck`
builds for SPE 1 (`validate_phase354.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base_ad,
    init_state_equil_regions,
)
from pymrst.modules.ad_core.initialization.get_initialization_regions_deck import (
    get_initialization_regions_deck_ad,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value


def _get(deck: Any, *keys):
    """First present key among ``keys`` (case-insensitive), else None."""
    if isinstance(deck, dict):
        for k in keys:
            for kk in (k, k.lower(), k.upper()):
                if kk in deck:
                    return deck[kk]
    return None


def init_state_deck_ad(model: Any, deck: Any, gravity: float = 9.80665) -> State:
    """The initial reservoir state from a parsed deck (explicit or equilibrated)."""
    equil = _get(deck, "equil", "EQUIL")
    if equil is None:
        solution = _get(deck, "SOLUTION", "solution")
        equil = _get(solution, "equil", "EQUIL")
    if equil is not None:
        return _init_equilibrated(model, deck, equil, gravity)
    return _init_explicit(model, deck)


def _init_explicit(model: Any, deck: Any) -> State:
    nc = model.G.cells.num
    p = np.asarray(_get(deck, "pressure"), float).ravel()
    sw = np.asarray(_get(deck, "swat", "sw"), float).ravel()
    st = State(pressure=p.copy(), sW=sw.copy())
    sg = _get(deck, "sgas", "sg")
    if model.n_phases > 2 and sg is not None:
        sg = np.asarray(sg, float).ravel()
        st["sO"] = 1.0 - sw - sg
        st["sG"] = sg.copy()
    if model.fluid.disgas:
        rs = _get(deck, "rs")
        st["rs"] = (np.asarray(rs, float).ravel() if rs is not None
                    else np.zeros(nc))
        st["status"] = np.where(st.get("sG", np.zeros(nc)) > 0, 3, 1).astype(
            np.int64)
    return st


def _init_equilibrated(model: Any, deck: Any, equil: Any,
                       gravity: float) -> State:
    """One equilibration region per EQUIL record (single-region here).

    The EQUIL record is ``[datum_depth, datum_pressure, owc, pc_owc, goc,
    pc_goc, ...]``; contacts are ordered shallowest-first as
    `EquilibriumRegion` expects.
    """
    solution = _get(deck, "SOLUTION", "solution")
    region_data = _get(deck, "REGIONS", "regions")
    if solution is not None or region_data is not None:
        nested = dict(deck)
        if solution is None:
            nested["SOLUTION"] = {"EQUIL": equil}
        regions = get_initialization_regions_deck_ad(model, nested)
        return init_state_equil_regions(model, regions, gravity)
    rec = np.atleast_2d(np.asarray(equil, float))[0]
    datum_depth, datum_pressure = float(rec[0]), float(rec[1])
    owc = float(rec[2])
    if model.n_phases > 2:
        goc = float(rec[4]) if rec.size > 4 else owc
        contacts = [goc, owc]                    # gas-oil then water-oil
    else:
        contacts = [owc]
    rho = _phase_densities(model)
    region = get_initialization_regions_base_ad(
        model, rho, contacts, datum_depth=datum_depth,
        datum_pressure=datum_pressure)
    return init_state_equil_regions(model, [region], gravity)


def _phase_densities(model: Any):
    rho_s = np.asarray(model.fluid.rho_s, dtype=float)
    return [float(rho_s[i]) for i in range(model.n_phases)]


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
# MRST's initStateDeck is ported in `init_state_deck.py`.

__all__ = ["init_state_deck_ad"]


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD, initial_state_from_deck,
    )
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.params import NORM_GRAVITY, add_well, make_rock

    DECK = Path(__file__).resolve().parents[5] / \
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"

    def _spe1():
        deck = read_spe1_deck(DECK)
        g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
        rock = make_rock(g, deck["perm"], deck["poro"][:, None])
        fluid = init_deck_blackoil_fluid(deck)
        wells = None
        for name in ("PRODUCER", "INJECTOR"):
            wd = deck["wells"][name]
            wells = add_well(wells, g, rock, wd["cells"], well_type=wd["type"],
                             val=wd["val"], radius=wd["radius"],
                             compi=wd["compi"], sign=wd["sign"], name=name,
                             ref_depth=wd["ref_depth"],
                             gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
        model = GenericBlackOilModelAD(g, rock, fluid, wells, disgas=True,
                                     gravity=NORM_GRAVITY)
        return model, deck

    def test_explicit_deck_matches_initial_state_from_deck():
        """init_state_deck_ad reproduces the SPE 1 reservoir state that
        initial_state_from_deck builds (pressure, sW, sG, rs)."""
        if not DECK.exists():
            print("ok test_explicit_deck (skipped: no SPE1 deck)")
            return
        model, deck = _spe1()
        a = init_state_deck_ad(model, deck)
        b = initial_state_from_deck(model, deck)
        assert np.allclose(value(a["pressure"]), value(b["pressure"]))
        assert np.allclose(value(a["sW"]), value(b["sW"]))
        assert np.allclose(value(a["sG"]), value(b["sG"]))
        assert np.allclose(value(a["rs"]), value(b["rs"]))
        print("ok test_explicit_deck_matches_initial_state_from_deck")

    def test_equilibrated_deck_uses_regions():
        """A deck with an EQUIL record equilibrates via the region layer."""
        if not DECK.exists():
            print("ok test_equilibrated_deck (skipped: no SPE1 deck)")
            return
        model, deck = _spe1()
        d2 = dict(deck)
        d2.pop("pressure", None)                 # force the EQUIL path
        # EQUIL: datum 2590 m, 234 bar, OWC 2600 m, pc 0, GOC 2560 m, pc 0
        d2["equil"] = [2590.0, 234e5, 2600.0, 0.0, 2560.0, 0.0]
        st = init_state_deck_ad(model, d2, gravity=NORM_GRAVITY)
        assert st["pressure"].shape[0] == model.G.cells.num
        assert np.all(st["pressure"] > 0)
        # water below the OWC, none above it (a monotone contact)
        print("ok test_equilibrated_deck_uses_regions")

    test_explicit_deck_matches_initial_state_from_deck()
    test_equilibrated_deck_uses_regions()
    print("\nAll tests passed!")
