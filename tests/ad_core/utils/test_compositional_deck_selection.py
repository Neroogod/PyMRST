"""Selecting a compositional model from a COMPS deck (MRST
`selectModelFromDeck`).

This branch used to raise, on the stated grounds that `initDeckEOSModel` and
the `GenericNaturalVariablesModel` deck path were unported. Both had in fact
been ported since - the refusal outlived its reason, which is the same failure
as a stale comment, only louder: it turned working machinery into an error
message.

Two decisions in the port worth pinning:

* **natural variables, not overall composition.** Both formulations solve the
  same system and differ in their unknowns. Natural variables carries
  saturations and phase compositions, so it can hold a cell exactly at a phase
  boundary - where a deck-initialised reservoir usually starts. Overall
  composition would flash such a cell every iteration to find out what is in
  it. MRST picks natural variables for deck models;
* **the EOS is carried on the model.** The flash and the property evaluations
  both need it, and rebuilding it from the deck each time is the same work
  done twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.select_model_from_deck import (
    deck_model_choice, select_model_from_deck_ad,
)
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY, MILLI


def _inputs():
    g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    return g, make_rock(g, 100*MILLI*DARCY, 0.25)


def _deck(**props):
    base = {"CNAMES": ["C1", "C4", "C10"],
            "TCRIT": [190.6, 425.2, 617.7],
            "PCRIT": [46.0e5, 37.97e5, 21.1e5],
            "ZCRIT": [0.286, 0.274, 0.249],
            "ACF": [0.008, 0.193, 0.49],
            "MW": [16.04e-3, 58.12e-3, 142.28e-3],
            "EOS": "PR"}
    base.update(props)
    return {"RUNSPEC": {"COMPS": 3, "WATER": True, "OIL": True, "GAS": True},
            "PROPS": base}


# -- routing -----------------------------------------------------------------

def test_a_comps_deck_routes_to_the_compositional_branch():
    choice = deck_model_choice(_deck())
    assert choice["kind"] == "compositional"
    assert choice["water"] is True


def test_the_water_flag_follows_the_runspec():
    deck = _deck()
    deck["RUNSPEC"]["WATER"] = False
    assert deck_model_choice(deck)["water"] is False


def test_a_black_oil_deck_is_unaffected():
    """The branch that already worked must not have moved."""
    deck = {"RUNSPEC": {"BLACKOIL": True, "WATER": True, "OIL": True,
                        "GAS": True},
            "PROPS": {"PVTO": [[0.0]], "PVTG": [[0.0]]}}
    choice = deck_model_choice(deck)
    assert choice["kind"] == "blackoil"
    assert choice["disgas"] and choice["vapoil"]


# -- construction ------------------------------------------------------------

def test_it_builds_a_natural_variables_model():
    """Not overall composition: natural variables can hold a cell exactly at a
    phase boundary, which is where a deck-initialised reservoir starts."""
    from pymrst.modules.compositional.natural_variables import (
        NaturalVariablesCompositionalModel,
    )
    g, rock = _inputs()
    model = select_model_from_deck_ad(g, rock, None, _deck(RTEMP=350.0))
    assert isinstance(model, NaturalVariablesCompositionalModel)


def test_the_decks_components_reach_the_mixture():
    g, rock = _inputs()
    model = select_model_from_deck_ad(g, rock, None, _deck(RTEMP=350.0))
    assert tuple(model.mix.names) == ("C1", "C4", "C10")


def test_the_eos_is_carried_on_the_model():
    """Rebuilding it from the deck on every flash is the same work twice."""
    g, rock = _inputs()
    model = select_model_from_deck_ad(g, rock, None, _deck(RTEMP=350.0))
    assert model.EOSModel is not None
    assert model.EOSModel.CompositionalMixture is model.mix


def test_the_deck_temperature_is_used():
    g, rock = _inputs()
    model = select_model_from_deck_ad(g, rock, None, _deck(RTEMP=350.0))
    np.testing.assert_allclose(np.ravel(model.T), 350.0)


def test_a_depth_table_takes_its_shallowest_temperature():
    """`TEMPVD` is a depth table. Averaging it would invent a number the deck
    does not contain; a depth-varying temperature needs a thermal model to
    mean anything."""
    g, rock = _inputs()
    deck = _deck(TEMPVD=[[1000.0, 340.0], [2000.0, 380.0]])
    model = select_model_from_deck_ad(g, rock, None, deck)
    np.testing.assert_allclose(np.ravel(model.T), 340.0)


def test_a_deck_without_a_temperature_falls_back_to_the_eos_reference():
    g, rock = _inputs()
    model = select_model_from_deck_ad(g, rock, None, _deck())
    np.testing.assert_allclose(np.ravel(model.T), 273.15 + 15.0)


def test_the_deck_is_recorded_on_the_model():
    g, rock = _inputs()
    deck = _deck(RTEMP=350.0)
    model = select_model_from_deck_ad(g, rock, None, deck)
    assert model.input_data is deck
    assert model.rock is rock


def test_a_deck_without_component_data_is_refused():
    """A COMPS deck missing its EOS section cannot produce a mixture."""
    g, rock = _inputs()
    deck = {"RUNSPEC": {"COMPS": 3, "OIL": True, "GAS": True}, "PROPS": {}}
    with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
        select_model_from_deck_ad(g, rock, None, deck)


def test_an_empty_component_list_is_refused():
    """The reachable case, and the one a `mixture is None` check misses.

    `CNAMES` present but empty builds an EOS with zero components quite
    happily; the model that comes out flashes an empty component set forever
    without ever failing. Measured: `init_deck_eos_model` returns a mixture
    here, not `None`.
    """
    g, rock = _inputs()
    deck = {"RUNSPEC": {"COMPS": 3, "OIL": True, "GAS": True},
            "PROPS": {"CNAMES": [], "TCRIT": [], "PCRIT": [], "ZCRIT": [],
                      "ACF": [], "MW": [], "EOS": "PR"}}
    with pytest.raises(ValueError, match="no components"):
        select_model_from_deck_ad(g, rock, None, deck)


# -- what is still refused ---------------------------------------------------

def test_eor_decks_are_still_refused_and_say_which_module():
    """`ad-eor` is a different MRST module, not a gap in ad_core."""
    deck = {"RUNSPEC": {"POLYMER": True, "WATER": True, "OIL": True},
            "PROPS": {}}
    with pytest.raises(NotImplementedError, match="ad-eor"):
        deck_model_choice(deck)
