"""The empty deck structure (MRST ``deckinput/initializeDeck.m``).

Every section reader is handed a deck and returns it enriched, so the readers
never have to test whether a section exists yet. The defaults here are the
ones ECLIPSE assumes when a deck stays silent - notably the 1983-01-01 start
date and the all-false SATOPTS - so a deck that omits a keyword and a deck
that states its default parse to the same structure.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

import numpy as np

# E100's default start date, as in initializeDeck.m.
DEFAULT_START = date(1983, 1, 1)

SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS",
            "SOLUTION", "SUMMARY", "SCHEDULE")


@dataclass
class Schedule:
    """Well controls and the time steps that use them."""

    control: List[Dict[str, Any]] = field(default_factory=list)
    step: Dict[str, List] = field(
        default_factory=lambda: {"control": [], "val": []})


def init_runspec() -> Dict[str, Any]:
    return {
        "cartDims": np.array([-1, -1, -1], dtype=np.int64),
        "START": DEFAULT_START,
        "DUALPORO": False,
        "SATOPTS": {"DIRECT": False, "HYSTER": False, "IRREVERS": False},
    }


def initialize_deck() -> Dict[str, Any]:
    """Return a deck with every section present and empty."""
    deck: Dict[str, Any] = {name: {} for name in SECTIONS}
    deck["RUNSPEC"] = init_runspec()
    deck["SCHEDULE"] = Schedule()
    # Keywords a reader recognised as a keyword but does not act on, kept per
    # section so a caller can see what a deck asked for and did not get.
    deck["UnhandledKeywords"] = {name: [] for name in SECTIONS}
    return deck


initializeDeck = initialize_deck


if __name__ == "__main__":

    def test_every_section_is_present() -> None:
        deck = initialize_deck()
        for section in SECTIONS:
            assert section in deck, section
        print("ok test_every_section_is_present")

    def test_runspec_carries_the_eclipse_defaults() -> None:
        runspec = initialize_deck()["RUNSPEC"]
        assert runspec["START"] == date(1983, 1, 1), runspec["START"]
        assert runspec["DUALPORO"] is False
        assert not any(runspec["SATOPTS"].values()), runspec["SATOPTS"]
        print("ok test_runspec_carries_the_eclipse_defaults")

    def test_cart_dims_start_undefined() -> None:
        """-1 marks "not read yet"; 0 would be a legitimate-looking size."""
        assert np.all(initialize_deck()["RUNSPEC"]["cartDims"] == -1)
        print("ok test_cart_dims_start_undefined")

    def test_schedule_starts_empty() -> None:
        schedule = initialize_deck()["SCHEDULE"]
        assert schedule.control == [] and schedule.step["val"] == []
        print("ok test_schedule_starts_empty")

    def test_unhandled_keywords_are_tracked_per_section() -> None:
        assert set(initialize_deck()["UnhandledKeywords"]) == set(SECTIONS)
        print("ok test_unhandled_keywords_are_tracked_per_section")

    def test_decks_do_not_share_mutable_state() -> None:
        first, second = initialize_deck(), initialize_deck()
        first["PROPS"]["SWOF"] = "x"
        first["UnhandledKeywords"]["GRID"].append("x")
        assert second["PROPS"] == {} and second["UnhandledKeywords"]["GRID"] == []
        print("ok test_decks_do_not_share_mutable_state")

    test_every_section_is_present()
    test_runspec_carries_the_eclipse_defaults()
    test_cart_dims_start_undefined()
    test_schedule_starts_empty()
    test_unhandled_keywords_are_tracked_per_section()
    test_decks_do_not_share_mutable_state()
    print("\nAll tests passed!")
