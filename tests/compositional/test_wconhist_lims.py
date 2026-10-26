"""Regression tests for WCONHIST limits (facility._prod_lims).

MRST's ``process_wconhist`` (processWells.m) initializes every rate limit to
``-inf`` and the bhp limit to ``1*atm``, then overwrites *only* the single
field matching the record's own declared control type. A RESV_HISTORY well
(the deck's ``type == 'resv'``) has no matching case in MRST's switch block,
so its lims stay entirely at the defaults.

A prior bug here copied every WCONHIST rate column (orat/wrat/grat/lrat) into
``lims`` regardless of the record's actual control type, and defaulted bhp
via a stray ``resv_history`` key. For a RESV_HISTORY well this planted finite
"limits" equal to the well's own unrelated historical rates, which spuriously
tripped ``apply_well_limits`` control switching mid-simulation (observed on
Norne's D-1H well: control silently flipped from ``resv_history`` to a bare
``orat`` equation, leaving BHP essentially unconstrained).
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.facility import _prod_lims
from pymrst.utils.units import ATM


def _wconhist_record(mode: str, orat: float, wrat: float, grat: float, bhp: float) -> list:
    # Positional WCONHIST record: name, status, type, orat, wrat, grat, lrat,
    # vrat, wct/gor placeholder, bhp (10 columns -> non-compact / "full").
    return ["D-1H", "OPEN", mode, orat, wrat, grat, 0.0, 0.0, 0.0, bhp]


def test_resv_history_record_leaves_all_rate_limits_at_mrst_defaults():
    record = _wconhist_record("resv", orat=100.0, wrat=5.0, grat=2000.0, bhp=250.0)
    lims = _prod_lims(record, keyword="WCONHIST")
    assert lims["orat"] == -np.inf
    assert lims["wrat"] == -np.inf
    assert lims["grat"] == -np.inf
    assert lims["lrat"] == -np.inf
    assert lims["vrat"] == -np.inf
    assert lims["bhp"] == ATM
    assert "resv_history" not in lims


def test_orat_record_sets_only_the_orat_limit():
    record = _wconhist_record("orat", orat=100.0, wrat=5.0, grat=2000.0, bhp=250.0)
    lims = _prod_lims(record, keyword="WCONHIST")
    assert lims["orat"] == -100.0
    assert lims["wrat"] == -np.inf
    assert lims["grat"] == -np.inf
    assert lims["lrat"] == -np.inf
    assert lims["bhp"] == ATM


def test_bhp_record_sets_only_the_bhp_limit_unnegated():
    record = _wconhist_record("bhp", orat=100.0, wrat=5.0, grat=2000.0, bhp=250.0)
    lims = _prod_lims(record, keyword="WCONHIST")
    assert lims["bhp"] == 250.0
    assert lims["orat"] == -np.inf
    assert lims["wrat"] == -np.inf
    assert lims["grat"] == -np.inf


def test_lrat_record_sums_only_oil_and_water():
    record = _wconhist_record("lrat", orat=100.0, wrat=5.0, grat=2000.0, bhp=250.0)
    lims = _prod_lims(record, keyword="WCONHIST")
    assert lims["lrat"] == -105.0
    assert lims["orat"] == -np.inf
    assert lims["grat"] == -np.inf
    assert lims["bhp"] == ATM


def test_resv_history_with_zero_orat_column_does_not_trip_a_rate_limit():
    # D-1H's actual historical record: a nonzero orat column on a resv-typed
    # well. Before the fix this produced lims['orat'] far below the well's
    # real production rate, which apply_well_limits read as an already
    # violated limit and used to switch the well off its resv_history target.
    record = _wconhist_record("resv", orat=5432.5, wrat=0.0, grat=651533.6, bhp=0.0)
    lims = _prod_lims(record, keyword="WCONHIST")
    assert lims["orat"] == -np.inf
    assert lims["grat"] == -np.inf
    assert lims["bhp"] == ATM
