"""The general PROPS reader must parse the SPE decks as the bespoke ones do.

PyMRST grew a hand-written reader per benchmark deck (`spe1/read_spe1_deck.py`
and friends) before it had a general one. Those readers are trusted - their
results drive the simulation examples - so they are the reference the general
`deckformat/deckinput` reader has to reproduce before anything is migrated
onto it.

The comparison is deliberately made **up to a per-column constant**. The
bespoke readers return SI values; the general reader returns the deck's own
field units, because unit conversion belongs to `convert_deck_units` rather
than to a parser. Asking only that each column agree up to one constant tests
what is actually in question - did the two readers pull the same numbers out
of the same file - without this test having to restate every unit factor, and
without it passing when a column is scaled inconsistently row to row.
"""

from pathlib import Path

import numpy as np
import pytest

from pymrst.modules.deckformat.deckinput.read_defaulted_record import DeckCursor
from pymrst.modules.deckformat.deckinput.read_props import read_props, TableCounts

DATA = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/examples/data")


def read_general(deck_path: Path) -> dict:
    """Read a deck's PROPS section through the general reader."""
    cursor = DeckCursor.from_text(deck_path.read_text(errors="ignore"))
    while True:
        keyword = cursor.next_keyword()
        if keyword is None:
            raise AssertionError(f"{deck_path.name} has no PROPS section")
        if keyword == "PROPS":
            break
    return read_props(cursor, TableCounts())


def assert_same_up_to_column_scaling(general, reference, what: str) -> None:
    """Each column of ``reference`` must be one constant times ``general``."""
    general = np.atleast_2d(np.asarray(general, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    assert general.shape == reference.shape, (
        f"{what}: shape {general.shape} vs {reference.shape}")

    for column in range(general.shape[1]):
        mine, theirs = general[:, column], reference[:, column]
        nonzero = np.abs(mine) > 0
        if not np.any(nonzero):
            assert np.allclose(theirs, 0.0), f"{what}: column {column}"
            continue
        ratios = theirs[nonzero] / mine[nonzero]
        assert np.allclose(ratios, ratios[0], rtol=1e-9), (
            f"{what}: column {column} is not a single scaling "
            f"(ratios {ratios.min()}..{ratios.max()})")
        assert np.allclose(theirs[~nonzero], 0.0), (
            f"{what}: column {column} has a zero that the reference does not")


def assert_branches_match(packed, branches, what: str) -> None:
    """A packed PVTO/PVTG table must contain the branches the bespoke reader found.

    "Contain", not "equal": the general reader also applies MRST's
    `fill_usat_invlinear`, which extends a branch that states only its bubble
    point with derived undersaturated rows. None of the bespoke readers do
    that, so every bespoke branch is a *prefix* of the corresponding general
    one. SPE9 is where this bites - ten of its eleven PVTO branches are
    saturated-only, so the bespoke reader yields 12 rows where MRST yields 22.
    """
    assert packed.key.size == len(branches), (
        f"{what}: {packed.key.size} branches vs {len(branches)}")

    keys = np.array([key for key, _ in branches], dtype=float)
    assert_same_up_to_column_scaling(packed.key[:, None], keys[:, None],
                                     f"{what} keys")
    for index, (_, rows) in enumerate(branches):
        block = packed.data[packed.pos[index]:packed.pos[index + 1]]
        assert block.shape[0] >= rows.shape[0], (
            f"{what}: branch {index} has {block.shape[0]} rows, fewer than the "
            f"{rows.shape[0]} the bespoke reader found")
        assert_same_up_to_column_scaling(block[:rows.shape[0]], rows,
                                         f"{what} branch {index}")


# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spe1():
    path = DATA / "SPE1" / "BENCH_SPE1.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    from pymrst.modules.deckformat.spe1.read_spe1_deck import read_spe1_deck
    return read_general(path), read_spe1_deck(path)


@pytest.fixture(scope="module")
def spe3():
    path = DATA / "SPE3" / "BENCH_SPE3.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    from pymrst.modules.deckformat.spe3.read_spe3_deck import read_spe3_deck
    return read_general(path), read_spe3_deck(path)


@pytest.fixture(scope="module")
def spe9():
    path = DATA / "SPE9" / "BENCH_SPE9.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    from pymrst.modules.deckformat.spe9.read_spe9_deck import read_spe9_deck
    return read_general(path), read_spe9_deck(path)


@pytest.mark.parametrize("deck", ["spe1", "spe9"])
def test_saturation_tables_match(deck, request):
    general, bespoke = request.getfixturevalue(deck)
    for keyword in ("SWOF", "SGOF"):
        assert_same_up_to_column_scaling(
            general[keyword][0], bespoke[keyword.lower()], f"{deck} {keyword}")


@pytest.mark.parametrize("deck,keyword,field", [
    ("spe1", "PVDG", "pvdg"), ("spe9", "PVDG", "pvdg"), ("spe3", "PVDO", "pvdo"),
])
def test_immiscible_pvt_tables_match(deck, keyword, field, request):
    general, bespoke = request.getfixturevalue(deck)
    assert_same_up_to_column_scaling(general[keyword][0], bespoke[field],
                                     f"{deck} {keyword}")


@pytest.mark.parametrize("deck,keyword,field", [
    ("spe1", "PVTO", "pvto"), ("spe9", "PVTO", "pvto"), ("spe3", "PVTG", "pvtg"),
])
def test_miscible_pvt_tables_match(deck, keyword, field, request):
    """The packed table must hold the same branches, in the same order."""
    general, bespoke = request.getfixturevalue(deck)
    assert_branches_match(general[keyword][0], bespoke[field], f"{deck} {keyword}")


# The single-record keywords are checked against the deck text itself rather
# than against the bespoke readers. A one-row table has one entry per column,
# so "agrees up to a per-column constant" is vacuously true for it - the check
# that works for a 30-row table asserts nothing here. These literals are
# transcribed from the PVTW, ROCK and DENSITY records of each deck.
RECORD_LITERALS = {
    "spe1": {"PVTW": [4014.7, 1.029, 3.13e-6, 0.31, 0.0],
             "ROCK": [14.7, 3.0e-6],
             "DENSITY": [49.1, 64.79, 0.06054]},
    "spe3": {"PVTW": [3550.0, 1.0, 3.0e-6, 0.70, 0.0],
             "ROCK": [3550.0, 4.0e-6],
             "DENSITY": [48.8308, 62.4280, 0.0725]},
    "spe9": {"PVTW": [3600.0, 1.0034, 1.0e-6, 0.96, 0.0],
             "ROCK": [3600.0, 1.0e-6],
             "DENSITY": [44.98, 63.01, 0.0702]},
}


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
@pytest.mark.parametrize("keyword", ["PVTW", "ROCK", "DENSITY"])
def test_single_record_keywords_match_the_deck_text(deck, keyword, request):
    general, _ = request.getfixturevalue(deck)
    expected = RECORD_LITERALS[deck][keyword]
    record = general[keyword][0]
    # A record is as wide as its template, so ROCK is six columns even though
    # these decks state two. The unstated fields default to NaN, which is how
    # MRST distinguishes "not given" from "given as zero".
    assert np.allclose(record[:len(expected)], expected), (
        f"{deck} {keyword}: {record} != {expected}")
    assert np.all(np.isnan(record[len(expected):])), (
        f"{deck} {keyword}: unstated fields must be NaN, got {record}")


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_surface_densities_reach_the_bespoke_readers(deck, request):
    """DENSITY is (oil, water, gas); the bespoke readers store (water, oil, gas).

    Checked as a ratio between the two non-trivially-different densities, which
    is unit-free - so this pins the ordering without restating lb/ft3 -> kg/m3.
    """
    general, bespoke = request.getfixturevalue(deck)
    oil, water, _ = general["DENSITY"][0]
    rho_water, rho_oil, _ = bespoke["rhoS"]
    assert np.isclose(water / oil, rho_water / rho_oil, rtol=1e-9), (
        f"{deck}: density ordering differs between the readers")


# ---------------------------------------------------------------------------
# End to end: reader plus unit conversion against the bespoke readers' SI
# values. The comparisons above test parsing; these test the whole pipeline,
# with no allowance for a scale factor.

@pytest.fixture(scope="module")
def converted():
    """Each deck read through the general path and converted to SI."""
    from pymrst.modules.deckformat.deckinput.convert_deck_units import (
        convert_deck_units,
    )
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
        read_eclipse_deck,
    )
    out = {}
    for name, stem in [("spe1", "SPE1"), ("spe3", "SPE3"), ("spe9", "SPE9")]:
        path = DATA / stem / f"BENCH_{stem}.DATA"
        if not path.exists():
            pytest.skip("MRST example data not available")
        out[name] = convert_deck_units(read_eclipse_deck(path))["PROPS"]
    return out


@pytest.mark.parametrize("deck,keyword,field", [
    ("spe1", "SWOF", "swof"), ("spe1", "SGOF", "sgof"), ("spe1", "PVDG", "pvdg"),
    ("spe9", "SWOF", "swof"), ("spe9", "SGOF", "sgof"), ("spe9", "PVDG", "pvdg"),
    ("spe3", "PVDO", "pvdo"),
])
def test_si_tables_match_exactly(deck, keyword, field, converted, request):
    _, bespoke = request.getfixturevalue(deck)
    assert np.allclose(converted[deck][keyword][0], bespoke[field]), (
        f"{deck} {keyword} differs after unit conversion")


@pytest.mark.parametrize("deck,keyword,field", [
    ("spe1", "PVTO", "pvto"), ("spe9", "PVTO", "pvto"), ("spe3", "PVTG", "pvtg"),
])
def test_si_miscible_tables_match_exactly(deck, keyword, field, converted,
                                          request):
    _, bespoke = request.getfixturevalue(deck)
    table = converted[deck][keyword]
    assert len(table) == 1
    table = table[0]
    for index, (key, rows) in enumerate(bespoke[field]):
        assert np.isclose(table.key[index], key), f"{deck} {keyword} key {index}"
        block = table.data[table.pos[index]:table.pos[index + 1]]
        # A prefix comparison, for the reason given on assert_branches_match:
        # the general reader adds the undersaturated rows MRST derives.
        assert np.allclose(block[:rows.shape[0]], rows), (
            f"{deck} {keyword} branch {index}")


def test_spe9_pvto_gains_the_undersaturated_rows_the_bespoke_reader_lacks(
        converted, spe9):
    """SPE9 is the case that separates the two readers.

    Ten of SPE9's eleven PVTO branches give only a bubble point. MRST derives
    an undersaturated row for each; PyMRST's `spe9/read_spe9_deck.py` does
    not, so it hands the simulator a table whose branches end at their bubble
    point. This records the difference rather than accommodating it.
    """
    _, bespoke = spe9
    table = converted["spe9"]["PVTO"]
    assert len(table) == 1
    table = table[0]
    assert table.data.shape[0] == 22, table.data.shape
    assert sum(rows.shape[0] for _, rows in bespoke["pvto"]) == 12
    counts = np.diff(table.pos)
    assert np.all(counts == 2), counts


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_si_pvtw_and_rock_match_exactly(deck, converted, request):
    _, bespoke = request.getfixturevalue(deck)
    pvtw, rock = bespoke["pvtw"], bespoke["rock_cp"]
    assert np.allclose(converted[deck]["PVTW"][0],
                       [pvtw["p_ref"], pvtw["b_ref"], pvtw["c"],
                        pvtw["mu_ref"], pvtw["viscosibility"]]), f"{deck} PVTW"
    assert np.allclose(converted[deck]["ROCK"][0][:2],
                       [rock["p_ref"], rock["c"]]), f"{deck} ROCK"


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_si_densities_match_exactly(deck, converted, request):
    _, bespoke = request.getfixturevalue(deck)
    oil, water, gas = converted[deck]["DENSITY"][0]
    assert np.allclose([water, oil, gas], bespoke["rhoS"]), f"{deck} DENSITY"


def test_spe1_shapes_are_the_documented_ones(spe1):
    """A guard on the comparison itself: these tables are not empty."""
    general, _ = spe1
    assert general["SWOF"][0].shape == (14, 4)
    assert general["SGOF"][0].shape == (15, 4)
    assert general["PVTO"][0].key.size == 9
    assert general["PVTO"][0].data.shape == (26, 3)


def test_spe3_uses_the_three_column_saturation_family(spe3):
    """SPE3 is SWFN/SGFN/SOF3, which exercises a different reader width."""
    general, _ = spe3
    for keyword in ("SWFN", "SGFN", "SOF3"):
        assert general[keyword][0].shape[1] == 3, keyword
