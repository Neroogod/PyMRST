"""A deck becomes a fluid: reader, unit conversion, dispatcher.

The pieces are tested separately elsewhere - `read_props` against MRST's
`readEclipseDeck`, the assign functions against MRST's `assign*`. This checks
that they compose: a real deck file goes in one end and a usable fluid comes
out the other, with every property the deck's phases require.

The SPE decks exercise three different keyword families between them, which is
the point of using all three rather than one:

- SPE1 and SPE9 are live oil with dry gas (PVTO + PVDG, SWOF + SGOF);
- SPE3 is dead oil with wet gas (PVDO + PVTG) and the three-column
  saturation family (SWFN + SGFN + SOF3).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_adi_fluid
from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)

DATA = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/examples/data")


def build(name: str):
    """Read, convert and dispatch one deck, failing on any warning."""
    path = DATA / name / f"BENCH_{name}.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    deck = convert_deck_units(read_eclipse_deck(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fluid = init_deck_adi_fluid(deck)
    return fluid, [str(w.message) for w in caught]


@pytest.fixture(scope="module")
def spe1():
    return build("SPE1")


@pytest.fixture(scope="module")
def spe3():
    return build("SPE3")


@pytest.fixture(scope="module")
def spe9():
    return build("SPE9")


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_no_property_fails_to_assign(deck, request):
    """A warning here means a keyword the deck states was silently dropped."""
    _, warned = request.getfixturevalue(deck)
    assert warned == [], warned


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_the_fluid_has_every_phase_property(deck, request):
    fluid, _ = request.getfixturevalue(deck)
    for field in ("bW", "muW", "bO", "muO", "bG", "muG",
                  "krW", "krG", "krOW", "krOG",
                  "rhoWS", "rhoOS", "rhoGS", "pvMultR"):
        assert field in fluid, f"{deck} is missing {field}"


@pytest.mark.parametrize("deck", ["spe1", "spe9"])
def test_live_oil_decks_carry_the_saturation_envelope(deck, request):
    """PVTO gives Rs against pressure and its inverse."""
    fluid, _ = request.getfixturevalue(deck)
    assert "rsSat" in fluid and "pb" in fluid
    assert "rvSat" not in fluid, "dry gas has no vaporised oil"


def test_the_wet_gas_deck_carries_the_vaporised_oil_envelope(spe3):
    fluid, _ = spe3
    assert "rvSat" in fluid
    assert "rsSat" not in fluid, "SPE3's oil is dead"


@pytest.mark.parametrize("deck", ["spe1", "spe9"])
def test_live_oil_properties_take_pressure_saturation_and_a_flag(deck, request):
    fluid, _ = request.getfixturevalue(deck)
    p = np.array([2.0e7, 2.5e7])
    rs = fluid["rsSat"](p)
    b = fluid["bO"](p, rs, np.array([True, True]))
    assert b.shape == (2,) and np.all(b > 0), b


def test_dead_oil_properties_take_pressure_alone(spe3):
    """And reject the live-oil call signature rather than returning NaN."""
    fluid, _ = spe3
    p = np.array([2.0e7, 2.5e7])
    b = fluid["bO"](p)
    assert b.shape == (2,) and np.all(b > 0), b
    with pytest.raises(TypeError):
        fluid["bO"](p, np.zeros(2), np.ones(2, dtype=bool))


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_relative_permeability_is_bounded_on_its_range(deck, request):
    fluid, _ = request.getfixturevalue(deck)
    saturations = np.linspace(0.0, 1.0, 41)
    for field in ("krW", "krG"):
        values = fluid[field](saturations)
        assert np.all(values >= 0.0), f"{deck} {field} goes negative"
        assert np.all(values <= 1.0 + 1e-12), f"{deck} {field} exceeds 1"


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_relative_permeability_stays_bounded_outside_zero_to_one(deck, request):
    """A Newton step proposes saturations outside [0, 1]; flat extension holds."""
    fluid, _ = request.getfixturevalue(deck)
    for field in ("krW", "krG"):
        values = fluid[field](np.array([-0.5, 1.5]))
        assert np.all(np.isfinite(values)), f"{deck} {field}"
        assert np.all(values >= 0.0), f"{deck} {field} goes negative"


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_surface_densities_are_positive(deck, request):
    fluid, _ = request.getfixturevalue(deck)
    for field in ("rhoWS", "rhoOS", "rhoGS"):
        assert np.all(np.atleast_1d(fluid[field]) > 0), f"{deck} {field}"


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_water_shrinkage_rises_with_pressure(deck, request):
    fluid, _ = request.getfixturevalue(deck)
    p = np.array([1.0e7, 2.0e7, 3.0e7])
    assert np.all(np.diff(fluid["bW"](p)) > 0), fluid["bW"](p)


@pytest.mark.parametrize("deck", ["spe1", "spe3", "spe9"])
def test_a_single_region_fluid_stores_bare_callables(deck, request):
    """None of the SPE decks partition, so nothing should be a one-item list."""
    fluid, _ = request.getfixturevalue(deck)
    listed = [k for k, v in fluid.items() if isinstance(v, list)]
    assert listed == [], listed


def test_the_saturation_end_points_are_recorded(spe1):
    fluid, _ = spe1
    assert set(fluid["krPts"]) == {"w", "ow", "g", "og"}
    assert np.isclose(fluid["krPts"]["w"][0][0], 0.12), "SPE1's connate water"


def test_spe3_uses_the_three_column_family(spe3):
    """SWFN + SGFN + SOF3 produce the same fields as SWOF + SGOF."""
    fluid, _ = spe3
    assert set(fluid["krPts"]) == {"w", "g", "ow", "og"}
    assert callable(fluid["krOW"]) and callable(fluid["krOG"])


def test_the_dispatcher_and_the_bespoke_reader_agree_on_spe1():
    """A cross-check against the reader the examples currently use."""
    path = DATA / "SPE1" / "BENCH_SPE1.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    from pymrst.modules.deckformat.spe1.read_spe1_deck import read_spe1_deck

    fluid, _ = build("SPE1")
    bespoke = read_spe1_deck(path)
    # Each density is a one-element array (one PVT region), so flatten before
    # comparing against the bespoke reader's plain triple.
    densities = np.array([fluid["rhoWS"], fluid["rhoOS"],
                          fluid["rhoGS"]]).ravel()
    assert np.allclose(densities, np.asarray(bespoke["rhoS"]).ravel())
    sw = np.array([0.12, 0.3, 0.5, 0.9])
    assert np.allclose(fluid["krW"](sw),
                       np.interp(sw, bespoke["swof"][:, 0], bespoke["swof"][:, 1]))
