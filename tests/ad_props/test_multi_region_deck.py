"""A deck with several property regions, read end to end.

None of the SPE benchmarks partitions its field, so every path that depends on
TABDIMS, SATNUM and PVTNUM is exercised elsewhere only through hand-built
`Regions` objects. This drives the same machinery from deck text, which is the
only way to catch the joins between the stages:

- TABDIMS must reach the PROPS reader, or it reads one table and drops the
  rest without complaint;
- the REGIONS section must reach `handle_regions`, converting the deck's
  1-based numbering to 0-based exactly once;
- the assign functions must produce one closure per region and the dispatcher
  must *not* unwrap those lists, since there is more than one.

The deck below is small enough to reason about and states two saturation
regions and two PVT regions, with deliberately different curves so a cell
reading the wrong region's table gives a visibly wrong answer.
"""

import numpy as np
import pytest

from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_adi_fluid
from pymrst.modules.ad_props.utils.handle_regions import handle_regions
from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck_text,
)

DECK = """\
RUNSPEC

DIMENS
 4 1 1 /

OIL
WATER
GAS
METRIC

TABDIMS
 2 2 /

PROPS

-- region 1 water is twice as mobile as region 2
SWOF
 0.10 0.00 1.00 0.0
 0.50 0.40 0.40 0.0
 0.90 1.00 0.00 0.0 /
 0.10 0.00 1.00 0.0
 0.50 0.20 0.40 0.0
 0.90 1.00 0.00 0.0 /

SGOF
 0.00 0.00 1.00 0.0
 0.40 0.30 0.35 0.0
 0.90 1.00 0.00 0.0 /
 0.00 0.00 1.00 0.0
 0.40 0.15 0.35 0.0
 0.90 1.00 0.00 0.0 /

DENSITY
 800 1000 1.0 /
 850 1010 1.2 /

PVTW
 200 1.0 4.0E-5 0.5 0.0 /
 200 1.0 4.0E-5 1.0 0.0 /

PVDO
 100 1.20 1.0
 300 1.10 1.2 /
 100 2.40 1.0
 300 2.20 1.2 /

PVDG
 100 0.010 0.014
 300 0.005 0.020 /
 100 0.020 0.014
 300 0.010 0.020 /

ROCK
 200 5.0E-5 /
 200 5.0E-5 /

REGIONS

SATNUM
 1 1 2 2 /

PVTNUM
 1 2 1 2 /

END
"""


@pytest.fixture(scope="module")
def deck():
    return convert_deck_units(read_eclipse_deck_text(DECK))


@pytest.fixture(scope="module")
def fluid(deck):
    return init_deck_adi_fluid(deck, ignore_errors=False)


# --------------------------------------------------------------- the reader

def test_tabdims_reaches_the_props_reader(deck):
    """Without it only the first table of each keyword is read."""
    assert len(deck["PROPS"]["SWOF"]) == 2
    assert len(deck["PROPS"]["PVDO"]) == 2
    assert deck["PROPS"]["DENSITY"].shape == (2, 3)


def test_the_second_table_is_the_one_the_deck_states(deck):
    second = deck["PROPS"]["SWOF"][1]
    assert np.isclose(second[1, 1], 0.20), "region 2's krW at Sw = 0.5"


def test_the_regions_section_is_read(deck):
    assert np.array_equal(deck["REGIONS"]["SATNUM"], [1, 1, 2, 2])
    assert np.array_equal(deck["REGIONS"]["PVTNUM"], [1, 2, 1, 2])


# -------------------------------------------------------------- the regions

def test_region_numbers_become_zero_based_exactly_once(deck):
    reg = handle_regions(deck)
    assert np.array_equal(reg.SATNUM, [0, 0, 1, 1]), reg.SATNUM
    assert np.array_equal(reg.PVTNUM, [0, 1, 0, 1]), reg.PVTNUM


def test_the_partitions_are_independent(deck):
    """SATNUM and PVTNUM split the same four cells differently."""
    reg = handle_regions(deck)
    assert not np.array_equal(reg.SATNUM, reg.PVTNUM)
    assert reg.count("SAT") == 2 and reg.count("PVT") == 2


def test_table_counts_come_from_tabdims(deck):
    reg = handle_regions(deck)
    assert reg.sat == 2 and reg.pvt == 2


# --------------------------------------------------------------- the fluid

def test_a_multi_region_fluid_keeps_one_closure_per_region(fluid):
    """The dispatcher unwraps one-element lists; these must survive."""
    for name in ("krW", "krG", "bO", "muO", "bW"):
        assert isinstance(fluid[name], list), f"{name} was unwrapped"
        assert len(fluid[name]) == 2, name


def test_each_region_reads_its_own_saturation_table(fluid):
    assert np.isclose(fluid["krW"][0](0.5), 0.40), "region 1"
    assert np.isclose(fluid["krW"][1](0.5), 0.20), "region 2"


def test_each_region_reads_its_own_pvt_table(fluid):
    """Region 2's oil FVF is twice region 1's, so b is half."""
    assert np.isclose(fluid["bO"][0](2.0e7) / fluid["bO"][1](2.0e7), 2.0)


def test_water_viscosity_differs_between_pvt_regions(fluid):
    first = fluid["muW"][0](2.0e7)
    second = fluid["muW"][1](2.0e7)
    assert np.isclose(np.ravel(second)[0] / np.ravel(first)[0], 2.0)


def test_surface_densities_are_one_row_per_region(fluid):
    assert np.allclose(fluid["rhoOS"], [800.0, 850.0])
    assert np.allclose(fluid["rhoWS"], [1000.0, 1010.0])


def test_end_points_are_one_row_per_saturation_region(fluid):
    assert fluid["krPts"]["w"].shape == (2, 4)
    assert fluid["krPts"]["g"].shape == (2, 4)


def test_the_gas_table_used_each_regions_own_connate_water(fluid):
    """Both regions state Swcon = 0.10, so both krOG curves shift equally."""
    assert np.allclose(fluid["krPts"]["w"][:, 0], 0.10)
    assert np.isclose(fluid["krOG"][0](1.0 - 0.40 - 0.10), 0.35)


def test_nothing_failed_to_assign(deck):
    """Strict mode: any keyword that could not be assigned raises."""
    init_deck_adi_fluid(deck, ignore_errors=False)


def test_a_region_count_mismatch_is_caught_while_reading():
    """TABDIMS claiming three PVT tables where the deck states two.

    The reader runs past the keyword's records into the next keyword, so the
    failure surfaces here rather than at assignment. The message must name the
    keyword and TABDIMS - the bare "could not convert string to float: 'PVTW'"
    it would otherwise raise sends the reader to the wrong keyword entirely.
    """
    text = DECK.replace("TABDIMS\n 2 2 /", "TABDIMS\n 2 3 /")
    with pytest.raises(ValueError) as caught:
        read_eclipse_deck_text(text)
    message = str(caught.value)
    assert "DENSITY" in message and "TABDIMS" in message, message
