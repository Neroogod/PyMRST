"""Build the property-region partition (MRST ``ad-props/utils/handleRegions.m``).

Reads `deck.REGIONS` and produces, for each family of regions, the per-cell
region number and the per-region cell lists that `get_reg_map` consumes.

Five families are handled, matching MRST: PVT (`PVTNUM`), saturation
(`SATNUM`), imbibition (`IMBNUM`), rock compaction (`ROCKNUM`) and surfactant
(`SURFNUM`). When a family has a single region MRST stores an empty number
array and a `':'` index; PyMRST stores `None` and `ALL`, which read the same
way at the call site and let a property skip the region loop entirely.

Region numbers are 0-based here, converted from the deck's 1-based values on
the way in.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import ALL, RegionIndex


@dataclass
class Regions:
    """The region partition of a deck, one family per pair of fields."""

    PVTNUM: Optional[np.ndarray] = None
    PVTINX: Any = ALL
    SATNUM: Optional[np.ndarray] = None
    SATINX: Any = ALL
    IMBNUM: Optional[np.ndarray] = None
    IMBINX: Any = ALL
    ROCKNUM: Optional[np.ndarray] = None
    ROCKINX: Any = ALL
    SURFNUM: Optional[np.ndarray] = None
    SURFINX: Any = ALL

    # Fields below are not set by handleRegions. MRST's `reg` is a plain
    # struct that initDeckADIFluid extends before handing it to the assign
    # functions, so they live here with defaults that make a bare Regions
    # usable on its own - which is what the assign functions' own tests need.
    #
    # `pvt` and `sat` come from TABDIMS and are the number of tables each
    # keyword owns. They are not the same as count("PVT"): a deck can declare
    # more tables than its REGIONS section partitions cells into.
    pvt: int = 1
    sat: int = 1
    # Sample pressures for tabulating a property instead of evaluating it.
    # Empty means "evaluate the closed form directly".
    prange: Optional[np.ndarray] = None
    interp1d: Any = None
    interp1d_uniform: Any = None
    # initDeckADIFluid's defaults, and they differ by phase: oil measures each
    # undersaturated branch from its own bubble point ('parallel'), gas slides
    # the abscissa between neighbouring saturated points ('linshift').
    pvt_method_oil: str = "parallel"
    pvt_method_gas: str = "linshift"
    optimize: bool = False
    #: How many sample pressures `initDeckADIFluid` should build `prange` from.
    #: MRST's `resamplePoly`, and 0 means "none", because `linspace(a, b, 0)`
    #: is empty - which is how the default path ends up evaluating directly.
    resample_poly: int = 0

    def interp(self, x, values, xi):
        """Interpolate, using the deck's interpolator or MRST's default.

        MRST defaults `reg.interp1d` to `interpTable`; a caller overrides it to
        swap in a MEX or a smooth interpolator. Routing every property through
        here means an override reaches all of them.
        """
        from pymrst.utils.interp_table import interp_table
        return (self.interp1d or interp_table)(x, values, xi)

    def interp_uniform(self, x, values, xi):
        """As `interp`, for tables sampled on a uniform grid."""
        from pymrst.utils.interp_table import interp_table
        return (self.interp1d_uniform or self.interp1d or interp_table)(
            x, values, xi)

    def count(self, family: str) -> int:
        """How many regions this family has; 1 when it is not partitioned."""
        index = getattr(self, f"{family}INX")
        return 1 if index is ALL or isinstance(index, slice) else len(index)


def _active(deck, grid) -> Any:
    """The active-cell selector: the grid's index map, ACTNUM, or everything."""
    if grid is not None and getattr(grid.cells, "indexMap", None) is not None:
        return np.asarray(grid.cells.indexMap, dtype=np.int64)
    actnum = _section(deck, "GRID").get("ACTNUM")
    if actnum is not None:
        return np.flatnonzero(np.asarray(actnum).ravel())
    return ALL


def _section(deck, name: str) -> dict:
    if deck is None:
        return {}
    section = deck.get(name) if hasattr(deck, "get") else getattr(deck, name, None)
    if section is None:
        return {}
    return section if hasattr(section, "get") else vars(section)


def _numbers(regions: dict, key: str, active) -> Optional[np.ndarray]:
    """Per-cell region numbers for one keyword, 0-based and active-only."""
    raw = regions.get(key)
    if raw is None:
        return None
    values = np.asarray(raw).ravel()
    values = values if active is ALL else values[active]
    return values.astype(np.int64) - 1              # deck is 1-based


def _split(numbers: Optional[np.ndarray], count: int) -> List[RegionIndex]:
    return [np.flatnonzero(numbers == region) for region in range(count)]


def handle_regions(deck, grid=None) -> Regions:
    """Return the `Regions` partition described by ``deck``."""
    active = _active(deck, grid)
    regions = _section(deck, "REGIONS")
    out = Regions()

    pvt = _numbers(regions, "PVTNUM", active)
    if pvt is not None and pvt.max(initial=-1) + 1 > 1:
        out.PVTNUM, out.PVTINX = pvt, _split(pvt, int(pvt.max()) + 1)

    sat = _numbers(regions, "SATNUM", active)
    surf = _numbers(regions, "SURFNUM", active)
    if surf is not None and sat is None:
        raise ValueError("SATNUM is required when SURFNUM is used")
    if sat is not None and surf is not None:
        # surfactant shares the saturation-function numbering, so both are
        # split over the larger of the two counts
        count = int(max(sat.max(), surf.max())) + 1
        out.SATNUM, out.SATINX = sat, _split(sat, count)
        out.SURFNUM, out.SURFINX = surf, _split(surf, count)
    elif sat is not None and int(sat.max()) + 1 > 1:
        out.SATNUM, out.SATINX = sat, _split(sat, int(sat.max()) + 1)

    imb = _numbers(regions, "IMBNUM", active)
    if imb is not None:
        # MRST halves the IMBNUM count: imbibition tables are stored after the
        # drainage ones in the same table list, so there are twice as many
        # numbers as there are imbibition regions.
        count = (int(imb.max()) + 1) // 2
        if count > 1:
            out.IMBNUM, out.IMBINX = imb, _split(imb, count)

    rock = _numbers(regions, "ROCKNUM", active)
    if rock is not None and rock.max(initial=-1) + 1 > 1:
        out.ROCKNUM, out.ROCKINX = rock, _split(rock, int(rock.max()) + 1)

    # TABDIMS says how many tables each keyword carries. MRST falls back to
    # the partition size and warns; a deck without TABDIMS has one table.
    tabdims = _section(deck, "RUNSPEC").get("TABDIMS")
    if tabdims is not None and len(tabdims) >= 2:
        out.sat, out.pvt = int(tabdims[0]), int(tabdims[1])
    else:
        out.sat, out.pvt = out.count("SAT"), out.count("PVT")
    return out


handleRegions = handle_regions


if __name__ == "__main__":

    def _deck(**regions):
        return {"REGIONS": regions, "GRID": {}}

    def test_no_regions_section_gives_one_region_everywhere() -> None:
        reg = handle_regions(_deck())
        for family in ("PVT", "SAT", "IMB", "ROCK", "SURF"):
            assert getattr(reg, f"{family}NUM") is None, family
            assert reg.count(family) == 1, family
        print("ok test_no_regions_section_gives_one_region_everywhere")

    def test_a_single_region_is_not_partitioned() -> None:
        """One region is stored as 'no partition', so callers skip the loop."""
        reg = handle_regions(_deck(PVTNUM=np.ones(6, dtype=int)))
        assert reg.PVTNUM is None and reg.count("PVT") == 1
        print("ok test_a_single_region_is_not_partitioned")

    def test_two_pvt_regions_are_split_and_zero_based() -> None:
        reg = handle_regions(_deck(PVTNUM=np.array([1, 1, 2, 2, 1])))
        assert reg.count("PVT") == 2
        assert np.array_equal(reg.PVTNUM, [0, 0, 1, 1, 0]), reg.PVTNUM
        assert np.array_equal(reg.PVTINX[0], [0, 1, 4])
        assert np.array_equal(reg.PVTINX[1], [2, 3])
        print("ok test_two_pvt_regions_are_split_and_zero_based")

    def test_actnum_restricts_the_partition_to_active_cells() -> None:
        deck = {"REGIONS": {"SATNUM": np.array([1, 2, 1, 2])},
                "GRID": {"ACTNUM": np.array([1, 0, 1, 1])}}
        reg = handle_regions(deck)
        assert reg.SATNUM.size == 3, reg.SATNUM
        assert np.array_equal(reg.SATNUM, [0, 0, 1]), reg.SATNUM
        print("ok test_actnum_restricts_the_partition_to_active_cells")

    def test_surfactant_shares_the_saturation_numbering() -> None:
        reg = handle_regions(_deck(SATNUM=np.array([1, 1, 2]),
                                   SURFNUM=np.array([1, 2, 3])))
        assert reg.count("SAT") == 3 and reg.count("SURF") == 3
        print("ok test_surfactant_shares_the_saturation_numbering")

    def test_surfactant_without_satnum_is_rejected() -> None:
        try:
            handle_regions(_deck(SURFNUM=np.array([1, 2])))
        except ValueError:
            print("ok test_surfactant_without_satnum_is_rejected")
            return
        raise AssertionError("SURFNUM without SATNUM must be rejected")

    def test_imbibition_count_is_halved_as_mrst_does() -> None:
        reg = handle_regions(_deck(IMBNUM=np.array([1, 2, 3, 4])))
        assert reg.count("IMB") == 2, reg.count("IMB")
        print("ok test_imbibition_count_is_halved_as_mrst_does")

    def test_every_cell_appears_once_per_family() -> None:
        reg = handle_regions(_deck(SATNUM=np.array([1, 2, 3, 1, 2])))
        stacked = np.sort(np.concatenate(reg.SATINX))
        assert np.array_equal(stacked, np.arange(5))
        print("ok test_every_cell_appears_once_per_family")

    test_no_regions_section_gives_one_region_everywhere()
    test_a_single_region_is_not_partitioned()
    test_two_pvt_regions_are_split_and_zero_based()
    test_actnum_restricts_the_partition_to_active_cells()
    test_surfactant_shares_the_saturation_numbering()
    test_surfactant_without_satnum_is_rejected()
    test_imbibition_count_is_halved_as_mrst_does()
    test_every_cell_appears_once_per_family()
    print("\nAll tests passed!")
