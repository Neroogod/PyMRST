"""Build a fluid from a deck (MRST ``ad-props/props/initDeckADIFluid.m``).

The dispatcher: for each keyword in ``deck.PROPS``, call the matching
``assign<KEYWORD>`` and let it add its fields to the fluid. Nothing here knows
what any individual property means, which is what makes the 60-odd assign
functions independent of one another.

Three things the loop does that are not obvious:

**SWFN and SWOF go first.** SGOF and SGFN need the connate water saturation
from the water table's end points, so a dictionary-ordered pass would assign
them before it exists. MRST reorders for exactly this reason and so does
`_prioritise`.

**Some keywords are deliberately not assignable.** The end-point scaling
family (SWL, KRW, ...) and the compositional keywords are consumed elsewhere -
by `impose_relperm_scaling` and the equation-of-state model - so calling an
assign function for them would be wrong even if one existed. `EXCLUDED` lists
them, following MRST's `exclude_properties`.

**Single-region lists are unwrapped.** Each assign function stores one closure
per region; when there is only one, the list is replaced by its element, so
callers of a single-region fluid write ``fluid['bO'](p, rs, flag)`` rather
than ``fluid['bO'][0](...)``. This is why `assign_relperm` can require plain
callables.

``ignore_errors`` defaults to True, as in MRST: a deck keyword this port
cannot assign produces a warning and a fluid missing that property, not an
exception. Pass False when a silently incomplete fluid would be worse.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pymrst.modules.ad_props.props.assign_density import assign_density
from pymrst.modules.ad_props.props.assign_ehystr import assign_ehystr
from pymrst.modules.ad_props.props.assign_gravity import assign_gravity
from pymrst.modules.ad_props.props.assign_misc import assign_misc
from pymrst.modules.ad_props.props.assign_msfn import assign_msfn
from pymrst.modules.ad_props.props.assign_oilvist import assign_oilvist
from pymrst.modules.ad_props.props.assign_plmixpar import assign_plmixpar
from pymrst.modules.ad_props.props.assign_plyads import assign_plyads
from pymrst.modules.ad_props.props.assign_plymax import assign_plymax
from pymrst.modules.ad_props.props.assign_plyrock import assign_plyrock
from pymrst.modules.ad_props.props.assign_plyshear import assign_plyshear
from pymrst.modules.ad_props.props.assign_plyshlog import assign_plyshlog
from pymrst.modules.ad_props.props.assign_plyvisc import assign_plyvisc
from pymrst.modules.ad_props.props.assign_pmisc import assign_pmisc
from pymrst.modules.ad_props.props.assign_pvcdo import assign_pvcdo
from pymrst.modules.ad_props.props.assign_pvdg import assign_pvdg
from pymrst.modules.ad_props.props.assign_pvdo import assign_pvdo
from pymrst.modules.ad_props.props.assign_pvds import assign_pvds
from pymrst.modules.ad_props.props.assign_pvtg import assign_pvtg
from pymrst.modules.ad_props.props.assign_pvto import assign_pvto
from pymrst.modules.ad_props.props.assign_pvtw import assign_pvtw
from pymrst.modules.ad_props.props.assign_relperm import assign_relperm
from pymrst.modules.ad_props.props.assign_rock import assign_rock
from pymrst.modules.ad_props.props.assign_rocktab import assign_rocktab
from pymrst.modules.ad_props.props.assign_rsconstt import assign_rsconstt
from pymrst.modules.ad_props.props.assign_sdensity import assign_sdensity
from pymrst.modules.ad_props.props.assign_sgfn import assign_sgfn
from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
from pymrst.modules.ad_props.props.assign_sgwfn import assign_sgwfn
from pymrst.modules.ad_props.props.assign_shrate import assign_shrate
from pymrst.modules.ad_props.props.assign_slgof import assign_slgof
from pymrst.modules.ad_props.props.assign_sof2 import assign_sof2
from pymrst.modules.ad_props.props.assign_sof3 import assign_sof3
from pymrst.modules.ad_props.props.assign_specheat import assign_specheat
from pymrst.modules.ad_props.props.assign_specrock import assign_specrock
from pymrst.modules.ad_props.props.assign_ssfn import assign_ssfn
from pymrst.modules.ad_props.props.assign_surfads import assign_surfads
from pymrst.modules.ad_props.props.assign_surfcapd import assign_surfcapd
from pymrst.modules.ad_props.props.assign_surfrock import assign_surfrock
from pymrst.modules.ad_props.props.assign_surfst import assign_surfst
from pymrst.modules.ad_props.props.assign_surfvisc import assign_surfvisc
from pymrst.modules.ad_props.props.assign_swfn import assign_swfn
from pymrst.modules.ad_props.props.assign_swof import assign_swof
from pymrst.modules.ad_props.props.assign_tlmixpar import assign_tlmixpar
from pymrst.modules.ad_props.props.assign_watervist import assign_watervist
from pymrst.modules.ad_props.utils.handle_regions import Regions, handle_regions
from pymrst.utils.units import BARSA
from pymrst.utils.units.units import KILOGRAM, METER

ASSIGNERS: Dict[str, Callable] = {
    "DENSITY": assign_density, "SDENSITY": assign_sdensity,
    "GRAVITY": assign_gravity, "ROCK": assign_rock, "ROCKTAB": assign_rocktab,
    "PVTW": assign_pvtw, "PVCDO": assign_pvcdo, "PVDO": assign_pvdo,
    "PVDG": assign_pvdg, "PVDS": assign_pvds, "PVTO": assign_pvto,
    "PVTG": assign_pvtg, "RSCONSTT": assign_rsconstt,
    "SWOF": assign_swof, "SGOF": assign_sgof, "SLGOF": assign_slgof,
    "SWFN": assign_swfn, "SGFN": assign_sgfn, "SGWFN": assign_sgwfn,
    "SOF2": assign_sof2, "SOF3": assign_sof3, "SSFN": assign_ssfn,
    "MSFN": assign_msfn, "EHYSTR": assign_ehystr,
    "PLYVISC": assign_plyvisc, "PLYADS": assign_plyads,
    "PLYSHEAR": assign_plyshear, "PLYSHLOG": assign_plyshlog,
    "PLYMAX": assign_plymax, "PLYROCK": assign_plyrock,
    "PLMIXPAR": assign_plmixpar, "TLMIXPAR": assign_tlmixpar,
    "SHRATE": assign_shrate,
    "SURFST": assign_surfst, "SURFVISC": assign_surfvisc,
    "SURFADS": assign_surfads, "SURFCAPD": assign_surfcapd,
    "SURFROCK": assign_surfrock,
    "MISC": assign_misc, "PMISC": assign_pmisc,
    "SPECHEAT": assign_specheat, "SPECROCK": assign_specrock,
    "WATERVIST": assign_watervist, "OILVIST": assign_oilvist,
}

# Keywords that carry data but produce no property function of their own.
# Transcribed from MRST's exclude_properties().
EXCLUDED = frozenset({
    "KRW", "KRO", "KRG",
    "SWL", "SWCR", "SWU", "SGL", "SGCR", "SGU", "SOWCR", "SOGCR",
    "ISWL", "ISWCR", "ISWU", "ISGL", "ISGCR", "ISGU", "ISOWCR", "ISOGCR",
    "CNAMES", "BIC", "ACF", "PCRIT", "TCRIT", "VCRIT", "MW", "ZCRIT", "EOS",
    "ZMFVD", "TEMPVD", "SSHIFT", "SCALECRS", "SWATINIT", "PRCORR", "STCOND",
})

# The water tables, which the gas tables read connate water from.
FIRST = ("SWFN", "SWOF")

# MRST's fallback when a deck states no DENSITY.
DEFAULT_DENSITY = np.array([[600.0, 999.014, 1.0]]) * KILOGRAM / METER ** 3


def _prioritise(names: List[str]) -> List[str]:
    """Put the water saturation tables first; see the module docstring."""
    head = [name for name in FIRST if name in names]
    return head + [name for name in names if name not in head]


def _get_pressures(props, resample_poly: int) -> Optional[np.ndarray]:
    """MRST's `get_pressures`: sample pressures spanning the dead-oil/gas tables.

    The range is taken from PVDO and PVDG, the two keywords whose first column
    is a pressure axis, and falls back to MRST's 0..1000 bar when the deck has
    neither. `resample_poly = 0` returns nothing at all, matching MATLAB's
    `linspace(a, b, 0)` - which is why the default path evaluates properties
    directly instead of tabulating them.
    """
    if resample_poly <= 0:
        return None
    p_min, p_max = np.inf, -np.inf
    for keyword in ("PVDG", "PVDO"):
        for table in props.get(keyword, ()) or ():
            column = np.asarray(table, dtype=float)[:, 0]
            if column.size:
                p_min = min(p_min, float(column.min()))
                p_max = max(p_max, float(column.max()))
    if not np.isfinite(p_max):
        p_max = 1000.0 * BARSA
    if not np.isfinite(p_min):
        p_min = 0.0
    return np.linspace(p_min, p_max, int(resample_poly))


def init_deck_adi_fluid(deck, grid=None, regions: Optional[Regions] = None,
                        ignore_errors: bool = True,
                        **options) -> Dict[str, Any]:
    """Build a fluid by dispatching each PROPS keyword to its assign function."""
    props = dict(deck["PROPS"] if isinstance(deck, dict) else deck.PROPS)
    if "DENSITY" not in props and "GRAVITY" not in props:
        props["DENSITY"] = DEFAULT_DENSITY

    reg = regions if regions is not None else handle_regions(deck, grid)
    for name, value in options.items():
        if not hasattr(reg, name):
            raise TypeError(f"unknown option '{name}'")
        setattr(reg, name, value)

    # MRST's `optimize` is a shorthand: it turns on tabulation by choosing a
    # sample count, and everything downstream reacts to `prange` alone. Both
    # default to off, so this leaves the direct-evaluation path untouched.
    if reg.optimize and not reg.resample_poly:
        reg.resample_poly = 20
    if reg.prange is None:
        reg.prange = _get_pressures(props, reg.resample_poly)

    fluid: Dict[str, Any] = {}
    for keyword in _prioritise(list(props)):
        if keyword in EXCLUDED:
            continue
        assign = ASSIGNERS.get(keyword)
        if assign is None:
            if not ignore_errors:
                raise KeyError(f"no assign function for PROPS keyword "
                               f"'{keyword}'")
            continue
        try:
            fluid = assign(fluid, props[keyword], reg)
        except Exception as error:                       # noqa: BLE001
            if not ignore_errors:
                raise
            warnings.warn(f"could not assign property '{keyword}': {error}")

    # One region: store the closure itself rather than a one-element list.
    for name, item in list(fluid.items()):
        if isinstance(item, list) and len(item) == 1:
            fluid[name] = item[0]

    _guard_surface_densities(fluid, deck)
    return fluid


def _guard_surface_densities(fluid: Dict[str, Any], deck) -> None:
    """A present phase may not have zero surface density.

    A zero would divide by zero when converting reservoir to surface volumes.
    MRST substitutes 1, which is arbitrary but finite, and only for phases the
    RUNSPEC declares.
    """
    runspec = (deck.get("RUNSPEC", {}) if isinstance(deck, dict)
               else getattr(deck, "RUNSPEC", {}))
    for phase, field in (("GAS", "rhoGS"), ("OIL", "rhoOS")):
        if runspec.get(phase) and field in fluid:
            values = np.atleast_1d(np.asarray(fluid[field], dtype=float))
            if np.any(values == 0):
                values = np.where(values == 0, 1.0, values)
                fluid[field] = values


initDeckADIFluid = init_deck_adi_fluid


if __name__ == "__main__":

    SWOF = np.array([[0.12, 0.00, 1.00, 0.0],
                     [0.50, 0.20, 0.40, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])
    SGOF = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.40, 0.30, 0.35, 0.0],
                     [0.88, 1.00, 0.00, 0.0]])
    PVDG = np.array([[1.0e7, 5.0e-3, 1.4e-5], [3.0e7, 2.5e-3, 2.0e-5]])

    def _deck(**props):
        return {"RUNSPEC": {"OIL": True, "WATER": True, "GAS": True},
                "REGIONS": {}, "GRID": {},
                "PROPS": props}

    def test_each_keyword_reaches_its_assign_function() -> None:
        fluid = init_deck_adi_fluid(_deck(
            DENSITY=np.array([[800.0, 1000.0, 1.2]]),
            PVDG=[PVDG], SWOF=[SWOF], SGOF=[SGOF]))
        for field in ("rhoOS", "bG", "krW", "krG"):
            assert field in fluid, field
        print("ok test_each_keyword_reaches_its_assign_function")

    def test_the_water_table_is_assigned_before_the_gas_table() -> None:
        """SGOF needs SWOF's connate water; dict order would not guarantee it."""
        fluid = init_deck_adi_fluid(_deck(SGOF=[SGOF], SWOF=[SWOF]))
        assert "krOG" in fluid, "SGOF must have succeeded"
        assert set(fluid["krPts"]) == {"w", "ow", "g", "og"}
        print("ok test_the_water_table_is_assigned_before_the_gas_table")

    def test_prioritise_moves_the_water_tables_to_the_front() -> None:
        assert _prioritise(["SGOF", "PVDG", "SWOF"])[0] == "SWOF"
        assert _prioritise(["SGFN", "SWFN"])[0] == "SWFN"
        assert _prioritise(["PVDG", "PVTW"]) == ["PVDG", "PVTW"]
        print("ok test_prioritise_moves_the_water_tables_to_the_front")

    def test_a_single_region_fluid_stores_bare_callables() -> None:
        fluid = init_deck_adi_fluid(_deck(SWOF=[SWOF]))
        assert callable(fluid["krW"]), "not a one-element list"
        assert np.isclose(fluid["krW"](0.50), 0.20)
        print("ok test_a_single_region_fluid_stores_bare_callables")

    def test_a_multi_region_fluid_keeps_its_lists() -> None:
        deck = _deck(SWOF=[SWOF, SWOF])
        deck["RUNSPEC"]["TABDIMS"] = [2, 1]
        deck["REGIONS"] = {"SATNUM": np.array([1, 1, 2, 2])}
        fluid = init_deck_adi_fluid(deck)
        assert isinstance(fluid["krW"], list) and len(fluid["krW"]) == 2
        print("ok test_a_multi_region_fluid_keeps_its_lists")

    def test_excluded_keywords_are_skipped() -> None:
        """SWL belongs to end-point scaling, not to an assign function."""
        fluid = init_deck_adi_fluid(_deck(SWOF=[SWOF], SWL=np.zeros(4)))
        assert "SWL" not in fluid
        print("ok test_excluded_keywords_are_skipped")

    def test_a_missing_density_gets_mrsts_default() -> None:
        fluid = init_deck_adi_fluid(_deck(SWOF=[SWOF]))
        assert np.isclose(fluid["rhoOS"], 600.0), fluid["rhoOS"]
        print("ok test_a_missing_density_gets_mrsts_default")

    def test_a_stated_density_wins() -> None:
        fluid = init_deck_adi_fluid(
            _deck(DENSITY=np.array([[850.0, 1010.0, 0.9]])))
        assert np.isclose(fluid["rhoOS"], 850.0)
        print("ok test_a_stated_density_wins")

    def test_a_zero_density_for_a_present_phase_is_replaced() -> None:
        fluid = init_deck_adi_fluid(
            _deck(DENSITY=np.array([[800.0, 1000.0, 0.0]])))
        assert np.all(np.atleast_1d(fluid["rhoGS"]) == 1.0), fluid["rhoGS"]
        print("ok test_a_zero_density_for_a_present_phase_is_replaced")

    def test_a_zero_density_for_an_absent_phase_is_left_alone() -> None:
        deck = _deck(DENSITY=np.array([[800.0, 1000.0, 0.0]]))
        deck["RUNSPEC"] = {"OIL": True, "WATER": True}     # no GAS
        fluid = init_deck_adi_fluid(deck)
        assert np.all(np.atleast_1d(fluid["rhoGS"]) == 0.0)
        print("ok test_a_zero_density_for_an_absent_phase_is_left_alone")

    def test_an_unknown_keyword_is_tolerated_by_default() -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fluid = init_deck_adi_fluid(_deck(SWOF=[SWOF], MYSTERY=[1, 2]))
        assert "krW" in fluid, "the rest of the deck must still assign"
        print("ok test_an_unknown_keyword_is_tolerated_by_default")

    def test_strict_mode_reports_an_unknown_keyword() -> None:
        try:
            init_deck_adi_fluid(_deck(SWOF=[SWOF], MYSTERY=[1, 2]),
                                ignore_errors=False)
        except KeyError as error:
            assert "MYSTERY" in str(error)
            print("ok test_strict_mode_reports_an_unknown_keyword")
            return
        raise AssertionError("strict mode must report an unknown keyword")

    # No row with zero krG: MRST's getPoints indexes an empty find there.
    SGOF_BAD = np.array([[0.00, 0.10, 1.00, 0.0],
                         [0.88, 1.00, 0.00, 0.0]])

    def test_a_failing_assign_is_warned_about_not_raised() -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fluid = init_deck_adi_fluid(_deck(SWOF=[SWOF], SGOF=[SGOF_BAD]))
        assert any("SGOF" in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
        assert "krG" not in fluid
        print("ok test_a_failing_assign_is_warned_about_not_raised")

    def test_strict_mode_reraises_a_failing_assign() -> None:
        try:
            init_deck_adi_fluid(_deck(SWOF=[SWOF], SGOF=[SGOF_BAD]), ignore_errors=False)
        except IndexError:
            print("ok test_strict_mode_reraises_a_failing_assign")
            return
        raise AssertionError("strict mode must reraise")

    def test_prange_is_empty_unless_asked_for() -> None:
        """The default path evaluates directly - MRST's `linspace(a, b, 0)`."""
        assert _get_pressures({"PVDG": [PVDG]}, 0) is None
        print("ok test_prange_is_empty_unless_asked_for")

    def test_prange_spans_the_dead_gas_table() -> None:
        column = np.asarray(PVDG, dtype=float)[:, 0]
        out = _get_pressures({"PVDG": [PVDG]}, 5)
        assert out.size == 5
        assert np.isclose(out[0], column.min())
        assert np.isclose(out[-1], column.max())
        print("ok test_prange_spans_the_dead_gas_table")

    def test_prange_falls_back_when_the_deck_has_no_pressure_axis() -> None:
        out = _get_pressures({}, 4)
        assert np.isclose(out[0], 0.0) and np.isclose(out[-1], 1000.0 * BARSA)
        print("ok test_prange_falls_back_when_the_deck_has_no_pressure_axis")

    def test_optimize_turns_tabulation_on() -> None:
        """`optimize` is a shorthand for a sample count; nothing downstream
        reads it, so if it did not set `prange` it would do nothing at all."""
        plain = handle_regions(_deck(PVDG=[PVDG]))
        init_deck_adi_fluid(_deck(PVDG=[PVDG]), regions=plain)
        assert plain.prange is None

        tuned = handle_regions(_deck(PVDG=[PVDG]))
        init_deck_adi_fluid(_deck(PVDG=[PVDG]), regions=tuned, optimize=True)
        assert tuned.resample_poly == 20
        assert tuned.prange is not None and tuned.prange.size == 20
        print("ok test_optimize_turns_tabulation_on")

    def test_an_explicit_prange_is_not_overwritten() -> None:
        given = np.linspace(1.0e6, 2.0e6, 7)
        reg = handle_regions(_deck(PVDG=[PVDG]))
        reg.prange = given
        init_deck_adi_fluid(_deck(PVDG=[PVDG]), regions=reg, optimize=True)
        assert np.allclose(reg.prange, given), reg.prange
        print("ok test_an_explicit_prange_is_not_overwritten")

    def test_options_reach_the_region_object() -> None:
        fluid = init_deck_adi_fluid(_deck(PVDG=[PVDG]),
                                    pvt_method_gas="parallel")
        assert "bG" in fluid
        print("ok test_options_reach_the_region_object")

    def test_an_unknown_option_is_rejected() -> None:
        try:
            init_deck_adi_fluid(_deck(SWOF=[SWOF]), nonsense=1)
        except TypeError as error:
            assert "nonsense" in str(error)
            print("ok test_an_unknown_option_is_rejected")
            return
        raise AssertionError("an unknown option must be rejected")

    def test_every_assigner_names_a_keyword_the_reader_can_read() -> None:
        """An assigner for an unreadable keyword can never fire.

        The two exceptions are MRST's own: `assignOILVIST` and
        `assignWATERVIST` exist, but ``OILVIST`` and ``WATERVIST`` appear
        nowhere in `readPROPS.m`, so no deck MRST can read will ever reach
        them. They are ported for completeness and are unreachable through
        this dispatcher, exactly as in MRST.
        """
        from pymrst.modules.deckformat.deckinput.read_props import SPECS
        unreachable = {"OILVIST", "WATERVIST"}
        unknown = sorted(set(ASSIGNERS) - set(SPECS) - unreachable)
        assert not unknown, \
            f"assigners for keywords the reader cannot read: {unknown}"
        assert unreachable <= set(ASSIGNERS), "both are still ported"
        assert not (unreachable & set(SPECS)), \
            "readPROPS gained these keywords; they are now reachable"
        print(f"ok test_every_assigner_names_a_keyword_the_reader_can_read "
              f"({len(ASSIGNERS)} assigners)")

    def test_every_readable_table_keyword_has_an_assigner() -> None:
        """The other direction: a keyword read but never assigned is dead data."""
        from pymrst.modules.deckformat.deckinput.read_props import SPECS
        missing = sorted(set(SPECS) - set(ASSIGNERS) - EXCLUDED)
        # RTEMP/RTEMPA and PVCO are read by readPROPS but have no assign
        # function in MRST either; they are consumed by the models directly.
        expected = {"RTEMP", "RTEMPA", "PVCO", "PARACHOR"}
        assert set(missing) == expected, sorted(missing)
        print("ok test_every_readable_table_keyword_has_an_assigner")

    def test_assigners_and_excluded_do_not_overlap() -> None:
        overlap = sorted(set(ASSIGNERS) & EXCLUDED)
        assert not overlap, overlap
        print("ok test_assigners_and_excluded_do_not_overlap")

    test_each_keyword_reaches_its_assign_function()
    test_the_water_table_is_assigned_before_the_gas_table()
    test_prioritise_moves_the_water_tables_to_the_front()
    test_a_single_region_fluid_stores_bare_callables()
    test_a_multi_region_fluid_keeps_its_lists()
    test_excluded_keywords_are_skipped()
    test_a_missing_density_gets_mrsts_default()
    test_a_stated_density_wins()
    test_a_zero_density_for_a_present_phase_is_replaced()
    test_a_zero_density_for_an_absent_phase_is_left_alone()
    test_an_unknown_keyword_is_tolerated_by_default()
    test_strict_mode_reports_an_unknown_keyword()
    test_a_failing_assign_is_warned_about_not_raised()
    test_strict_mode_reraises_a_failing_assign()
    test_options_reach_the_region_object()
    test_an_unknown_option_is_rejected()
    test_every_assigner_names_a_keyword_the_reader_can_read()
    test_every_readable_table_keyword_has_an_assigner()
    test_assigners_and_excluded_do_not_overlap()
    test_prange_is_empty_unless_asked_for()
    test_prange_spans_the_dead_gas_table()
    test_prange_falls_back_when_the_deck_has_no_pressure_axis()
    test_optimize_turns_tabulation_on()
    test_an_explicit_prange_is_not_overwritten()
    print("\nAll tests passed!")
