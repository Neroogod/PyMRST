"""The Egg model's deck (MRST ``test-suite/dataset-functions/egg/getDeckEGG.m``).

MRST reads the benchmark with the *general* deck reader and then converts its
units:

    deck = readEclipseDeck(fn);
    ... optionally replace PERMX/PERMY/PERMZ for realization > 0 ...
    deck = convertDeckUnits(deck);

This is that, unchanged. It replaced a bespoke Egg-only parser, which produced
a flat dictionary rather than a deck and therefore could not be handed to
`init_eclipse_grid`, `init_eclipse_rock`, `init_deck_adi_fluid` or
`init_eclipse_problem_ad_ad` - the four functions MRST's own Egg workflow is built
out of. Reading it as a deck is what makes `setup_egg` a transcription of
`setupEGG.m` instead of a re-implementation of it.

The Egg model ships a hundred permeability realizations of one geology.
Realization 0 is the base case in the DATA file; 1 to 100 replace the three
permeability arrays from `Permeability_Realizations/PERM<n>_ECL.INC`, which is
what makes the dataset usable for ensemble work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)

#: The keywords a realization file replaces (MRST's `kw` in `getDeckEGG`).
REALIZATION_KEYWORDS = ("PERMX", "PERMY", "PERMZ")


def get_deck_egg(realization: int = 0, *, data_root=None) -> Dict[str, Any]:
    """Read the Egg deck, in SI units."""
    from pymrst.modules.deckformat.egg.egg_dataset import egg_data_directory

    if not 0 <= int(realization) <= 100:
        raise ValueError("Egg realization must be a number between 0 and 100")

    path = Path(egg_data_directory(data_root))
    deck = read_eclipse_deck(path/"MRST"/"Egg_Model_ECL.DATA")

    if int(realization) > 0:
        n_cells = int(np.prod(deck["RUNSPEC"]["cartDims"]))
        include = (path/"Permeability_Realizations"
                   / f"PERM{int(realization)}_ECL.INC")
        deck["GRID"] = replace_keywords(deck["GRID"], include,
                                        REALIZATION_KEYWORDS, n_cells)

    deck = convert_deck_units(deck)
    deck["realization"] = int(realization)
    return deck


getDeckEGG = get_deck_egg


def replace_keywords(section: Dict[str, Any], path, keywords, n_cells: int):
    """Replace ``keywords`` in ``section`` from ``path`` (MRST
    ``replaceKeywords``).

    Replacing rather than merging is the point. A realization file states
    `PERMX` alone and leaves `PERMY` and `PERMZ` to the `COPY`/`MULTIPLY`
    operators it carries; a reader that kept the base deck's arrays for the
    keywords the file did not restate would run realization *n*'s `PERMX`
    against realization 0's `PERMZ`, and the vertical communication would come
    from the wrong geology.
    """
    from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
        read_keywords,
    )

    wanted = {str(name).upper() for name in keywords}
    out = dict(section)
    # `read_keywords` needs the box dimensions the operators work against, and
    # a realization file states none - so they are carried in from the deck.
    text = (f"SPECGRID\n {out['cartDims'][0]} {out['cartDims'][1]} "
            f"{out['cartDims'][2]} 1 F /\n" + Path(path).read_text())
    replacement = _read_text(text)

    missing = wanted - set(replacement)
    if missing:
        raise KeyError(
            f"{Path(path).name} does not supply {sorted(missing)}; the deck's "
            "own arrays would be left in place and the realizations mixed")
    for name in wanted:
        values = np.asarray(replacement[name], dtype=float).ravel()
        if values.size != n_cells:
            raise ValueError(
                f"{name} in {Path(path).name} has {values.size} values, not "
                f"{n_cells}")
        out[name] = values
    return out


replaceKeywords = replace_keywords


def _read_text(text: str) -> Dict[str, Any]:
    from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
        read_keywords_text,
    )
    return read_keywords_text(text)


__all__ = ["REALIZATION_KEYWORDS", "getDeckEGG", "get_deck_egg",
           "replaceKeywords", "replace_keywords"]


if __name__ == "__main__":
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

    def _available() -> bool:
        from pymrst.modules.deckformat.egg.egg_dataset import (
            egg_data_directory,
        )
        try:
            egg_data_directory()
            return True
        except FileNotFoundError:
            return False

    if not _available():
        print("skip (Egg dataset not installed)")
        raise SystemExit(0)

    DECK = get_deck_egg()

    def test_the_grid_is_the_published_sixty_by_sixty_by_seven():
        assert tuple(DECK["RUNSPEC"]["cartDims"]) == (60, 60, 7)
        assert int(np.sum(DECK["GRID"]["ACTNUM"])) == 18553
        print("ok test_the_grid_is_the_published_sixty_by_sixty_by_seven")

    def test_permeability_arrives_in_si():
        """The deck writes millidarcy. Left unconverted the field is twelve
        orders of magnitude too permeable and every well runs unconstrained."""
        kx = np.asarray(DECK["GRID"]["PERMX"], float)
        assert np.isclose(kx[0]/(MILLI*DARCY), 1288.7), kx[0]/(MILLI*DARCY)
        print("ok test_permeability_arrives_in_si")

    def test_the_vertical_permeability_is_a_tenth_of_the_horizontal():
        """`mDARCY.INC` states PERMX and derives the rest with COPY and
        MULTIPLY. A reader that skips the operators leaves PERMZ equal to
        PERMX, ten times too high."""
        kx = np.asarray(DECK["GRID"]["PERMX"], float)/(MILLI*DARCY)
        kz = np.asarray(DECK["GRID"]["PERMZ"], float)/(MILLI*DARCY)
        assert np.allclose(kz, 0.1*kx, atol=0.0), (kx[:3], kz[:3])
        print("ok test_the_vertical_permeability_is_a_tenth_of_the_horizontal")

    def test_the_equilibrium_datum_is_four_hundred_bar_at_four_thousand_metres():
        equil = DECK["SOLUTION"]["EQUIL"][0]
        assert np.isclose(equil[0], 4000.0)
        assert np.isclose(equil[1]/BARSA, 400.0), equil[1]/BARSA
        assert np.isclose(equil[2], 5000.0), "the water-oil contact"
        print("ok test_the_equilibrium_datum_is_four_hundred_bar_at_four_thousand_metres")

    def test_the_schedule_is_one_period_of_a_hundred_and_twenty_three_steps():
        schedule = DECK["SCHEDULE"]
        assert len(schedule.control) == 1, len(schedule.control)
        assert len(schedule.step["val"]) == 123, len(schedule.step["val"])
        assert np.isclose(sum(schedule.step["val"])/DAY, 3600.0)
        print("ok test_the_schedule_is_one_period_of_a_hundred_and_twenty_three_steps")

    def test_the_twelve_wells_are_eight_injectors_and_four_producers():
        control = DECK["SCHEDULE"].control[0]
        assert len(control["WELSPECS"]) == 12
        assert len(control["WCONINJE"]) == 8
        assert len(control["WCONPROD"]) == 4
        print("ok test_the_twelve_wells_are_eight_injectors_and_four_producers")

    def test_the_well_targets_are_the_published_ones():
        control = DECK["SCHEDULE"].control[0]
        rate = control["WCONINJE"][0]
        assert np.isclose(rate[4]*DAY, 79.5), rate[4]*DAY
        assert np.isclose(rate[6]/BARSA, 420.0), rate[6]/BARSA
        assert np.isclose(control["WCONPROD"][0][8]/BARSA, 395.0)
        print("ok test_the_well_targets_are_the_published_ones")

    def test_a_realization_replaces_the_permeability():
        """The whole reason the dataset exists: one geology, a hundred
        equiprobable permeability fields.

        Compared in millidarcy rather than in SI. A permeability in SI is of
        order 1e-13, and `allclose`'s default `atol` is 1e-8 - so *any* two
        permeability fields compare equal in SI and the check passes whatever
        the reader did.
        """
        other = get_deck_egg(realization=1)
        base = np.asarray(DECK["GRID"]["PERMX"], float)/(MILLI*DARCY)
        drawn = np.asarray(other["GRID"]["PERMX"], float)/(MILLI*DARCY)
        assert drawn.shape == base.shape
        assert not np.allclose(drawn, base), "realization 1 equals the base case"
        assert np.isclose(drawn[0], 366.20), drawn[0]
        print("ok test_a_realization_replaces_the_permeability")

    def test_a_realization_keeps_the_tenth_ratio_between_its_own_arrays():
        """The defect a merge rather than a replace produces: realization n's
        PERMX against realization 0's PERMZ.

        The realization files *state* all three arrays rather than deriving
        them, and they are written to three or four significant figures - 7.4
        where a tenth of 73.5 is 7.35. Measured across realization 1, the
        ratio holds to 0.68%; the base deck's `MULTIPLY` gives it exactly.
        """
        other = get_deck_egg(realization=1)
        kx = np.asarray(other["GRID"]["PERMX"], float)/(MILLI*DARCY)
        kz = np.asarray(other["GRID"]["PERMZ"], float)/(MILLI*DARCY)
        assert np.allclose(kz, 0.1*kx, rtol=1e-2, atol=0.0), (kx[:3], kz[:3])
        print("ok test_a_realization_keeps_the_tenth_ratio_between_its_own_arrays")

    def test_a_realization_outside_the_published_range_is_refused():
        for bad in (-1, 101):
            try:
                get_deck_egg(realization=bad)
            except ValueError:
                continue
            raise AssertionError(f"realization {bad} must be refused")
        print("ok test_a_realization_outside_the_published_range_is_refused")

    test_the_grid_is_the_published_sixty_by_sixty_by_seven()
    test_permeability_arrives_in_si()
    test_the_vertical_permeability_is_a_tenth_of_the_horizontal()
    test_the_equilibrium_datum_is_four_hundred_bar_at_four_thousand_metres()
    test_the_schedule_is_one_period_of_a_hundred_and_twenty_three_steps()
    test_the_twelve_wells_are_eight_injectors_and_four_producers()
    test_the_well_targets_are_the_published_ones()
    test_a_realization_replaces_the_permeability()
    test_a_realization_keeps_the_tenth_ratio_between_its_own_arrays()
    test_a_realization_outside_the_published_range_is_refused()
    print("\nAll tests passed!")
