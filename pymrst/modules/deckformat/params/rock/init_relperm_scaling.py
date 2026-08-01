"""Relative permeability end-point scaling (MRST ``params/rock/initRelpermScaling.m``).

Collects the per-cell saturation end points a deck states, so a single set of
saturation tables can be stretched cell by cell instead of tabulating a curve
for every cell.

Four numbers per phase, in this order: connate saturation, critical
saturation, maximum saturation, and maximum relative permeability. Each comes
from a differently-named keyword, and the naming is where the care is needed:

- water and gas take their maximum from ``KRW`` and ``KRG``;
- **the two oil curves share ``KRO``** - there is no KROW or KROG - because
  oil has one maximum relative permeability whichever phase it is displacing.

Every family exists twice: drainage, and imbibition under an ``I`` prefix
(``ISWL``, ``IKRW``). Missing entries stay NaN rather than defaulting, so a
consumer can tell "the deck said nothing" from "the deck said zero" - a
critical saturation of zero is a real and different statement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

PHASES = ("W", "OW", "OG", "G")


@dataclass
class PhaseScaling:
    """End points for one phase: (connate, critical, maximum, max relperm)."""

    w: np.ndarray
    ow: np.ndarray
    og: np.ndarray
    g: np.ndarray
    present: bool = False


@dataclass
class RelPermScaling:
    """Drainage and imbibition end points."""

    drainage: PhaseScaling
    imbibition: PhaseScaling


def _keywords(prefix: str, phase: str) -> Tuple[str, str, str, str]:
    """The four keywords holding one phase's end points."""
    connate = f"{prefix}S{phase}L"
    critical = f"{prefix}S{phase}CR"
    maximum = f"{prefix}S{phase}U"
    # The oil curves share KRO; only water and gas have their own maximum.
    max_kr = f"{prefix}KRO" if phase in ("OW", "OG") else f"{prefix}KR{phase}"
    return connate, critical, maximum, max_kr


def get_relperm_scaling(props: Dict[str, Any], n_cells: int, prefix: str,
                        phase: str) -> Tuple[np.ndarray, bool]:
    """End points for one phase, NaN where the deck says nothing."""
    points = np.full((n_cells, 4), np.nan)
    present = False
    for column, keyword in enumerate(_keywords(prefix, phase)):
        if keyword in props:
            points[:, column] = np.asarray(props[keyword], dtype=float).ravel()
            present = True
    return points, present


def _three_phase(props: Dict[str, Any], n_cells: int,
                 prefix: str) -> PhaseScaling:
    values = {}
    present = False
    for phase in PHASES:
        points, found = get_relperm_scaling(props, n_cells, prefix, phase)
        values[phase.lower()] = points
        present = present or found
    return PhaseScaling(present=present, **values)


def init_relperm_scaling(deck, n_cells: int) -> RelPermScaling:
    """Build the drainage and imbibition end points from ``deck.PROPS``."""
    props = deck["PROPS"] if isinstance(deck, dict) else deck.PROPS
    return RelPermScaling(drainage=_three_phase(props, n_cells, ""),
                          imbibition=_three_phase(props, n_cells, "I"))


initRelpermScaling = init_relperm_scaling


if __name__ == "__main__":

    def test_missing_keywords_stay_nan() -> None:
        """NaN distinguishes 'unstated' from a stated zero."""
        scaling = init_relperm_scaling({"PROPS": {}}, 3)
        assert np.all(np.isnan(scaling.drainage.w)), scaling.drainage.w
        assert scaling.drainage.present is False
        print("ok test_missing_keywords_stay_nan")

    def test_a_stated_zero_is_not_nan() -> None:
        scaling = init_relperm_scaling({"PROPS": {"SWL": np.zeros(3)}}, 3)
        assert np.allclose(scaling.drainage.w[:, 0], 0.0)
        assert not np.any(np.isnan(scaling.drainage.w[:, 0]))
        print("ok test_a_stated_zero_is_not_nan")

    def test_the_four_columns_are_in_keyword_order() -> None:
        props = {"SWL": [0.1] * 2, "SWCR": [0.15] * 2,
                 "SWU": [0.9] * 2, "KRW": [0.6] * 2}
        scaling = init_relperm_scaling({"PROPS": props}, 2)
        assert np.allclose(scaling.drainage.w[0], [0.1, 0.15, 0.9, 0.6])
        print("ok test_the_four_columns_are_in_keyword_order")

    def test_both_oil_curves_share_kro() -> None:
        """There is no KROW or KROG; oil has one maximum relperm."""
        scaling = init_relperm_scaling({"PROPS": {"KRO": [0.8] * 2}}, 2)
        assert np.allclose(scaling.drainage.ow[:, 3], 0.8)
        assert np.allclose(scaling.drainage.og[:, 3], 0.8)
        print("ok test_both_oil_curves_share_kro")

    def test_water_and_gas_have_their_own_maxima() -> None:
        props = {"KRW": [0.6] * 2, "KRG": [0.95] * 2}
        scaling = init_relperm_scaling({"PROPS": props}, 2)
        assert np.allclose(scaling.drainage.w[:, 3], 0.6)
        assert np.allclose(scaling.drainage.g[:, 3], 0.95)
        print("ok test_water_and_gas_have_their_own_maxima")

    def test_the_imbibition_family_uses_the_i_prefix() -> None:
        props = {"SWL": [0.1] * 2, "ISWL": [0.2] * 2}
        scaling = init_relperm_scaling({"PROPS": props}, 2)
        assert np.allclose(scaling.drainage.w[:, 0], 0.1)
        assert np.allclose(scaling.imbibition.w[:, 0], 0.2)
        print("ok test_the_imbibition_family_uses_the_i_prefix")

    def test_drainage_only_leaves_imbibition_absent() -> None:
        scaling = init_relperm_scaling({"PROPS": {"SWL": [0.1] * 2}}, 2)
        assert scaling.drainage.present is True
        assert scaling.imbibition.present is False
        print("ok test_drainage_only_leaves_imbibition_absent")

    def test_one_row_per_cell() -> None:
        scaling = init_relperm_scaling({"PROPS": {"SWL": [0.1] * 7}}, 7)
        assert scaling.drainage.w.shape == (7, 4)
        print("ok test_one_row_per_cell")

    def test_the_oil_and_gas_families_are_independent() -> None:
        props = {"SOWCR": [0.2] * 2, "SOGCR": [0.05] * 2}
        scaling = init_relperm_scaling({"PROPS": props}, 2)
        assert np.allclose(scaling.drainage.ow[:, 1], 0.2)
        assert np.allclose(scaling.drainage.og[:, 1], 0.05)
        print("ok test_the_oil_and_gas_families_are_independent")

    test_missing_keywords_stay_nan()
    test_a_stated_zero_is_not_nan()
    test_the_four_columns_are_in_keyword_order()
    test_both_oil_curves_share_kro()
    test_water_and_gas_have_their_own_maxima()
    test_the_imbibition_family_uses_the_i_prefix()
    test_drainage_only_leaves_imbibition_absent()
    test_one_row_per_cell()
    test_the_oil_and_gas_families_are_independent()
    print("\nAll tests passed!")
