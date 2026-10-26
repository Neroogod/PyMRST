"""Gas-oil functions from SLGOF (MRST ``props/assignSLGOF.m``).

SLGOF states the same curves as SGOF against *liquid* saturation rather than
gas saturation. Converting is two steps: Sg = 1 - Sl, then reverse the rows so
the abscissa increases again. The remaining columns are untouched, because
they were never functions of the abscissa's name.

The `max(1 - Sl, 0)` guard is MRST's and it matters: decks sometimes tabulate
Sl slightly above 1, which would otherwise produce a small negative gas
saturation and a non-monotone table.

Everything after the conversion is SGOF's, including the requirement that the
water table be assigned first.
"""

from typing import Any, Dict, List

import numpy as np

from pymrst.modules.ad_props.props.assign_sgof import assign_sgof


def slgof_to_sgof(slgof) -> List[np.ndarray]:
    """Convert SLGOF tables to SGOF tables."""
    out = []
    for table in slgof:
        table = np.asarray(table, dtype=float)
        converted = np.column_stack(
            [np.maximum(1.0 - table[:, 0], 0.0), table[:, 1:]])
        out.append(converted[::-1])
    return out


def assign_slgof(f: Dict[str, Any], slgof, reg) -> Dict[str, Any]:
    """Convert SLGOF to SGOF and assign it."""
    return assign_sgof(f, slgof_to_sgof(slgof), reg)


assignSLGOF = assign_slgof


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_swof import assign_swof
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    SWOF = np.array([[0.12, 0.00, 1.00, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])
    #         Sl    krG   krOG  Pcog     (Sl decreasing -> Sg increasing)
    SLGOF = np.array([[0.12, 1.00, 0.00, 0.0],
                      [0.50, 0.40, 0.10, 0.0],
                      [0.80, 0.05, 0.55, 0.0],
                      [1.00, 0.00, 1.00, 0.0]])

    def test_the_abscissa_becomes_gas_saturation() -> None:
        sgof = slgof_to_sgof([SLGOF])[0]
        assert np.allclose(sgof[:, 0], [0.00, 0.20, 0.50, 0.88])
        print("ok test_the_abscissa_becomes_gas_saturation")

    def test_the_rows_are_reversed_so_the_abscissa_increases() -> None:
        sgof = slgof_to_sgof([SLGOF])[0]
        assert np.all(np.diff(sgof[:, 0]) > 0), sgof[:, 0]
        print("ok test_the_rows_are_reversed_so_the_abscissa_increases")

    def test_the_other_columns_travel_with_their_rows() -> None:
        sgof = slgof_to_sgof([SLGOF])[0]
        # Sl = 0.80 -> Sg = 0.20, and that row's krG was 0.05
        row = sgof[np.isclose(sgof[:, 0], 0.20)][0]
        assert np.allclose(row[1:], [0.05, 0.55, 0.0]), row
        print("ok test_the_other_columns_travel_with_their_rows")

    def test_liquid_saturation_above_one_is_clamped() -> None:
        table = SLGOF.copy()
        table[-1, 0] = 1.02
        sgof = slgof_to_sgof([table])[0]
        assert sgof[:, 0].min() == 0.0, sgof[:, 0]
        assert np.all(np.diff(sgof[:, 0]) >= 0), "must stay monotone"
        print("ok test_liquid_saturation_above_one_is_clamped")

    def test_it_produces_the_same_curves_as_the_equivalent_sgof() -> None:
        from pymrst.modules.ad_props.props.assign_sgof import assign_sgof as sg
        reg = Regions()
        direct = sg(assign_swof({}, [SWOF], reg), slgof_to_sgof([SLGOF]), reg)
        viaslgof = assign_slgof(assign_swof({}, [SWOF], reg), [SLGOF], reg)
        probe = np.array([0.0, 0.2, 0.5, 0.88])
        assert np.allclose(direct["krG"][0](probe), viaslgof["krG"][0](probe))
        print("ok test_it_produces_the_same_curves_as_the_equivalent_sgof")

    def test_without_swof_it_warns_as_mrsts_assign_sgof_does() -> None:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f = assign_slgof({}, [SLGOF], Regions())
        assert any("No relperm points" in str(w.message) for w in caught)
        assert "krG" in f and "krOG" in f
        print("ok test_without_swof_it_warns_as_mrsts_assign_sgof_does")

    test_the_abscissa_becomes_gas_saturation()
    test_the_rows_are_reversed_so_the_abscissa_increases()
    test_the_other_columns_travel_with_their_rows()
    test_liquid_saturation_above_one_is_clamped()
    test_it_produces_the_same_curves_as_the_equivalent_sgof()
    test_without_swof_it_warns_as_mrsts_assign_sgof_does()
    print("\nAll tests passed!")
