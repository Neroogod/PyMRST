"""Attach end-point scaling to a model (MRST ``ad-props/imposeRelpermScaling.m``).

Lets a caller impose per-cell saturation end points on a model built from
tables that have none - the programmatic equivalent of a deck's ENDSCALE
section. It records the end points on ``model.rock.krscale`` and writes the
RUNSPEC and PROPS entries the scaling machinery reads.

Two- or three-point scaling is selected by `n_points`, which is written into
the deck as SCALECRS: 'NO' for two-point, 'YES' for three. Two-point moves the
connate and maximum saturations; three-point also honours the critical
saturation, which matters when residual oil varies across the field.

**The oil curve is renamed, not scaled.** A two-phase model built from SOF2
has a single ``krO``, but the scaling machinery works on ``krOW`` and
``krOG``. So `krO` is promoted to whichever the model's phases imply and then
removed. If the model has both water and gas and only a generic `krO`, there
is no way to choose, and MRST warns and leaves it - this port raises, because
silently unscaled oil is exactly the failure the caller is trying to avoid.

MRST's own invalid-keyword branch calls ``warnProblem(prob)`` with ``prob``
undefined, so passing an unrecognised keyword raises an unrelated error there.
This port names the offending keywords instead.
"""

from typing import Any, Dict, Optional

import numpy as np

from pymrst.modules.deckformat.params.rock.init_relperm_scaling import (
    init_relperm_scaling,
)

VALID = ("SWL", "SWCR", "SWU", "SGL", "SGCR", "SGU", "SOWCR", "SOGCR",
         "KRW", "KRO", "KRG")


def impose_relperm_scaling(model, n_points: int = 2,
                           strict: bool = True, **scaling):
    """Impose per-cell end points on ``model``.

    ``scaling`` takes any of `VALID` as keyword arguments, each a scalar or one
    value per cell. ``strict`` rejects unrecognised keywords; set it False to
    warn and drop them, as MRST intends to.
    """
    fluid = model.fluid
    if "krPts" not in fluid:
        raise ValueError(
            "to impose rel-perm scaling the fluid must carry 'krPts'; assign "
            "the saturation keywords first")
    if n_points not in (2, 3):
        raise ValueError(f"only 2- or 3-point scaling is supported, "
                         f"not {n_points}")

    points = {name.upper(): value for name, value in scaling.items()}
    unknown = sorted(set(points) - set(VALID))
    if unknown:
        message = (f"unrecognised scaling keyword(s): {', '.join(unknown)}; "
                   f"valid are {', '.join(VALID)}")
        if strict:
            raise ValueError(message)
        import warnings
        warnings.warn(message)
        points = {k: v for k, v in points.items() if k in VALID}

    n_cells = model.G.cells.num
    for name, value in list(points.items()):
        value = np.asarray(value, dtype=float).ravel()
        if value.size == 1:
            value = np.full(n_cells, value[0])       # expand a single value
        elif value.size != n_cells:
            raise ValueError(
                f"{name} has {value.size} values but the grid has {n_cells} "
                f"cells")
        points[name] = value

    model.rock.krscale = init_relperm_scaling({"PROPS": points}, n_cells)
    model.inputdata = {
        "RUNSPEC": {"ENDSCALE": ["NODIR", "REVERS", 1, 20, 0]},
        "PROPS": {"SCALECRS": ["YES" if n_points == 3 else "NO"]},
        "GRID": None,
        "SOLUTION": None,
    }

    if getattr(model, "oil", False) and "krO" in fluid:
        _promote_oil_curve(model, fluid)
    return model


def _promote_oil_curve(model, fluid: Dict[str, Any]) -> None:
    """Rename a generic ``krO`` to the curve the model's phases imply."""
    water, gas = getattr(model, "water", False), getattr(model, "gas", False)

    if water and not gas and "krOW" not in fluid:
        fluid["krOW"] = fluid["krO"]
        if "o" in fluid.get("krPts", {}):
            fluid["krPts"]["ow"] = fluid["krPts"]["o"]
    if gas and not water and "krOG" not in fluid:
        fluid["krOG"] = fluid["krG"]
        if "o" in fluid.get("krPts", {}):
            fluid["krPts"]["og"] = fluid["krPts"]["o"]

    if (not water or "krOW" in fluid) and (not gas or "krOG" in fluid):
        del fluid["krO"]
    else:
        raise ValueError(
            "the fluid has a generic 'krO' that cannot be assigned to either "
            "the water-oil or gas-oil curve, so it would go unscaled; supply "
            "krOW and krOG instead")


imposeRelpermScaling = impose_relperm_scaling


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(water=True, gas=True, oil=True, fluid=None, n_cells=4):
        grid = SimpleNamespace(cells=SimpleNamespace(num=n_cells))
        return SimpleNamespace(
            G=grid, rock=SimpleNamespace(), inputdata=None,
            water=water, gas=gas, oil=oil,
            fluid=fluid if fluid is not None else {"krPts": {"w": None}})

    def test_scaling_lands_on_the_rock() -> None:
        model = impose_relperm_scaling(_model(), SWL=0.1)
        assert np.allclose(model.rock.krscale.drainage.w[:, 0], 0.1)
        print("ok test_scaling_lands_on_the_rock")

    def test_a_single_value_is_expanded_to_every_cell() -> None:
        model = impose_relperm_scaling(_model(n_cells=6), SWCR=0.15)
        assert model.rock.krscale.drainage.w.shape == (6, 4)
        assert np.allclose(model.rock.krscale.drainage.w[:, 1], 0.15)
        print("ok test_a_single_value_is_expanded_to_every_cell")

    def test_a_per_cell_array_is_accepted() -> None:
        model = impose_relperm_scaling(_model(), SWL=[0.1, 0.2, 0.3, 0.4])
        assert np.allclose(model.rock.krscale.drainage.w[:, 0],
                           [0.1, 0.2, 0.3, 0.4])
        print("ok test_a_per_cell_array_is_accepted")

    def test_a_wrong_length_is_rejected() -> None:
        try:
            impose_relperm_scaling(_model(), SWL=[0.1, 0.2])
        except ValueError as error:
            assert "4 cells" in str(error), str(error)
            print("ok test_a_wrong_length_is_rejected")
            return
        raise AssertionError("a mismatched length must be rejected")

    def test_two_point_scaling_writes_scalecrs_no() -> None:
        model = impose_relperm_scaling(_model(), n_points=2, SWL=0.1)
        assert model.inputdata["PROPS"]["SCALECRS"] == ["NO"]
        print("ok test_two_point_scaling_writes_scalecrs_no")

    def test_three_point_scaling_writes_scalecrs_yes() -> None:
        model = impose_relperm_scaling(_model(), n_points=3, SWL=0.1)
        assert model.inputdata["PROPS"]["SCALECRS"] == ["YES"]
        print("ok test_three_point_scaling_writes_scalecrs_yes")

    def test_an_unsupported_point_count_is_rejected() -> None:
        try:
            impose_relperm_scaling(_model(), n_points=4, SWL=0.1)
        except ValueError as error:
            assert "2- or 3-point" in str(error)
            print("ok test_an_unsupported_point_count_is_rejected")
            return
        raise AssertionError("only 2- and 3-point scaling exist")

    def test_a_fluid_without_krpts_is_rejected() -> None:
        try:
            impose_relperm_scaling(_model(fluid={}), SWL=0.1)
        except ValueError as error:
            assert "krPts" in str(error)
            print("ok test_a_fluid_without_krpts_is_rejected")
            return
        raise AssertionError("krPts is required")

    def test_an_unknown_keyword_is_named() -> None:
        """MRST calls warnProblem(prob) with prob undefined; this names it."""
        try:
            impose_relperm_scaling(_model(), SWL=0.1, SOMETHING=0.2)
        except ValueError as error:
            assert "SOMETHING" in str(error), str(error)
            print("ok test_an_unknown_keyword_is_named")
            return
        raise AssertionError("an unknown keyword must be reported")

    def test_a_lenient_caller_can_drop_unknown_keywords() -> None:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = impose_relperm_scaling(_model(), strict=False,
                                           SWL=0.1, SOMETHING=0.2)
        assert caught and "SOMETHING" in str(caught[0].message)
        assert np.allclose(model.rock.krscale.drainage.w[:, 0], 0.1)
        print("ok test_a_lenient_caller_can_drop_unknown_keywords")

    def test_keywords_are_matched_case_insensitively() -> None:
        model = impose_relperm_scaling(_model(), swl=0.1)
        assert np.allclose(model.rock.krscale.drainage.w[:, 0], 0.1)
        print("ok test_keywords_are_matched_case_insensitively")

    def test_a_two_phase_water_oil_model_promotes_kro() -> None:
        fluid = {"krPts": {"o": "pts"}, "krO": "curve"}
        model = _model(water=True, gas=False, fluid=fluid)
        impose_relperm_scaling(model, SWL=0.1)
        assert model.fluid["krOW"] == "curve"
        assert model.fluid["krPts"]["ow"] == "pts"
        assert "krO" not in model.fluid, "the generic curve must be removed"
        print("ok test_a_two_phase_water_oil_model_promotes_kro")

    def test_a_two_phase_oil_gas_model_promotes_kro() -> None:
        fluid = {"krPts": {"o": "pts"}, "krO": "curve", "krG": "gas"}
        model = _model(water=False, gas=True, fluid=fluid)
        impose_relperm_scaling(model, SGL=0.0)
        assert model.fluid["krOG"] == "gas"
        assert "krO" not in model.fluid
        print("ok test_a_two_phase_oil_gas_model_promotes_kro")

    def test_a_three_phase_model_with_only_kro_is_rejected() -> None:
        """MRST warns and leaves the oil unscaled; that is the silent failure."""
        fluid = {"krPts": {"o": "pts"}, "krO": "curve"}
        try:
            impose_relperm_scaling(_model(water=True, gas=True, fluid=fluid),
                                   SWL=0.1)
        except ValueError as error:
            assert "unscaled" in str(error)
            print("ok test_a_three_phase_model_with_only_kro_is_rejected")
            return
        raise AssertionError("an unassignable krO must be reported")

    def test_a_fluid_with_both_oil_curves_is_left_alone() -> None:
        fluid = {"krPts": {}, "krOW": "ow", "krOG": "og"}
        model = impose_relperm_scaling(_model(fluid=fluid), SWL=0.1)
        assert model.fluid["krOW"] == "ow" and model.fluid["krOG"] == "og"
        print("ok test_a_fluid_with_both_oil_curves_is_left_alone")

    def test_unstated_end_points_stay_nan() -> None:
        model = impose_relperm_scaling(_model(), SWL=0.1)
        assert np.all(np.isnan(model.rock.krscale.drainage.g))
        print("ok test_unstated_end_points_stay_nan")

    test_scaling_lands_on_the_rock()
    test_a_single_value_is_expanded_to_every_cell()
    test_a_per_cell_array_is_accepted()
    test_a_wrong_length_is_rejected()
    test_two_point_scaling_writes_scalecrs_no()
    test_three_point_scaling_writes_scalecrs_yes()
    test_an_unsupported_point_count_is_rejected()
    test_a_fluid_without_krpts_is_rejected()
    test_an_unknown_keyword_is_named()
    test_a_lenient_caller_can_drop_unknown_keywords()
    test_keywords_are_matched_case_insensitively()
    test_a_two_phase_water_oil_model_promotes_kro()
    test_a_two_phase_oil_gas_model_promotes_kro()
    test_a_three_phase_model_with_only_kro_is_rejected()
    test_a_fluid_with_both_oil_curves_is_left_alone()
    test_unstated_end_points_stay_nan()
    print("\nAll tests passed!")
