"""MRST ``ad-core/statefunctions/flowprops/RvMax.m``.

The dew-point limit: the most oil gas can vaporise, `rvSat` at the gas-phase
pressure, optionally reduced as oil saturation falls below its historical
maximum (ECLIPSE's VAPPARS).

A Newton update can leave a cell with `sO < -1e-4`, so that the VAPPARS
factor is negative. MATLAB's power of a negative number to a non-integer
exponent is complex, and MRST iterates on with a complex `rv` (Norne, report
step 1). NumPy gives NaN there and the step is cut. The port takes the real
part of MATLAB's power, value and derivative (`_matlab_real_power`), which
reproduces MRST's Newton iterations and ministeps on Norne; only the
imaginary part MATLAB carries in those cells is left out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import ad_max, value

VAPPARS_OFFSET: float = 1e-4


def _cospi(x: float) -> float:
    """`cos(pi*x)`, exact where it is 0 or +-1, as MATLAB's complex power has it."""
    if float(2*x).is_integer():
        return (1.0, 0.0, -1.0, 0.0)[int(2*x) % 4]
    return float(np.cos(np.pi*x))


def _matlab_real_power(base: Any, exponent: float) -> Any:
    """`real(base.^exponent)` as MATLAB evaluates it, with derivatives: for a
    negative base, `|base|^exponent*cos(pi*exponent)`."""
    negative = np.asarray(value(base)) < 0
    if float(exponent).is_integer() or not np.any(negative):
        return base**exponent
    flip = np.where(negative, -1.0, 1.0)
    return ((base*flip)**exponent)*np.where(negative, _cospi(exponent), 1.0)


class RvMax(StateFunction):
    """`R_v^max = rvSat(p_g)`, times `(sO/sOmax)^rvReduction` when set."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="R_v^{max}", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.rvReduction = 0.0
        self.depends_on("PhasePressures")
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        p_phase = self.get_evaluated_dependencies(state, "PhasePressures")
        pg = p_phase[int(model.water) + int(model.oil) + int(model.gas) - 1]
        rv_sat = self.evaluate_fluid(model, "rvSat", pg)
        if self.rvReduction > 0 and "sMax" in state:
            so_max, so = model.get_props(state, "somax", "so")
            so_max = ad_max(so_max, so)
            factor = (so + VAPPARS_OFFSET)/(so_max + VAPPARS_OFFSET)
            rv_sat = rv_sat*_matlab_real_power(factor, self.rvReduction)
        return rv_sat

    evaluateOnDomain = evaluate_on_domain


__all__ = ["RvMax"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_rv_max_reads_the_curve_at_the_gas_pressure():
        model = toy_model("WOG", nx=2)
        model.fluid.rvSat = lambda p: 1e-9*p
        state = toy_state(model, pressure=np.array([1e5, 2e5]))
        model.get_prop(state, "PhasePressures")
        prop = RvMax(model)
        prop.struct_name = model.PVTPropertyFunctions.struct_name
        rv = prop.evaluate_on_domain(model, state)
        np.testing.assert_allclose(rv, [1e-4, 2e-4])
        print("ok test_rv_max_reads_the_curve_at_the_gas_pressure")

    def test_vappars_reduces_rv_with_oil_saturation_history():
        model = toy_model("WOG", nx=2)
        model.fluid.rvSat = lambda p: np.ones_like(np.asarray(p, dtype=float))
        s = np.array([[0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
        state = toy_state(model, s=s)
        state["sMax"] = np.array([[0.2, 0.8, 0.4], [0.2, 0.4, 0.4]])
        model.get_prop(state, "PhasePressures")
        prop = RvMax(model)
        prop.struct_name = model.PVTPropertyFunctions.struct_name
        prop.rvReduction = 1.0
        rv = prop.evaluate_on_domain(model, state)
        expected = (0.4 + VAPPARS_OFFSET)/(np.array([0.8, 0.4]) + VAPPARS_OFFSET)
        np.testing.assert_allclose(rv, expected)
        print("ok test_vappars_reduces_rv_with_oil_saturation_history")


    def test_negative_factor_takes_matlabs_real_part():
        # MATLAB 2024: real([-4 -2.5e-4 -0.3].^[0.5 0.5 0.7]) and the
        # derivative real(0.5*(-2.5e-4).^-0.5) are exactly these.
        from pymrst.utils.adi import init_variables_adi
        assert _matlab_real_power(np.array([-4.0, -2.5e-4]), 0.5).tolist() == [0.0, 0.0]
        assert _matlab_real_power(np.array([-0.3]), 0.7)[0] == -0.25304838132344892
        x, = init_variables_adi(np.array([-2.5e-4, 0.25]))
        y = _matlab_real_power(x, 0.5)
        assert value(y).tolist() == [0.0, 0.5]
        assert y.jac[0].toarray().diagonal().tolist() == [0.0, 1.0]
        print("ok test_negative_factor_takes_matlabs_real_part")

    def test_non_negative_factor_is_the_plain_power():
        f = np.array([0.0, 0.3, 1.7])
        assert np.array_equal(_matlab_real_power(f, 0.7), f**0.7)
        assert np.array_equal(_matlab_real_power(np.array([-0.3]), 2.0), np.array([0.09]))
        print("ok test_non_negative_factor_is_the_plain_power")

    test_rv_max_reads_the_curve_at_the_gas_pressure()
    test_negative_factor_takes_matlabs_real_part()
    test_non_negative_factor_is_the_plain_power()
    test_vappars_reduces_rv_with_oil_saturation_history()
    print("\nAll tests passed!")
