"""MRST ``ad-core/statefunctions/flowprops/NumericalPressureReductionFactors.m``.

Pressure-reduction weights computed from the accumulation terms, for any
model: with the reservoir's primary variables as AD, the weights `w` solve
`w*A = b`, where `A` is the Jacobian of `(mass - mass0)/dt` for all
components and `b` is `1/barsa` on the pressure columns and zero elsewhere,
so that the weighted accumulation depends on pressure alone. They need
`state.reductionFactorProps` (`mass0`, `dt` and the previous iteration's
`pressure` and `weights`), which `PressureModel` assigns for the
`'numerical'` strategy. When the pressure is AD and previous weights exist,
each weight gets a pressure derivative, the change in weights (rescaled by
the weighted mass) over the change in pressure.

With MRST's diagonal backend and its MEX enabled (`useDiagonalReduction`)
the cell-wise blocks are inverted one by one; otherwise one sparse system is
solved. The blocks are read here from each Jacobian's diagonal, whatever the
backend stores. With `checkDerivatives` a warning reports any non-pressure
derivative left in the weighted accumulation.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA

ZERO_PRESSURE_TOL = 1e-16
NONZERO_DERIVATIVE_TOL = 1e-8


class NumericalPressureReductionFactors(StateFunction):
    """MRST `NumericalPressureReductionFactors`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, regions=regions)
        self.includeDerivatives = True
        self.checkDerivatives = False
        self.rowMajor = False
        AD = model.AutoDiffBackend
        is_diag = type(AD).__name__ == "DiagonalAutoDiffBackend"
        self.useDiagonalReduction = is_diag and bool(AD.use_mex)
        if is_diag:
            self.rowMajor = bool(AD.row_major)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        state_ad, ncell, ncomp = self.get_state_ad(model, state)
        dt = state["reductionFactorProps"]["dt"]
        p = value(state["pressure"])
        mass = model.get_props(state_ad, "ComponentTotalMass")
        factor = dict(state["reductionFactorProps"])
        factor["mass"] = mass
        factor["pressure0"] = factor["pressure"]
        factor["pressure"] = value(p)
        w = self.get_weights(factor, dt, ncell, ncomp)
        w_prev = state["reductionFactorProps"]["weights"]
        pAD = state["pressure"]
        is_ad = hasattr(pAD, "jac")
        dwdp = None
        if is_ad and self.includeDerivatives:
            dwdp = self.get_weight_derivatives(w, w_prev, factor)
        has_weights = dwdp is not None and dwdp.size > 0
        weights = []
        for i in range(ncomp):
            Wp = w[:, i]
            if has_weights and np.any(dwdp[:, i]):
                Wp = model.AutoDiffBackend.convert_to_ad(Wp, pAD)
                Wp.jac[0] = sp.csr_array(sp.diags(dwdp[:, i], shape=(ncell, ncell)))
            weights.append(Wp)
        if self.checkDerivatives and is_ad:
            self._check_derivatives(model, mass, w, dwdp if has_weights else None, ncell)
        return weights

    def get_weights(self, factor: dict, dt: float, ncell: int, ncomp: int) -> np.ndarray:
        """The weights, `ncell` by `ncomp`."""
        acc = [(m - m0)/dt for m, m0 in zip(factor["mass"], factor["mass0"])]
        if all(not hasattr(a, "jac") for a in acc):
            return np.ones((ncell, ncomp))
        ndof = ncell*ncomp
        if self.useDiagonalReduction:
            return self._diagonal_weights(acc, ncell, ncomp)
        from pymrst.modules.ad_core.utils.combine_equations import combine_equations
        A = sp.csc_array(combine_equations(*acc).jac[0])
        _, jj, v = sp.find(A)
        ix = jj < ncell
        if not np.any(ix) or np.linalg.norm(v[ix], np.inf) < ZERO_PRESSURE_TOL:
            A = A + sp.csc_array((np.ones(ncell), (np.arange(ncell), np.arange(ncell))),
                                 shape=(ndof, ndof))
        b = np.zeros(ndof)
        b[:ncell] = 1/BARSA
        w = spsolve(A.T.tocsc(), b)
        return w.reshape(ncomp, ncell).T

    def _diagonal_weights(self, acc: List[Any], ncell: int, ncomp: int) -> np.ndarray:
        """MRST's `invv` route: each cell's `ncomp`-square block inverted."""
        blocks = np.stack([_diagonals(a, ncell) for a in acc], axis=2)
        b = np.zeros(ncomp)
        b[0] = 1/BARSA
        return np.einsum("cij,j->ci", np.linalg.inv(blocks), b)

    def get_state_ad(self, model: Any, state: Any) -> Tuple[Any, int, int]:
        """The state with only the pressure's model's variables as AD."""
        state = model.reduce_state(state, True)
        variables, names, origin = model.get_primary_variables(state)
        orig_p = origin[names.index("pressure")]
        local = [k for k, o in enumerate(origin) if o == orig_p]
        initialized = model.AutoDiffBackend.init_variables_ad(*[variables[k] for k in local])
        variables = list(variables)
        for k, var in zip(local, initialized):
            variables[k] = var
        state = model.init_state_ad(state, variables, names, origin)
        return state, model.G.cells.num, model.get_number_of_components()

    def get_weight_derivatives(self, w: np.ndarray, w0: Any, factor: dict) -> np.ndarray:
        """`dw/dp` from the change since the previous iteration, or empty."""
        if w0 is None or np.size(w0) == 0:
            return np.zeros((0, 0))
        w0 = np.asarray(w0, dtype=float)
        dp = np.asarray(factor["pressure"], dtype=float) - np.asarray(factor["pressure0"],
                                                                      dtype=float)
        m = np.column_stack([value(x) for x in factor["mass"]])
        a = np.sum(m*w, axis=1)/np.sum(m*w0, axis=1)
        dw = w - a[:, None]*w0
        with np.errstate(divide="ignore", invalid="ignore"):
            dwdp = dw/dp[:, None]
        dwdp[~np.isfinite(dwdp)] = 0
        return dwdp

    def _check_derivatives(self, model: Any, mass: List[Any], w: np.ndarray,
                           dwdp: Any, ncell: int) -> None:
        tmp = 0
        for i, m in enumerate(mass):
            Wp = w[:, i]
            if dwdp is not None and np.any(dwdp[:, i]):
                Wp = model.AutoDiffBackend.convert_to_ad(Wp, m)
                Wp.jac[0] = sp.csr_array(sp.diags(dwdp[:, i], shape=(ncell, Wp.jac[0].shape[1])))
            tmp = tmp + m*Wp
        scaler = np.mean(np.abs(_get_diagonal(tmp, 0)))
        for der_no in range(1, len(mass)):
            d = _get_diagonal(tmp, der_no)
            bad = np.abs(d) > NONZERO_DERIVATIVE_TOL*scaler
            if np.any(bad):
                warnings.warn(f"A_p with respect to variable {der_no + 1} has {int(bad.sum())} "
                              f"non-zero derivatives. Worst: {np.abs(d).max():g} "
                              f"Mean: {np.abs(d).mean():g}")

    evaluateOnDomain = evaluate_on_domain
    getWeights = get_weights
    getStateAD = get_state_ad
    getWeightDerivatives = get_weight_derivatives


def _get_diagonal(x: Any, i: int) -> np.ndarray:
    J = x.jac[i]
    return np.asarray(J.diagonal() if hasattr(J, "diagonal") else np.diag(J), dtype=float)


def _diagonals(acc: Any, ncell: int) -> np.ndarray:
    """`ncell` by `nvar`: each variable's derivative of a cell-wise term."""
    return np.column_stack([_get_diagonal(acc, k)[:ncell] for k in range(len(acc.jac))])


__all__ = ["NumericalPressureReductionFactors"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.utils.adi import init_variables_adi

    def _two_component_acc(p, s):
        """Water and oil mass in one-cell-per-entry form, `pv*rho*s`."""
        rho_w = 1000*(1 + 1e-9*(p - 1e7))
        rho_o = 800*(1 + 2e-9*(p - 1e7))
        return [rho_w*s, rho_o*(1 - s)]

    def _weights(p, s, mass0, **kw):
        prop = object.__new__(NumericalPressureReductionFactors)
        prop.useDiagonalReduction = kw.get("diagonal", False)
        prop.rowMajor = False
        pv, sv = init_variables_adi(p, s)
        factor = {"mass": _two_component_acc(pv, sv), "mass0": mass0}
        return prop, prop.get_weights(factor, 1.0, p.size, 2)

    def test_weighted_accumulation_has_no_saturation_derivative():
        p, s = np.array([1e7, 2e7, 1.5e7]), np.array([0.2, 0.5, 0.9])
        mass0 = [np.zeros(3), np.zeros(3)]
        _, w = _weights(p, s, mass0)
        pv, sv = init_variables_adi(p, s)
        acc = _two_component_acc(pv, sv)
        total = acc[0]*w[:, 0] + acc[1]*w[:, 1]
        assert np.allclose(total.jac[1].diagonal(), 0, atol=1e-20)
        assert np.allclose(total.jac[0].diagonal(), 1/BARSA)
        print("ok test_weighted_accumulation_has_no_saturation_derivative")

    def test_diagonal_route_matches_sparse_route():
        p, s = np.array([1e7, 2e7]), np.array([0.3, 0.6])
        mass0 = [np.zeros(2), np.zeros(2)]
        _, w_sparse = _weights(p, s, mass0)
        _, w_diag = _weights(p, s, mass0, diagonal=True)
        assert np.allclose(w_sparse, w_diag, rtol=1e-12)
        print("ok test_diagonal_route_matches_sparse_route")

    def test_numeric_accumulation_gives_unit_weights():
        prop = object.__new__(NumericalPressureReductionFactors)
        prop.useDiagonalReduction = False
        factor = {"mass": [np.ones(4), np.ones(4)], "mass0": [np.zeros(4), np.zeros(4)]}
        assert np.array_equal(prop.get_weights(factor, 1.0, 4, 2), np.ones((4, 2)))
        print("ok test_numeric_accumulation_gives_unit_weights")

    def test_weight_derivatives():
        prop = object.__new__(NumericalPressureReductionFactors)
        w0 = np.array([[1.0, 2.0], [1.0, 1.0]])
        w = np.array([[2.0, 4.0], [1.0, 3.0]])
        factor = {"mass": [np.ones(2), np.ones(2)], "pressure": np.array([2.0, 5.0]),
                  "pressure0": np.array([1.0, 5.0])}
        dwdp = prop.get_weight_derivatives(w, w0, factor)
        assert np.allclose(dwdp[0], [0.0, 0.0])
        assert np.array_equal(dwdp[1], [0.0, 0.0])
        assert prop.get_weight_derivatives(w, None, factor).size == 0
        print("ok test_weight_derivatives")

    def test_constructor_reads_the_backend():
        class DiagonalAutoDiffBackend:
            use_mex, row_major = True, True
        model = SimpleNamespace(AutoDiffBackend=DiagonalAutoDiffBackend())
        prop = NumericalPressureReductionFactors(model)
        assert prop.useDiagonalReduction and prop.rowMajor and prop.includeDerivatives
        print("ok test_constructor_reads_the_backend")

    test_weighted_accumulation_has_no_saturation_derivative()
    test_diagonal_route_matches_sparse_route()
    test_numeric_accumulation_gives_unit_weights()
    test_weight_derivatives()
    test_constructor_reads_the_backend()
    print("\nAll tests passed!")
