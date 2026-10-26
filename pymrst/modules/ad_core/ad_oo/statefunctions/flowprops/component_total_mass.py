"""MRST ``ad-core/statefunctions/flowprops/ComponentTotalMass.m``.

Total mass of each component, summed over the phases it occupies, with an
optional floor on the diagonal of each component's own Jacobian block - MRST's
guard against a component whose mass has (numerically) no derivative with
respect to its own variable.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction

DEFAULT_MINIMUM_DERIVATIVE: float = 1e-10


class ComponentTotalMass(StateFunction):
    """`M_i = sum_alpha M_{i,alpha}`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="M_i", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.minimumDerivatives: Optional[Any] = None
        self.depends_on("ComponentPhaseMass")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        phase_mass = self.get_evaluated_dependencies(state, "ComponentPhaseMass")
        mass: List[Any] = []
        for row in phase_mass:
            total = None
            for m in row:
                if m is not None:
                    total = m if total is None else total + m
            mass.append(total)
        return self.ensure_minimum_derivatives(mass)

    def set_minimum_derivatives(self, der: float = DEFAULT_MINIMUM_DERIVATIVE):
        self.minimumDerivatives = der
        return self

    def get_minimum_derivatives(self):
        return self.minimumDerivatives

    def ensure_minimum_derivatives(self, mass: List[Any]) -> List[Any]:
        """Raise `|dM_c/dx_c|` to the floor where it falls below it.

        Only the sparse backend's path: PyMRST has no diagonal AD backend, so
        MRST's diagonal branch has nothing to act on.
        """
        der = self.minimumDerivatives
        if der is None:
            return mass
        der = np.broadcast_to(np.atleast_1d(der), (len(mass),))
        out = list(mass)
        for c, m in enumerate(mass):
            if not hasattr(m, "jac"):
                continue
            if len(m.jac) <= c:
                break
            J = m.jac[c]
            n, l = J.shape
            if n != l:
                continue
            bad = np.abs(J.diagonal()) < der[c]
            if np.any(bad):
                m = m + 0.0
                m.jac[c] = (J + sp.diags(bad*der[c])).tocsr()
                out[c] = m
        return out

    evaluateOnDomain = evaluate_on_domain
    setMinimumDerivatives = set_minimum_derivatives
    getMinimumDerivatives = get_minimum_derivatives
    ensureMinimumDerivatives = ensure_minimum_derivatives


__all__ = ["ComponentTotalMass"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.utils.adi import init_variables_adi

    def _prop():
        return ComponentTotalMass(SimpleNamespace(AutoDiffBackend=None))

    def test_mass_sums_over_occupied_phases():
        prop = _prop()
        mass = prop.ensure_minimum_derivatives([np.array([1.0]) + np.array([2.0])])
        np.testing.assert_allclose(mass[0], 3.0)
        print("ok test_mass_sums_over_occupied_phases")

    def test_minimum_derivative_floors_the_own_block_diagonal():
        x, y = init_variables_adi(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        prop = _prop().set_minimum_derivatives(0.5)
        flat = 0.0*x + y
        mass = prop.ensure_minimum_derivatives([flat])
        np.testing.assert_allclose(mass[0].jac[0].diagonal(), 0.5)
        np.testing.assert_allclose(flat.jac[0].diagonal(), 0.0)
        print("ok test_minimum_derivative_floors_the_own_block_diagonal")

    test_mass_sums_over_occupied_phases()
    test_minimum_derivative_floors_the_own_block_diagonal()
    print("\nAll tests passed!")
