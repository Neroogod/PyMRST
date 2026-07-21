"""Coupling terms (MRST ad-core/statefunctions/composite).

Ports MRST's `CouplingTerm.m` and `SourceCouplingTerm.m` (AC9): the state
functions a `CompositeModel` uses to link its submodels. A coupling term
evaluates a source/sink ``q`` and *inserts* it into the equations of the
submodels it couples - with a sign per submodel, so mass leaving one enters the
other and the total is conserved.

A `CouplingTerm` carries a list of ``couplings``, one per submodel it touches:
each names the submodel, the equations (by component name) to add into, a sign
(``-1`` sink / ``+1`` source), and an optional cell subset. `insert_coupling`
walks that list and adds ``sign * q`` into the matching equations.

`SourceCouplingTerm` is the common case: a transfer between two submodels that
share component names - it removes the transfer from ``modelA`` and adds it to
``modelB``, which is exactly a conservative inter-model source. `q` itself is
supplied by the composite (a rate, e.g. proportional to an inter-model
potential difference), so the term is a pure, conservative redistribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pymrst.utils.adi import value as _value


class StateFunctionDependent:
    """Marker mixin (MRST `StateFunctionDependent`): a state function whose
    value is derived from other state functions rather than the raw state.
    Carried for MRST parity; coupling terms are its main use here."""


class CouplingTerm(StateFunctionDependent):
    """Base coupling: a source ``q`` inserted into named submodel equations."""

    def __init__(self, couplings: Optional[List[Dict[str, Any]]] = None,
                 label: str = "q_{coupling}"):
        self.couplings: List[Dict[str, Any]] = list(couplings or [])
        self.label = label

    def evaluate_source(self, model: Any, state: Any) -> Any:
        """The coupling source ``q``. Subclasses / the composite provide it;
        by default it is read off the composite state under ``self`` label."""
        raise NotImplementedError("coupling source must be supplied")

    def insert_coupling(self, model: Any,
                        eqs: Dict[str, List[Any]],
                        names: Dict[str, List[str]],
                        q: Any) -> Dict[str, List[Any]]:
        """Add ``sign * q`` into each coupled submodel's matching equations.

        ``eqs`` / ``names`` are keyed by submodel name; ``q`` is either one
        source (applied to every listed equation) or a list, one per equation.
        """
        for cpl in self.couplings:
            mname = cpl["model"]
            if mname not in eqs:
                continue
            eq_names = names[mname]
            sign = cpl.get("sign", 1)
            subset = cpl.get("subset", None)
            wanted = cpl["equations"]
            for i, comp in enumerate(wanted):
                if comp not in eq_names:
                    continue
                eq_ix = eq_names.index(comp)
                qi = q[i] if isinstance(q, (list, tuple)) else q
                eqs[mname][eq_ix] = _add_source(eqs[mname][eq_ix], sign * qi,
                                                subset)
        return eqs


class SourceCouplingTerm(CouplingTerm):
    """A conservative transfer between two submodels that share components.

    Removes the source from ``model_a`` (sign -1) and adds it to ``model_b``
    (sign +1). The two submodels must share the component names being coupled.
    """

    def __init__(self, model: Any, model_a: str, model_b: str,
                 components: Optional[Sequence[str]] = None,
                 cells_a: Any = None, cells_b: Any = None):
        names_a = model.submodels[model_a].get_component_names()
        names_b = model.submodels[model_b].get_component_names()
        shared = list(components) if components is not None else \
            [n for n in names_a if n in names_b]
        assert shared, "submodels share no component names to couple"
        assert all(c in names_a and c in names_b for c in shared), \
            "coupled components must be present in both submodels"
        couplings = [
            {"model": model_a, "equations": shared, "subset": cells_a,
             "sign": -1},
            {"model": model_b, "equations": shared, "subset": cells_b,
             "sign": +1},
        ]
        super().__init__(couplings, label="q_{A\\rightarrow B}")


class LinearPressureCoupling:
    """A conservative inter-model transfer ``q = coeff * (p_A - p_B)`` per cell.

    Physically a leaky connection between two co-located reservoir submodels
    (a fault, a shared aquifer): each shared component flows from the higher-
    pressure model to the lower at a rate linear in the pressure difference.
    The transfer is added as a sink on ``model_a`` and an equal source on
    ``model_b`` - conservative - and it couples the two Newton blocks through
    pressure, with the analytic derivative ``+/- coeff``.

    This is a concrete `CouplingTerm` used to drive and validate the composite
    coupled solve; the derivative is exact, so no reference is needed.
    """

    def __init__(self, model_a: str, model_b: str, coeff: Any,
                 components: Optional[Sequence[str]] = None):
        self.model_a = model_a
        self.model_b = model_b
        self.coeff = coeff

    def apply(self, model: Any, states: Dict[str, Any], residual: np.ndarray,
              jac: Any, offsets: Dict[str, Any]) -> None:
        import scipy.sparse as sp

        a, b = self.model_a, self.model_b
        r0a, nc, _ = offsets[a]
        r0b, _, _ = offsets[b]
        subA = model.submodels[a]
        pa = np.asarray(_value(states[a]["pressure"]), float)
        pb = np.asarray(_value(states[b]["pressure"]), float)
        coeff = np.broadcast_to(np.asarray(self.coeff, float), (nc,))
        q = coeff * (pa - pb)                          # per-cell transfer

        ncomp = subA.n_phases                          # one conservation eq/comp
        d = sp.diags(coeff, shape=(nc, nc))
        neg_d = sp.diags(-coeff, shape=(nc, nc))
        for c in range(ncomp):
            ra = r0a + c * nc
            rb = r0b + c * nc
            # residual: sink from A, source into B
            residual[ra:ra + nc] -= q
            residual[rb:rb + nc] += q
            # pressure columns: first nc columns of each submodel block
            ca, cb = r0a, r0b
            # d(-q)/dpA = -coeff ; d(-q)/dpB = +coeff  (rows of A)
            jac[ra:ra + nc, ca:ca + nc] += neg_d
            jac[ra:ra + nc, cb:cb + nc] += d
            # d(+q)/dpA = +coeff ; d(+q)/dpB = -coeff  (rows of B)
            jac[rb:rb + nc, ca:ca + nc] += d
            jac[rb:rb + nc, cb:cb + nc] += neg_d


def _add_source(eq: Any, q: Any, subset: Any) -> Any:
    """``eq + q`` on ``subset`` cells (all cells if ``subset`` is None)."""
    if subset is None:
        return eq + q
    n = np.asarray(eq if not hasattr(eq, "val") else eq.val).size
    scatter = np.zeros(n)
    idx = np.asarray(subset, int)
    qv = q if np.ndim(q) else np.full(idx.size, q)
    # build a full-length source vector, non-zero only on the subset
    if hasattr(q, "val"):
        full = q * 0.0                                # AD zeros of q's shape
        # q is defined on the subset; place it
        # (kept simple: subset couplings use per-cell q already sized to subset)
        return _scatter_add(eq, q, idx, n)
    scatter[idx] = qv
    return eq + scatter


def _scatter_add(eq: Any, q: Any, idx: np.ndarray, n: int) -> Any:
    """Add AD ``q`` (defined on ``idx``) into full-length ``eq``."""
    import scipy.sparse as sp
    from pymrst.utils.adi import ad_matmul
    scatter = sp.csr_array((np.ones(idx.size), (idx, np.arange(idx.size))),
                           shape=(n, idx.size))
    return eq + ad_matmul(scatter, q)


__all__ = ["CouplingTerm", "LinearPressureCoupling", "SourceCouplingTerm",
           "StateFunctionDependent"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.utils.adi import init_variables_adi, value

    class _Sub:
        def __init__(self, names):
            self._names = names

        def get_component_names(self):
            return self._names

    class _Composite:
        def __init__(self):
            self.submodels = {"A": _Sub(["water", "oil"]),
                              "B": _Sub(["water", "oil"])}

    def test_source_term_signs_are_opposite():
        """A source coupling removes from A and adds to B (conservative)."""
        m = _Composite()
        ct = SourceCouplingTerm(m, "A", "B")
        signs = {c["model"]: c["sign"] for c in ct.couplings}
        assert signs == {"A": -1, "B": +1}
        assert ct.couplings[0]["equations"] == ["water", "oil"]
        print("ok test_source_term_signs_are_opposite")

    def test_insert_coupling_conserves_total():
        """After insertion, eq_A + eq_B is unchanged (q leaves A, enters B)."""
        m = _Composite()
        ct = SourceCouplingTerm(m, "A", "B")
        (qw, qo) = init_variables_adi(np.array([1.0, 2.0, 3.0]),
                                      np.array([0.5, 0.5, 0.5]))
        eqs = {"A": [qw * 0 + 10.0, qo * 0 + 20.0],
               "B": [qw * 0 + 1.0, qo * 0 + 2.0]}
        names = {"A": ["water", "oil"], "B": ["water", "oil"]}
        before = [value(eqs["A"][i]) + value(eqs["B"][i]) for i in range(2)]
        out = ct.insert_coupling(m, eqs, names, [qw, qo])
        for i in range(2):
            after = value(out["A"][i]) + value(out["B"][i])
            assert np.allclose(after, before[i]), "coupling must conserve total"
        # and A lost exactly q, B gained exactly q
        assert np.allclose(value(out["A"][0]), 10.0 - value(qw))
        assert np.allclose(value(out["B"][0]), 1.0 + value(qw))
        print("ok test_insert_coupling_conserves_total")

    def test_requires_shared_components():
        class _M:
            submodels = {"A": _Sub(["water"]), "B": _Sub(["gas"])}
        try:
            SourceCouplingTerm(_M(), "A", "B")
            raise SystemExit("should reject models with no shared components")
        except AssertionError:
            pass
        print("ok test_requires_shared_components")

    test_source_term_signs_are_opposite()
    test_insert_coupling_conserves_total()
    test_requires_shared_components()
    print("\nAll tests passed!")
