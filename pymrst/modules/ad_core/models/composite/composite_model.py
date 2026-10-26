"""CompositeModel (MRST ad-core/models/composite) - core.

Ports the core of MRST's `CompositeModel.m` (AC12): a model built from several
*submodels*, each keeping its own state and equations, with `CouplingTerm`s
linking them. It is the general form of what `ad_mechanics` does by hand -
flow coupled to geomechanics - expressed as composable state functions rather
than a bespoke coupled residual.

The composite state is per-submodel: ``state[name]`` is submodel ``name``'s own
state. `get_model_equations` asks each submodel for its equations on its
sub-state, then applies every coupling term (each a conservative source between
submodels). The assembled system is the concatenation of the submodel
equations, coupled.

Scope of this core: submodel composition, the per-submodel equation assembly,
and conservative coupling-term insertion. The full MRST `CompositeModel`
additionally packs all submodel primary variables into one Newton system,
splits the linearized problem, and runs the coupled solve - and, crucially,
its headline use (flow + geomechanics) needs geomechanics expressed as a
state-function grouping, which PyMRST's hand-coupled `ad_mechanics` is not. So
the coupled *solve* against `ad_mechanics` is a documented follow-on; what is
validated here is the composition and the conservation of the coupling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.statefunctions.composite.coupling_term import (
    CouplingTerm,
)


class CompositeModel(PhysicalModel):
    """A model assembled from named submodels linked by coupling terms."""

    def __init__(self, submodels: Dict[str, Any]):
        super().__init__()
        assert submodels, "a composite model needs at least one submodel"
        self.submodels: Dict[str, Any] = dict(submodels)
        self.coupling_terms: Dict[str, CouplingTerm] = {}

    def get_model_names(self) -> List[str]:
        return list(self.submodels)

    def num_models(self) -> int:
        return len(self.submodels)

    def set_coupling_term(self, coupling: CouplingTerm, name: str
                          ) -> "CompositeModel":
        """Register a coupling term under ``name``."""
        self.coupling_terms[name] = coupling
        return self

    def get_active_models(self, state: Any) -> List[str]:
        return [m for m in self.submodels if m in state]

    def get_model_equations(self, state0: Any, state: Any, dt: float,
                            sources: Optional[Dict[str, Any]] = None):
        """Per-submodel equations, coupled.

        Returns ``(eqs, names)`` keyed by submodel name. ``sources`` optionally
        maps a coupling name to its evaluated source ``q``; a coupling with no
        supplied source is skipped.
        """
        active = self.get_active_models(state)
        eqs: Dict[str, List[Any]] = {}
        names: Dict[str, List[str]] = {}
        for mname in active:
            sub = self.submodels[mname]
            eqs[mname] = sub.get_model_equations(state0[mname], state[mname], dt)
            names[mname] = sub.get_component_names()

        sources = sources or {}
        for cname, ct in self.coupling_terms.items():
            if cname not in sources:
                continue
            eqs = ct.insert_coupling(self, eqs, names, sources[cname])
        return eqs, names

    # -- MRST CompositeModel.m, ported ------------------------------------

    def set_primary_variables(self, state: Any, names) -> Any:
        """Record each submodel's primary-variable names on its own state.

        Recurses into nested composites, because a submodel may itself be one -
        writing the names at the top level only would leave the inner model
        unable to split its own increments.
        """
        active = self.get_active_models(state)
        if len(names) != len(active):
            raise ValueError(
                f"{len(names)} name lists for {len(active)} active submodels "
                f"{active}")
        for mname, sub_names in zip(active, names):
            submodel = self.submodels[mname]
            if isinstance(submodel, CompositeModel):
                state[mname] = submodel.set_primary_variables(state[mname],
                                                              sub_names)
            else:
                state[mname]["primaryVariables"] = list(sub_names)
        return state

    setPrimaryVariables = set_primary_variables

    def split_problem(self, problem, remove_couplings: bool = False):
        """Split a coupled problem into one problem per submodel.

        Equations are assigned by the `"<submodel>."` prefix on their names,
        and the prefix is stripped on the way out so each subproblem reads as
        though it had been assembled alone.

        `remove_couplings` additionally drops the off-diagonal Jacobian blocks,
        which turns the split into a set of *independent* problems. That is a
        different thing from splitting - the coupled system's answer is not the
        concatenation of the decoupled ones - so it is off by default.
        """
        from pymrst.modules.ad_core.solvers.linear_solver_ad import (
            LinearizedProblem,
        )

        names = self.get_active_models(problem.state)
        problems, mapping = [], [None]*problem.num_eq()

        for index, mname in enumerate(names):
            prefix = f"{mname}."
            selected = [i for i, n in enumerate(problem.equation_names)
                        if str(n).startswith(prefix)]
            stripped = [str(problem.equation_names[i])[len(prefix):]
                        for i in selected]
            sub = LinearizedProblem.from_equations(
                [problem.equations[i] for i in selected],
                [problem.types[i] for i in selected],
                stripped,
                [problem.primary_variables[i] for i in selected
                 if i < len(problem.primary_variables)],
                problem.state[mname] if mname in problem.state else None,
                problem.dt)
            for i in selected:
                mapping[i] = index
            if remove_couplings:
                for equation in sub.equations:
                    if hasattr(equation, "jac"):
                        equation.jac = [equation.jac[i] for i in selected
                                        if i < len(equation.jac)]
            problems.append(sub)
        return problems, mapping

    splitProblem = split_problem

    def flatten_equations(self, eqs: Dict[str, List[Any]]) -> List[Any]:
        """The submodel equation lists concatenated, in submodel order."""
        out: List[Any] = []
        for mname in self.submodels:
            if mname in eqs:
                out.extend(eqs[mname])
        return out

    # -- the coupled Newton system -----------------------------------------
    # Each submodel assembles its own residual and Jacobian in its own AD
    # context (a diagonal block); the coupling terms contribute the off-diagonal
    # blocks that make the submodels solve *together*. This is what turns the
    # composition above into a system a Newton step can advance.

    def submodel_system(self, mname: str, state0: Any, state: Any, dt: float):
        """One submodel's residual vector and (square) Jacobian block."""
        import numpy as np

        from pymrst.utils.adi import combine_equations
        sub = self.submodels[mname]
        eqs = sub.get_equations(state0, state, dt)
        eq = combine_equations(*eqs)
        return np.asarray(eq.val, float), eq.jac[0]

    def assemble_coupled_system(self, states0: Dict[str, Any],
                                states: Dict[str, Any], dt: float,
                                couplings: Optional[List[Any]] = None):
        """Global coupled residual and Jacobian over all submodels.

        Returns ``(residual, jacobian, layout)`` where ``residual`` is the
        stacked submodel residuals with coupling sources added, ``jacobian`` is
        the block system (submodel diagonals + coupling off-diagonals), and
        ``layout`` maps each submodel to its ``(row0, ncell)`` offset.
        """
        import numpy as np
        import scipy.sparse as sp

        active = self.get_active_models(states)
        r_blocks, j_blocks, offsets = [], {}, {}
        row = 0
        for m in active:
            r_i, j_ii = self.submodel_system(m, states0[m], states[m], dt)
            offsets[m] = (row, self.submodels[m].G.cells.num, r_i.size)
            r_blocks.append(r_i)
            j_blocks[m] = j_ii
            row += r_i.size
        total = row

        residual = np.concatenate(r_blocks)
        J = sp.lil_matrix((total, total))
        for m in active:
            r0, _, sz = offsets[m]
            J[r0:r0 + sz, r0:r0 + sz] = j_blocks[m].tocsr()

        for cpl in (couplings or []):
            cpl.apply(self, states, residual, J, offsets)

        return residual, J.tocsr(), offsets


__all__ = ["CompositeModel"]


__all__ = ["CompositeModel"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.statefunctions.composite.coupling_term import (
        SourceCouplingTerm,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _reservoir():
        g = compute_geometry(cart_grid([4, 3, 1], [200.0, 150.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModelAD(g, rock, fluid, w)
        st0 = State(pressure=np.full(g.cells.num, 200 * BARSA),
                    sW=np.full(g.cells.num, 0.4),
                    qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                    bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st0), model.init_state_ad(st0)

    def test_single_submodel_reproduces_standalone():
        """A one-submodel composite yields the submodel's own equations."""
        sub, st, st0 = _reservoir()
        comp = CompositeModel({"res": sub})
        eqs, _ = comp.get_model_equations({"res": st0}, {"res": st}, 30.0 * DAY)
        ref = sub.conservation_equations(st0, st, 30.0 * DAY)
        for i in range(len(ref)):
            assert np.allclose(value(eqs["res"][i]), value(ref[i]))
        print("ok test_single_submodel_reproduces_standalone")

    def test_two_submodels_concatenate():
        """With no coupling, the composite equations are both submodels'."""
        subA, stA, st0A = _reservoir()
        subB, stB, st0B = _reservoir()
        comp = CompositeModel({"A": subA, "B": subB})
        eqs, _ = comp.get_model_equations(
            {"A": st0A, "B": st0B}, {"A": stA, "B": stB}, 30.0 * DAY)
        flat = comp.flatten_equations(eqs)
        assert len(flat) == len(subA.get_component_names()) + \
            len(subB.get_component_names())
        refA = subA.conservation_equations(st0A, stA, 30.0 * DAY)
        assert np.allclose(value(eqs["A"][0]), value(refA[0]))
        print("ok test_two_submodels_concatenate")

    def test_coupling_conserves_total_mass():
        """A source coupling A->B leaves the summed residual unchanged: what
        the coupling removes from A it adds to B (conservative)."""
        subA, stA, st0A = _reservoir()
        subB, stB, st0B = _reservoir()
        comp = CompositeModel({"A": subA, "B": subB})
        comp.set_coupling_term(SourceCouplingTerm(comp, "A", "B"), "AtoB")

        state = {"A": stA, "B": stB}
        state0 = {"A": st0A, "B": st0B}
        # an inter-model transfer q per component (here a fixed test rate)
        nc = subA.G.cells.num
        q = [np.full(nc, 1e-3), np.full(nc, 2e-3)]

        eqs0, _ = comp.get_model_equations(state0, state, 30.0 * DAY)
        summed_before = [value(eqs0["A"][i]) + value(eqs0["B"][i])
                         for i in range(2)]
        eqs1, _ = comp.get_model_equations(state0, state, 30.0 * DAY,
                                           sources={"AtoB": q})
        for i in range(2):
            summed_after = value(eqs1["A"][i]) + value(eqs1["B"][i])
            assert np.allclose(summed_after, summed_before[i])
            # A lost q, B gained q
            assert np.allclose(value(eqs1["A"][i]) - value(eqs0["A"][i]), -q[i])
            assert np.allclose(value(eqs1["B"][i]) - value(eqs0["B"][i]), +q[i])
        print("ok test_coupling_conserves_total_mass")

    test_single_submodel_reproduces_standalone()
    test_two_submodels_concatenate()
    test_coupling_conserves_total_mass()
    print("\nAll tests passed!")


# -- MRST's module-local variable tree helpers ------------------------------
#
# A composite model's variables arrive as a *nested* structure - a list per
# submodel, and a submodel that is itself composite contributes another list.
# The AD layer wants one flat list, and afterwards the results have to go back
# into the same shape. These two do that, and the property that matters is that
# they are inverses: `pack(unpack(x)) == x` for any nesting.


def unpack_vars(variables, level: int = 1, num: int = 1,
                levels=None, nodes=None):
    """Flatten a nested variable tree, recording where each leaf came from.

    Returns `(flat, levels, nodes)`. The two index lists are what make the
    flattening reversible - without them a flat list cannot say which submodel,
    at which depth, each entry belonged to.
    """
    levels = [] if levels is None else levels
    nodes = [] if nodes is None else nodes
    flat = []
    for index, item in enumerate(variables, start=1):
        if isinstance(item, (list, tuple)):
            inner, levels, nodes = unpack_vars(item, level + 1, index,
                                               levels, nodes)
            flat += inner
        else:
            flat.append(item)
            levels.append(level)
            nodes.append(num)
    return flat, levels, nodes


unpackVars = unpack_vars


def pack_vars(flat, levels, nodes, level: int | None = None):
    """Rebuild the nested tree that `unpack_vars` flattened.

    `level` defaults to the shallowest level present. MRST hard-codes 2 at its
    one call site, which works because a composite model's variables are
    uniformly nested one level deep; deriving it here means the round trip
    holds without the caller having to know the depth.

    A tree of *mixed* depth is refused rather than silently truncated. MRST
    takes the node count from the current level only - `max(mapn(subsetl))` -
    so a leaf sitting beside a nested list is dropped without a word. Checking
    that every leaf was consumed turns that into an error.
    """
    import numpy as _np

    levels = _np.asarray(levels)
    nodes = _np.asarray(nodes)
    if level is None:
        if levels.size == 0:
            return []
        level = int(levels.min())
        packed = _pack_at(flat, levels, nodes, level)
        if _leaf_count(packed) != len(flat):
            raise ValueError(
                f"variable tree mixes nesting depths ({sorted(set(levels.tolist()))}); "
                f"packing kept {_leaf_count(packed)} of {len(flat)} entries. "
                f"MRST drops the remainder silently; this refuses instead")
        return packed
    return _pack_at(flat, levels, nodes, level)


def _leaf_count(tree) -> int:
    if isinstance(tree, (list, tuple)):
        return sum(_leaf_count(t) for t in tree)
    return 1


def _pack_at(flat, levels, nodes, level: int):
    import numpy as _np

    levels = _np.asarray(levels)
    nodes = _np.asarray(nodes)
    here = levels == level
    if not here.any():
        return list(flat)

    count = int(nodes[here].max())
    at_level = [v for v, keep in zip(flat, here) if keep]
    deeper = [v for v, keep in zip(flat, here) if not keep]

    out = []
    for node in range(1, count + 1):
        mask = nodes[here] == node
        if mask.any():
            out.append([v for v, keep in zip(at_level, mask) if keep])
        else:
            out.append(_pack_at(deeper, levels[~here], nodes[~here],
                                level + 1))
    return out


packVars = pack_vars
