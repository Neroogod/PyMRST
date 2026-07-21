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

    def flatten_equations(self, eqs: Dict[str, List[Any]]) -> List[Any]:
        """The submodel equation lists concatenated, in submodel order."""
        out: List[Any] = []
        for mname in self.submodels:
            if mname in eqs:
                out.extend(eqs[mname])
        return out


__all__ = ["CompositeModel"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
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
        model = GenericBlackOilModel(g, rock, fluid, w)
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
