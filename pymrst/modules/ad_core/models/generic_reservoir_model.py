"""GenericReservoirModel (MRST ad-core/models).

Ports MRST's `GenericReservoirModel.m`. In MRST this is a *marker* class - it
carries no code of its own; a concrete model (`GenericBlackOilModel`,
`GenericOverallCompositionModel`, ...) mixes it in to declare "my equations are
assembled generically from the state-function graph and my components, not from
a hand-written residual." The generic assembly itself is the reservoir model's
`conservation_equations`, which reads `ComponentTotalFlux` /
`ComponentTotalMass` off the graph for *whatever* components the model lists.

PyMRST's `ReservoirModel` already assembles this way (its
`conservation_equations` is component-generic). This class makes that explicit:
it is the base a graph-assembled model inherits, adds the marker
`is_generic_reservoir_model`, and exposes `get_model_equations` as the named
generic-assembly entry point MRST models call. `GenericBlackOilModel` is made
an instance of it, so the black-oil model *is* a generic reservoir model - the
same is-a relationship MRST declares.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.statefunctions.state_function import State


class GenericReservoirModel(ReservoirModel):
    """A reservoir model whose equations come from the state-function graph."""

    is_generic_reservoir_model: bool = True

    def get_model_equations(self, state0: State, state: State,
                            dt: float) -> List[Any]:
        """The component conservation equations, assembled generically from the
        flow-discretization grouping and the model's components (MRST's
        `getModelEquations` reservoir contribution)."""
        return self.conservation_equations(state0, state, dt)


__all__ = ["GenericReservoirModel"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    # import via the package path so the class identity matches the base the
    # black-oil model inherits (running this file as __main__ makes a second,
    # distinct copy of the class defined above)
    from pymrst.modules.ad_core.models.generic_reservoir_model import (
        GenericReservoirModel as GenericReservoirModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model_and_states():
        g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
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
        st = model.init_state_ad(st0)
        return model, st, model.init_state_ad(st0)

    def test_black_oil_is_a_generic_reservoir_model():
        """GenericBlackOilModel is-a GenericReservoirModel (the MRST marker)."""
        model, _, _ = _model_and_states()
        assert isinstance(model, GenericReservoirModel)
        assert model.is_generic_reservoir_model is True
        print("ok test_black_oil_is_a_generic_reservoir_model")

    def test_generic_assembly_matches_conservation_equations():
        """get_model_equations is the generic graph assembly - identical to the
        model's conservation_equations, component for component."""
        model, st, st0 = _model_and_states()
        a = model.get_model_equations(st0, st, 30.0 * DAY)
        b = model.conservation_equations(st0, st, 30.0 * DAY)
        assert len(a) == len(b)
        for i in range(len(a)):
            assert np.allclose(value(a[i]), value(b[i]))
            assert np.allclose((a[i] - b[i]).jac[0].toarray(), 0.0)
        print("ok test_generic_assembly_matches_conservation_equations")

    test_black_oil_is_a_generic_reservoir_model()
    test_generic_assembly_matches_conservation_equations()
    print("\nAll tests passed!")
