"""PermeabilityPotentialGradient (MRST ad-core/statefunctions/flux).

Ports MRST's `PermeabilityPotentialGradient.m`: the face value of ``K`` times
the phase potential difference, one entry per phase. It holds a
`PermeabilityGradientDiscretization` (TPFA by default) and defers the actual
``K grad`` stencil to it.

This is the state function that sits between `PhasePotentialDifference` and
`PhaseFlux` in MRST's flux graph. PyMRST's current `PhaseFlux` folds the
``T * pot`` product into itself; expressing it as its own state function is
what lets an MPFA or mimetic discretization be dropped in without touching the
flux group. The Darcy flux is then ``-lambda_f * (K grad Phi)`` -
face mobility times this quantity.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.flux.permeability_gradient_discretization import (  # noqa: E501
    PermeabilityGradientDiscretization,
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    StateFunction,
)


class PermeabilityPotentialGradient(StateFunction):
    """``K grad(Phi_alpha)`` on interior faces, per phase."""

    def __init__(self, model: Any = None,
                 kgrad: PermeabilityGradientDiscretization = None):
        super().__init__(model, label="T_f \\Theta_\\alpha")
        self.kgrad = kgrad if kgrad is not None else TwoPointFluxApproximation(model)
        self.depends_on("PhasePotentialDifference")
        if isinstance(self.kgrad, TwoPointFluxApproximation):
            self.depends_on("Transmissibility")

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        pot = self.get_evaluated(state, "PhasePotentialDifference")
        return [self.kgrad.get_permeability_gradient(model, state, p)
                for p in pot]


__all__ = ["PermeabilityPotentialGradient"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _build_ow():
        """The oil-water quarter-spot of validate_phase5/24, with gravity on
        so the potential difference is not just the pressure gradient."""
        g = compute_geometry(cart_grid([6, 5, 3], [300.0, 250.0, 30.0]))
        kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
        rock = make_rock(g, kvals[:, None], 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModelAD(g, rock, fluid, w, gravity=9.80665)
        st = State(pressure=np.full(g.cells.num, 200 * BARSA),
                   sW=np.full(g.cells.num, 0.35),
                   qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                   bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st)

    def _register_kpg(model):
        from pymrst.modules.ad_core.statefunctions.flux.permeability_potential_gradient import (  # noqa: E501
            PermeabilityPotentialGradient as _KPG,
        )
        model.FlowDiscretization.set_state_function(
            "PermeabilityPotentialGradient", _KPG(model))

    def test_kpg_equals_T_times_potential():
        """K grad Phi = T .* PhasePotentialDifference, val and Jacobian."""
        model, st = _build_ow()
        _register_kpg(model)
        pot = model.get_props(st, "PhasePotentialDifference")
        t = model.operators.T
        kpg = model.get_props(st, "PermeabilityPotentialGradient")
        for ph in range(model.n_phases):
            assert np.allclose(value(kpg[ph]), t * value(pot[ph]))
            assert np.allclose((kpg[ph] - t * pot[ph]).jac[0].toarray(), 0.0)
        print("ok test_kpg_equals_T_times_potential")

    def test_phaseflux_is_facemob_times_kpg():
        """The validated PhaseFlux equals -FaceMobility * (K grad Phi).

        This is the equivalence that lets the new, discretization-agnostic
        state function replace the hard-coded ``-T*mob*pot`` in PhaseFlux
        without moving a single validated number.
        """
        model, st = _build_ow()
        _register_kpg(model)
        flux = model.get_props(st, "PhaseFlux")
        mobf = model.get_props(st, "FaceMobility")
        kpg = model.get_props(st, "PermeabilityPotentialGradient")
        for ph in range(model.n_phases):
            recomposed = -mobf[ph] * kpg[ph]
            assert np.allclose(value(flux[ph]), value(recomposed))
            assert np.allclose((flux[ph] - recomposed).jac[0].toarray(), 0.0,
                               atol=1e-20)
        print("ok test_phaseflux_is_facemob_times_kpg")

    def test_cache_shares_transmissibility():
        """Evaluating K grad reuses the cached Transmissibility, not a fresh
        one - the whole reason it is a graph node."""
        model, st = _build_ow()
        _register_kpg(model)
        model.get_props(st, "PermeabilityPotentialGradient")
        assert model.FlowDiscretization.is_evaluated(st, "Transmissibility")
        assert model.FlowDiscretization.is_evaluated(
            st, "PhasePotentialDifference")
        print("ok test_cache_shares_transmissibility")

    test_kpg_equals_T_times_potential()
    test_phaseflux_is_facemob_times_kpg()
    test_cache_shares_transmissibility()
    print("\nAll tests passed!")
