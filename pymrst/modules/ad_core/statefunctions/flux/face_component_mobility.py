"""FaceComponentMobility (MRST ad-core/statefunctions/flux).

Ports MRST's `FaceComponentMobility.m`: the mobile mass of each component in
each phase, upwinded to the interior faces. It is the component-aware analogue
of `FaceMobility` - the quantity a component flux reads, ``V_{i,alpha} =
FaceComponentMobility_{i,alpha} * (K grad Phi_alpha)``.

Deferred from AC1 because it needs both the upwind hierarchy (AC2's
`UpwindProperty` / `UpwindFunctionWrapperDiscretization`) and the cell-wise
`ComponentMobility` (AC6). With both landed it is a short port: upwind each
component/phase cell mobility with the phase's upwind flag, leaving the empty
(absent) entries empty.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    UpwindFunctionWrapperDiscretization,
    UpwindProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class FaceComponentMobility(StateFunction, UpwindProperty):
    """Upwinded per-component, per-phase mobility on interior faces."""

    def __init__(self, model: Any = None, upstream: Any = None,
                 upwind_name: str = "PhaseUpwindFlag"):
        StateFunction.__init__(self, model, label="\\lambda_{i,\\alpha}^f")
        if upstream is None:
            upstream = UpwindFunctionWrapperDiscretization(model)
        UpwindProperty.__init__(self, upstream)
        self.upwind_name = upwind_name
        self.depends_on(upwind_name)
        self.depends_on("ComponentMobility", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any
                           ) -> List[List[Optional[Any]]]:
        flag = self.get_evaluated(state, self.upwind_name)
        mob = self.get_evaluated_external(
            model, state, "FlowPropertyFunctions", "ComponentMobility")
        out: List[List[Optional[Any]]] = []
        for c in range(len(mob)):
            row: List[Optional[Any]] = []
            for ph in range(len(mob[c])):
                if mob[c][ph] is None:
                    row.append(None)
                else:
                    row.append(self.face_upstream(model, state, flag[ph],
                                                  mob[c][ph]))
            out.append(row)
        return out


__all__ = ["FaceComponentMobility"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.statefunctions.flowprops.component_properties import (  # noqa: E501
        ComponentMobility,
    )
    from pymrst.modules.ad_core.statefunctions.flux.permeability_potential_gradient import (  # noqa: E501
        PermeabilityPotentialGradient,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _ow_model():
        g = compute_geometry(cart_grid([6, 5, 3], [300.0, 250.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=3e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModelAD(g, rock, fluid, w, gravity=9.80665)
        model.FlowPropertyFunctions.set_state_function(
            "ComponentMobility", ComponentMobility(model))
        model.FlowDiscretization.set_state_function(
            "FaceComponentMobility", FaceComponentMobility(model))
        model.FlowDiscretization.set_state_function(
            "PermeabilityPotentialGradient", PermeabilityPotentialGradient(model))
        st = model.init_state_ad(State(
            pressure=np.linspace(230, 180, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.5), qWs=np.array([3e-3]),
            qOs=np.array([-3e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def test_face_component_mobility_upwinds_component_mobility():
        """Each face entry is the phase-upwinded cell component mobility."""
        model, st = _ow_model()
        fcm = model.get_props(st, "FaceComponentMobility")
        cmob = model.get_props(st, "ComponentMobility")
        flag = model.get_props(st, "PhaseUpwindFlag")
        upw = model.operators.upw
        for c in range(len(cmob)):
            for ph in range(model.n_phases):
                if cmob[c][ph] is None:
                    assert fcm[c][ph] is None
                else:
                    expect = upw(flag[ph], cmob[c][ph])
                    assert np.allclose(value(fcm[c][ph]), value(expect))
        print("ok test_face_component_mobility_upwinds_component_mobility")

    def test_reproduces_validated_component_flux():
        """scale * FaceComponentMobility * (K grad Phi) reproduces the
        validated ComponentPhaseFlux (single-point upwind: upw(a)*upw(b) =
        upw(a*b))."""
        model, st = _ow_model()
        existing = model.get_props(st, "ComponentPhaseFlux")
        fcm = model.get_props(st, "FaceComponentMobility")
        kpg = model.get_props(st, "PermeabilityPotentialGradient")
        scale = model._basis_scale()
        worst = fscale = 0.0
        for c in range(len(fcm)):
            for ph in range(model.n_phases):
                if fcm[c][ph] is None:
                    assert existing[c][ph] is None
                    continue
                recomposed = -scale[c] * fcm[c][ph] * kpg[ph]
                worst = max(worst, np.max(np.abs(
                    value(recomposed) - value(existing[c][ph]))))
                fscale = max(fscale, np.max(np.abs(value(existing[c][ph]))))
        print(f"  component flux via FaceComponentMobility vs validated: "
              f"{worst:.2e} (flux scale {fscale:.2e})")
        assert worst < 1e-12 * fscale
        print("ok test_reproduces_validated_component_flux")

    test_face_component_mobility_upwinds_component_mobility()
    test_reproduces_validated_component_flux()
    print("\nAll tests passed!")
