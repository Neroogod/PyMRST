"""Phase 337 (AC1): the pluggable flux discretization state functions.

Like Phase 24, these are graph machinery, not physics - MRST's `ad-core` flux
abstraction has no array of its own to dump. The proof is the stronger one the
graph phases use: the new, discretization-agnostic nodes must reproduce the
flux physics that is *already validated*, bit for bit.

Two independent checks, both against the existing validated path:

1. **PermeabilityPotentialGradient composes back to PhaseFlux.** MRST factors
   the Darcy flux as ``-lambda_f * (K grad Phi)``, where ``K grad Phi`` is a
   separate state function (`PermeabilityPotentialGradient`) so an MPFA or
   mimetic scheme can replace TPFA. PyMRST's validated `PhaseFlux` folds
   ``-T*mob*pot`` into one node. The two must agree to round-off - if they do,
   the new abstraction inherits `PhaseFlux`'s validation and can replace it.

2. **PhasePotentialDifferenceThresholded reduces to the parent.** With a zero
   `THPRES` threshold it must equal `PhasePotentialDifference` exactly, so a
   model that declares it but has no threshold pressures is undisturbed; and a
   threshold above the drop must block the phase (a conservation-style
   invariant: no flow without a super-threshold potential).

No `.mat` file: this compares two Python paths that must agree exactly, which
is a tighter bound than either against MATLAB (round-off, not solver
tolerance).
"""

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.flux import (
    PermeabilityPotentialGradient,
    PhasePotentialDifferenceThresholded,
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def build_ow(gravity=9.80665):
    """The oil-water case of validate_phase5/24, 3D so gravity is active."""
    g = compute_geometry(cart_grid([8, 6, 4], [400.0, 300.0, 40.0]))
    kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=3e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    model = GenericBlackOilModel(g, rock, fluid, w, gravity=gravity)
    st = State(pressure=np.linspace(240, 170, g.cells.num) * BARSA,
               sW=np.full(g.cells.num, 0.4),
               qWs=np.array([3e-3]), qOs=np.array([-3e-3]),
               bhp=np.array([150 * BARSA]))
    return model, model.init_state_ad(st)


def check_flux_equivalence():
    """PhaseFlux == -FaceMobility * PermeabilityPotentialGradient."""
    model, st = build_ow()
    model.FlowDiscretization.set_state_function(
        "PermeabilityPotentialGradient", PermeabilityPotentialGradient(model))
    flux = model.get_props(st, "PhaseFlux")
    mobf = model.get_props(st, "FaceMobility")
    kpg = model.get_props(st, "PermeabilityPotentialGradient")
    worst_val = worst_jac = 0.0
    for ph in range(model.n_phases):
        recomposed = -mobf[ph] * kpg[ph]
        worst_val = max(worst_val, np.max(np.abs(value(flux[ph])
                                                 - value(recomposed))))
        worst_jac = max(worst_jac,
                        np.max(np.abs((flux[ph] - recomposed).jac[0].toarray())))
    scale = max(np.max(np.abs(value(f))) for f in flux)
    print(f"  PhaseFlux vs -mob*(K grad Phi): "
          f"val {worst_val:.2e}, jac {worst_jac:.2e} (flux scale {scale:.2e})")
    # round-off in a mobility x (T x potential) product: bound relative to the
    # flux magnitude, not absolutely.
    assert worst_val < 1e-12 * scale and worst_jac < 1e-15
    # and K grad Phi is exactly T .* PhasePotentialDifference
    pot = model.get_props(st, "PhasePotentialDifference")
    t = model.operators.T
    for ph in range(model.n_phases):
        assert np.allclose(value(kpg[ph]), t * value(pot[ph]))
    assert isinstance(model.FlowDiscretization.get_state_function(
        "PermeabilityPotentialGradient").kgrad, TwoPointFluxApproximation)


def check_threshold_reduces_and_blocks():
    """Zero threshold == parent; a super-threshold blocks the phase."""
    model, st = build_ow()
    base = [value(p).copy()
            for p in model.get_props(st, "PhasePotentialDifference")]

    st.clear_cache()
    model.FlowDiscretization.set_state_function(
        "PhasePotentialDifference",
        PhasePotentialDifferenceThresholded(model, None))
    zero = model.get_props(st, "PhasePotentialDifference")
    dmax = max(np.max(np.abs(value(zero[ph]) - base[ph]))
               for ph in range(model.n_phases))
    print(f"  zero-threshold vs parent: {dmax:.2e}")
    assert dmax == 0.0

    nf = model.operators.T.size
    dp0 = base[0]
    huge = np.max(np.abs(dp0)) * 10.0
    thres = np.zeros((nf, 2))
    thres[:, 0] = -huge
    thres[:, 1] = huge
    st.clear_cache()
    model.FlowDiscretization.set_state_function(
        "PhasePotentialDifference",
        PhasePotentialDifferenceThresholded(model, thres))
    blocked = model.get_props(st, "PhasePotentialDifference")
    assert np.allclose(value(blocked[0]), 0.0)
    print("  super-threshold blocks the phase: ok")


def main():
    print("Phase 337 (AC1): pluggable flux discretization")
    check_flux_equivalence()
    check_threshold_reduces_and_blocks()
    print("Phase 337 passed - the flux abstraction reproduces the validated "
          "flux physics exactly.")


if __name__ == "__main__":
    main()
