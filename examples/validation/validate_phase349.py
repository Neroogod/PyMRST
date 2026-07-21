"""Phase 349 (AC12 coupled solve): the composite Newton system.

Completes AC12 - not just composing submodel equations, but solving them
*together*. Each submodel contributes its own residual and Jacobian as a
diagonal block; a coupling term contributes the off-diagonal blocks that make
the submodels couple in one Newton system.

Validated with two co-located oil-water reservoir submodels linked by a
`LinearPressureCoupling` (a leaky connection, ``q = coeff*(p_A - p_B)`` per
cell), which needs no geomechanics:

1. **Square and coupled.** The assembled system is square; without coupling the
   Jacobian is block-diagonal (zero cross-blocks), with coupling the cross-block
   from submodel A's equations to submodel B's pressure is exactly ``+coeff``
   (its analytic derivative), and the coupled Newton step genuinely differs from
   the uncoupled one.
2. **Jacobian is a correct linearization (finite difference).** Perturbing B's
   pressure and re-assembling, ``dR_A/dp_B`` matches the assembled cross-block
   to the finite-difference tolerance - the coupling derivative is right.
3. **Conservation.** The coupling adds exactly ``-q`` to A's residual and
   ``+q`` to B's, so the summed residual is unchanged.
4. **The coupled solve reduces the residual.** One Newton step on the coupled
   system (solve, unpack, update, re-assemble) reduces ``||R||`` - the system
   is solvable and the linearization is descent-correct.

No `.mat`: the coupling derivative is analytic and finite-difference-checked;
conservation and residual reduction are self-contained invariants.
"""

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.models.composite import CompositeModel
from pymrst.modules.ad_core.statefunctions.composite.coupling_term import (
    LinearPressureCoupling,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

DT = 30.0 * DAY
COEFF = 1e-11


def _reservoir(seed):
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
    st = State(pressure=np.full(g.cells.num, seed * BARSA),
               sW=np.full(g.cells.num, 0.4),
               qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -2e-3]),
               bhp=np.array([150 * BARSA, 150 * BARSA]))
    return model, st


def _unpack_update(model, state, dx):
    """Apply a Newton increment to one oil-water submodel's numeric state.
    Columns are [pressure, sW, qWs, qOs, bhp], per init_state_ad."""
    nc = model.G.cells.num
    nw = len(model.wells)
    out = State({k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in state.items()})
    out["pressure"] = state["pressure"] + dx[0:nc]
    out["sW"] = np.clip(state["sW"] + dx[nc:2 * nc], 0.0, 1.0)
    o = 2 * nc
    out["qWs"] = state["qWs"] + dx[o:o + nw]
    out["qOs"] = state["qOs"] + dx[o + nw:o + 2 * nw]
    out["bhp"] = state["bhp"] + dx[o + 2 * nw:o + 3 * nw]
    return out


def main():
    print("Phase 349 (AC12): composite coupled Newton system")
    mA, stA = _reservoir(205)
    mB, stB = _reservoir(195)
    comp = CompositeModel({"A": mA, "B": mB})
    coupling = [LinearPressureCoupling("A", "B", COEFF)]
    states = {"A": stA, "B": stB}
    states0 = {"A": stA, "B": stB}

    R, J, off = comp.assemble_coupled_system(states0, states, DT, coupling)
    nc = off["A"][1]
    r0a, r0b = off["A"][0], off["B"][0]
    assert J.shape[0] == J.shape[1] == R.size

    # (1) block structure
    R0, J0, _ = comp.assemble_coupled_system(states0, states, DT, [])
    cross = J[r0a:r0a + nc, r0b:r0b + nc].toarray()
    assert J0[r0a:r0a + nc, r0b:r0b + nc].nnz == 0, "uncoupled must be block-diag"
    assert np.allclose(np.diag(cross), COEFF) and \
        np.allclose(cross - np.diag(np.diag(cross)), 0.0)
    dx = spla.spsolve(J.tocsc(), -R)
    dx0 = spla.spsolve(J0.tocsc(), -R0)
    print(f"  square {R.size}x{R.size}; cross-block = +coeff; "
          f"coupled step differs from uncoupled by {np.linalg.norm(dx - dx0):.2e}")
    assert np.linalg.norm(dx - dx0) > 0

    # (2) finite-difference the coupling: dR_A/dp_B
    eps = 1e-3 * BARSA
    stB_p = State({k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in stB.items()})
    stB_p["pressure"] = stB["pressure"] + eps
    Rp, _, _ = comp.assemble_coupled_system(
        states0, {"A": stA, "B": stB_p}, DT, coupling)
    # A's water-conservation rows [r0a : r0a+nc]
    dRA_dpB = (Rp[r0a:r0a + nc] - R[r0a:r0a + nc]) / eps
    print(f"  FD dR_A/dp_B vs analytic +coeff: "
          f"max |diff| = {np.max(np.abs(dRA_dpB - COEFF)):.2e}")
    assert np.allclose(dRA_dpB, COEFF, atol=1e-6 * COEFF + 1e-18)

    # (3) conservation: coupling adds -q to A, +q to B (summed change is 0)
    dR = R - R0
    for c in range(mA.n_phases):
        summed = dR[r0a + c * nc:r0a + (c + 1) * nc] + \
            dR[r0b + c * nc:r0b + (c + 1) * nc]
        assert np.allclose(summed, 0.0), "coupling not conservative"
    print("  coupling conserves total (A loses exactly what B gains)")

    # (4) one coupled Newton step reduces the residual
    stA1 = _unpack_update(mA, stA, dx[off["A"][0]:off["A"][0] + off["A"][2]])
    stB1 = _unpack_update(mB, stB, dx[off["B"][0]:off["B"][0] + off["B"][2]])
    R1, _, _ = comp.assemble_coupled_system(
        states0, {"A": stA1, "B": stB1}, DT, coupling)
    print(f"  coupled Newton step: ||R|| {np.linalg.norm(R):.3e} -> "
          f"{np.linalg.norm(R1):.3e}")
    assert np.linalg.norm(R1) < np.linalg.norm(R)

    print("Phase 349 passed - the composite assembles and solves a coupled "
          "Newton system; the coupling Jacobian is FD-correct and conservative.")


if __name__ == "__main__":
    main()
