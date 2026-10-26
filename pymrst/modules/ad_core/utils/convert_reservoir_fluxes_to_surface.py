"""Reservoir face fluxes to surface/standard-condition fluxes (MRST
`convertReservoirFluxesToSurface`).

A simulator conserves mass and reports fluxes in *reservoir* volume, but rates a
reader cares about (and that well-management logic uses) are in *surface*
volume. The two differ by the b-factor `b = 1/B`, the reservoir-to-surface
shrinkage of each phase: the surface flux across a face is the reservoir flux
times the b-factor of the phase, taken from the upstream cell (the same cell
that donates the reservoir flux). This turns a state's `.flux` into
`.surfaceFlux` phase by phase.

MRST stores `.flux` over *all* faces and picks the interior ones; the ad_core
operators here are already interior-only, so the interior-face flux is used
directly.
"""

from typing import Dict, List

import numpy as np

from pymrst.utils.adi import value


def convert_reservoir_fluxes_to_surface(model, states) -> List[Dict]:
    """Add `surfaceFlux` (interior faces x phases) to each state.

    Each state must carry `flux` (interior faces x phases), `bfactor`
    (cells x phases) and `upstreamFlag` (interior faces x phases) - the fields a
    model writes when its `extraStateOutput` is on. `states` may be a single
    state or a list; a list is returned either way.
    """
    single = isinstance(states, dict)
    if single:
        states = [states]

    if "bfactor" not in states[0]:
        raise ValueError(
            "missing bfactor in state - was it produced by a model with "
            "extraStateOutput enabled?"
        )

    upw = model.operators.upw
    out = []
    for state in states:
        flux = np.asarray(value(state["flux"]), dtype=float)
        bfac = np.asarray(value(state["bfactor"]), dtype=float)
        upflag = np.asarray(state["upstreamFlag"])
        nf, nph = flux.shape
        surf = np.zeros((nf, nph))
        for j in range(nph):
            b_face = np.asarray(value(upw(upflag[:, j], bfac[:, j])), dtype=float)
            surf[:, j] = b_face * flux[:, j]
        state = dict(state)
        state["surfaceFlux"] = surf
        out.append(state)
    return out


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
convertReservoirFluxesToSurface = convert_reservoir_fluxes_to_surface

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    class _Model:
        def __init__(self, ops):
            self.operators = ops

    g = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = _Model(setup_operators_tpfa(g, rock))
    nf = model.operators.N.shape[0]     # 3 interior faces
    nc = g.cells.num                    # 4 cells

    def test_surface_flux_is_upstream_bfactor_times_flux():
        """One phase, all flow left-to-right: the surface flux on each face is
        the left cell's b-factor times the reservoir flux."""
        b = np.array([[1.1], [1.2], [1.3], [1.4]])          # per cell
        flux = np.array([[2.0], [3.0], [4.0]])              # per interior face
        upflag = np.ones((nf, 1), dtype=bool)               # left is upstream
        state = {"flux": flux, "bfactor": b, "upstreamFlag": upflag}
        out = convert_reservoir_fluxes_to_surface(model, state)[0]
        # faces connect cells (0,1),(1,2),(2,3); upstream = left cell 0,1,2
        expect = np.array([[1.1 * 2.0], [1.2 * 3.0], [1.3 * 4.0]])
        assert np.allclose(out["surfaceFlux"], expect), out["surfaceFlux"]
        print("ok test_surface_flux_is_upstream_bfactor_times_flux")

    def test_downstream_flag_takes_the_right_cell():
        """Flip the upstream flag: the right cell's b-factor is used instead."""
        b = np.array([[1.1], [1.2], [1.3], [1.4]])
        flux = np.array([[-2.0], [-3.0], [-4.0]])
        upflag = np.zeros((nf, 1), dtype=bool)              # right is upstream
        state = {"flux": flux, "bfactor": b, "upstreamFlag": upflag}
        out = convert_reservoir_fluxes_to_surface(model, state)[0]
        expect = np.array([[1.2 * -2.0], [1.3 * -3.0], [1.4 * -4.0]])
        assert np.allclose(out["surfaceFlux"], expect), out["surfaceFlux"]
        print("ok test_downstream_flag_takes_the_right_cell")

    def test_missing_bfactor_is_reported():
        try:
            convert_reservoir_fluxes_to_surface(model, {"flux": np.zeros((nf, 1))})
        except ValueError:
            print("ok test_missing_bfactor_is_reported")
            return
        raise AssertionError("missing bfactor should raise")

    test_surface_flux_is_upstream_bfactor_times_flux()
    test_downstream_flag_takes_the_right_cell()
    test_missing_bfactor_is_reported()
    print("\nAll tests passed!")
