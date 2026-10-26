"""Port of MRST ``modules/spe10/setupSPE10_AD.m``.

MRST's field-scale SPE10 oil-water benchmark: the full 60x220x85 SPE10 Model 2
permeability field (or an I/J/layer subset of it), four corner producers and
one center-well injector, a rate ramp-up schedule. This is a different,
larger dataset than SPE10 "Model I" (``tenth_csp_model_i.py``'s 100x1x20
synthetic oil-gas case) - not a subset of it.

``TwoPhaseOilWaterModelAD`` now carries gravity (see its own module
docstring), so this case runs with the same ``gravity = [0, 0, 9.80665]``
MRST's ``setupSPE10_AD.m`` uses. It does not yet have a minimum-pressure
clamp, so this port omits ``model.minimumPressure`` rather than silently
pretending it exists; that gap is recorded, not fixed here.

`setup_spe10_ad_ad` is PyMRST's earlier version on the legacy models, kept for
its callers; the port of MRST `setupSPE10_AD` is `pymrst/modules/spe10/setup_spe10_ad.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.ad_props import ADFluid
from pymrst.modules.spe10.get_spe10_rock import SPE10_CART_DIMS, get_spe10_rock
from pymrst.params import NORM_GRAVITY, make_rock, vertical_well
from pymrst.params.rock import Rock
from pymrst.utils.adi import ad_exp, ad_max, ad_min
from pymrst.utils.interp_table import interp_table
from pymrst.utils.units import CENTI, DAY, FT, INCH, POISE, POUND, PSIA, STB

GRAVITY_VEC = np.array([0.0, 0.0, NORM_GRAVITY])


def _fluid(datum_pressure: float) -> ADFluid:
    """MRST's hand-written SPE10 AD fluid, formula for formula.

    Water: exponential compressibility about ``bW0 = 1/1.01`` at
    ``datum_pressure``, constant viscosity. Oil: linear interpolation of
    ``1/B_o`` and mu_o over a 4-point pressure table. Both relperms are
    Corey, exponent 2, with connate/residual saturations both 0.2 - MRST's
    own ``coreyPhaseRelpermAD(n, sr, kwm, sr_tot)``, transcribed exactly
    (clip to [0, 1] before the power, not after).
    """
    srw = sro = 0.2
    b_w0 = 1.0 / 1.01

    def b_w(p, *_a, **_kw):
        return b_w0 * ad_exp((p - datum_pressure) * 3e-6 / PSIA)

    p_table = np.array([300.0, 800.0, 8000.0, 8001.0]) * PSIA
    b_table = 1.0 / np.array([1.05, 1.02, 1.01, 1.01])

    def b_o(p, *_a, **_kw):
        return interp_table(p_table, b_table, p)

    def _corey(s, sr, sr_tot, n=2.0, kwm=1.0):
        # `ad_min`/`ad_max` differentiate through the winning branch, so this
        # stays a real Jacobian at saturations that hit the clamp - unlike
        # `np.clip`, which would silently zero the derivative there.
        sat = ad_min(ad_max((s - sr) / (1.0 - sr_tot), 0.0), 1.0)
        return kwm * sat ** n

    fluid = ADFluid(
        rhoS=np.array([64.0 * POUND / FT ** 3, 53.0 * POUND / FT ** 3]),
        mu=np.array([0.3 * CENTI * POISE, 3.0 * CENTI * POISE]),
        b_w=b_w, b_o=b_o,
        kr_w=lambda sw: _corey(sw, srw, srw + sro),
        kr_o=lambda so: _corey(so, sro, srw + sro),
    )
    # MRST's `pvMultR`: `1 + cR*(p - pRef)`, cR = 1e-6/psia.
    fluid.pv_mult = lambda p: 1.0 + (1e-6 / PSIA) * (p - datum_pressure)
    return fluid


@dataclass
class SPE10ADCase:
    dataset_pore_volume_fraction: float
    grid: object
    rock: Rock
    fluid: ADFluid
    model: TwoPhaseOilWaterModelAD
    state0: OWState
    timesteps: np.ndarray
    wells: list


def setup_spe10_ad_ad(
    *, layers=None, i_range=None, j_range=None,
    dt: float = 30.0 * DAY, total_time: float = 2000.0 * DAY, rampup: int = 8,
    min_poro: float = 0.01,
) -> SPE10ADCase:
    """Build MRST's ``setupSPE10_AD`` case: an I/J/layer subset of SPE10
    Model 2, four corner producers, one center injector.

    `layers`/`i_range`/`j_range` are 1-based inclusive MRST-style ranges
    (``get_spe10_rock``'s own convention); omitted means "the whole extent",
    matching MRST's ``opt.layers = 1:85``, ``opt.I = 1:60``, ``opt.J = 1:220``
    defaults.
    """
    nx_full, ny_full, nz_full = SPE10_CART_DIMS
    layers = np.arange(1, nz_full + 1) if layers is None else np.atleast_1d(layers)
    i_range = np.arange(1, nx_full + 1) if i_range is None else np.atleast_1d(i_range)
    j_range = np.arange(1, ny_full + 1) if j_range is None else np.atleast_1d(j_range)

    raw = get_spe10_rock(i_range, j_range, layers)
    if raw is None:
        raise FileNotFoundError("SPE10 model 2 dataset (spe10_rock.mat) not found")

    # MRST: `volFrac = sum(rock.poro)/1.9141e+05` - the published sum of
    # porosity over the *whole* 1.122M-cell field, used to scale the
    # injector's rate to whatever fraction of the full reservoir this subset
    # represents.
    vol_frac = float(raw.poro.sum()) / 1.9141e5
    poro = np.maximum(raw.poro, min_poro)

    ni, nj, nl = i_range.size, j_range.size, layers.size
    cart_dims = (ni, nj, nl)
    phys_dims = np.asarray(cart_dims) * np.array([20.0, 10.0, 2.0]) * FT
    grid = compute_geometry(cart_grid(cart_dims, phys_dims))
    rock = make_rock(grid, raw.perm, poro[:, None])

    wells = None
    corners = [(0, 0, "P1"), (ni - 1, 0, "P2"),
               (ni - 1, nj - 1, "P3"), (0, nj - 1, "P4")]
    for i0, j0, well_name in corners:
        wells = vertical_well(
            wells, grid, rock, i0, j0,
            well_type="bhp", val=4000.0 * PSIA, radius=5.0 * INCH,
            compi=np.array([0.5, 0.5]), sign=-1, name=well_name,
            gravity=GRAVITY_VEC,
        )
    i_center, j_center = (ni - 1) // 2, (nj - 1) // 2
    wells = vertical_well(
        wells, grid, rock, i_center, j_center,
        well_type="rate", val=vol_frac * 5000.0 * STB / DAY, radius=5.0 * INCH,
        compi=np.array([1.0, 0.0]), sign=1, name="I1",
        gravity=GRAVITY_VEC,
    )

    fluid = _fluid(datum_pressure=6000.0 * PSIA)
    model = TwoPhaseOilWaterModelAD(grid, rock, fluid, wells,
                                    gravity=NORM_GRAVITY)
    state0 = model.initial_state(6000.0 * PSIA, 0.2)

    timesteps = rampup_timesteps(total_time, dt, rampup)
    return SPE10ADCase(vol_frac, grid, rock, fluid, model, state0, timesteps, wells)


def _solve_ministep(model, state, dt, max_cuts, depth=0):
    """`model.step` with MRST-style timestep cutting on Newton failure.

    `TwoPhaseOilWaterModelAD.step` is a bare max-iterations Newton loop with
    no retry of its own (its own `__main__` tests never need one). MRST's
    `NonLinearSolver.solveTimestep` halves `dt` and retries recursively on
    failure, up to `max_timestep_cuts`; ported here rather than in the model
    itself since cutting is a schedule-loop concern; `model.step` stays a
    single deterministic attempt other callers can rely on.
    """
    new_state, ok, its = model.step(state, dt)
    if ok:
        return new_state, its
    if depth >= max_cuts:
        raise RuntimeError(f"timestep cuts exhausted at dt={dt / DAY:.4f} days")
    half = dt / 2.0
    mid_state, its1 = _solve_ministep(model, state, half, max_cuts, depth + 1)
    end_state, its2 = _solve_ministep(model, mid_state, half, max_cuts, depth + 1)
    return end_state, its1 + its2


def run_spe10_ad(case: SPE10ADCase, nsteps: int | None = None, *,
                 max_timestep_cuts: int = 8, verbose: bool = True):
    """Loop `model.step` over the ramp-up schedule; wells hold fixed controls
    throughout, matching MRST's `setupSPE10_AD` (only dt varies)."""
    count = len(case.timesteps) if nsteps is None else min(int(nsteps), len(case.timesteps))
    state, states, iterations = case.state0.copy(), [], 0
    for step_index in range(count):
        dt = float(case.timesteps[step_index])
        if verbose:
            print(f"SPE10 AD step {step_index + 1}/{count}", flush=True)
        state, its = _solve_ministep(case.model, state, dt, max_timestep_cuts)
        iterations += its
        states.append(state.copy())
    return states, iterations


__all__ = ["SPE10ADCase", "setup_spe10_ad_ad", "run_spe10_ad"]


if __name__ == "__main__":
    def test_two_layer_subset_builds_and_steps():
        case = setup_spe10_ad_ad(layers=[1, 2], i_range=np.arange(1, 61),
                               j_range=np.arange(1, 221))
        assert case.grid.cells.num == 60 * 220 * 2, case.grid.cells.num
        assert len(case.wells) == 5
        assert case.wells[-1].type == "rate"
        assert 0.0 < case.dataset_pore_volume_fraction < 1.0
        states, its = run_spe10_ad(case, nsteps=2, verbose=False)
        assert len(states) == 2
        assert its > 0
        assert np.all(states[-1].pressure > 0.0)
        print(f"ok test_two_layer_subset_builds_and_steps ({its} Newton its)")

    def test_ministep_cuts_and_recovers_from_a_forced_failure():
        """A single call that never converges must still exhaust cuts and
        raise, and a model that fails once then succeeds must recover via
        exactly one halving - pins both ends of `_solve_ministep`."""
        from pymrst.modules.ad_blackoil.examples.spe10.setup_spe10_ad import (
            _solve_ministep,
        )

        class _AlwaysFails:
            def step(self, state, dt):
                return state, False, 25

        try:
            _solve_ministep(_AlwaysFails(), object(), 10.0, max_cuts=3)
            raise AssertionError("expected exhausted-cuts failure")
        except RuntimeError as exc:
            assert "cuts exhausted" in str(exc)

        calls = []

        class _FailsOnFirstFullStep:
            def step(self, state, dt):
                calls.append(dt)
                ok = dt < 9.99
                return (state + 1 if ok else state), ok, 5

        result, its = _solve_ministep(_FailsOnFirstFullStep(), 0, 10.0, max_cuts=3)
        assert calls == [10.0, 5.0, 5.0], calls
        assert result == 2 and its == 10, (result, its)
        print("ok test_ministep_cuts_and_recovers_from_a_forced_failure")

    test_two_layer_subset_builds_and_steps()
    test_ministep_cuts_and_recovers_from_a_forced_failure()
    print("\nAll tests passed!")
