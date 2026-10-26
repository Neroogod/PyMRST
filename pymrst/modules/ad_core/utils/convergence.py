"""Convergence criteria for the AD models (MRST ad-core/utils).

Every reservoir model asks the same question after a Newton step - *is the
residual small enough* - and every one of them used to answer it with its own
copy of the same arithmetic. MRST does not: it factors the CNV/material-balance
computation into `getConvergenceValuesCNV`, and the well part into
`checkWellConvergence`, so a single formula serves every model. This is that
factoring.

The **CNV** (a max-norm) and **MB** (a material-balance sum) both weigh each
phase's residual by its average formation volume factor `B_avg = mean(1/b)` -
which converts a surface-volume imbalance back into the reservoir volume it
actually occupies. Without that weight the phases are compared on incommensurable
scales: on SPE 1 the gas `b` is ~262, so an unweighted gas residual is 262x
larger than MRST's for the same physical error, and the tolerance is silently
262x stricter on gas than on water. This is the one place the choice of
equation currency (surface volume vs mass) has to be undone - see
`ReservoirModel.mass_basis` - which is why the caller passes residuals already
divided back to surface volume.

    CNV_i = B_avg_i * dt * max|r_i / pv|
    MB_i  = B_avg_i * dt * |sum r_i| / sum(pv)

The pore volume is the *current* one (it carries the rock-compressibility
multiplier), which the caller reads from the state.
"""

from typing import Any, List, Sequence, Tuple

import numpy as np

from pymrst.utils.adi import value


def formation_volume_factor_avg(shrinkage_factors: Sequence[Any]) -> np.ndarray:
    """`B_avg_i = mean(1/b_i)` per phase (MRST's `B_avg` in
    `getConvergenceValuesCNV`)."""
    return np.array([
        float(np.mean(1.0 / np.asarray(value(b), dtype=float)))
        for b in shrinkage_factors
    ])


def cnv_mb_values(cell_residuals: Sequence[np.ndarray], b_avg: np.ndarray,
                  pv: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-phase CNV (max-norm) and MB (material balance).

    `cell_residuals[i]` is phase `i`'s cell residual, **in surface volume** (the
    caller unscales the mass basis first). Returns `(cnv, mb)`, one value per
    phase, exactly as MRST's `getConvergenceValuesCNV` returns them before the
    solver takes the max.
    """
    pv = np.asarray(pv, dtype=float)
    pv_tot = pv.sum()
    n = len(cell_residuals)
    cnv = np.empty(n)
    mb = np.empty(n)
    for i in range(n):
        r = np.asarray(value(cell_residuals[i]), dtype=float)
        cnv[i] = b_avg[i] * dt * np.abs(r / pv).max()
        mb[i] = b_avg[i] * dt * abs(r.sum()) / pv_tot
    return cnv, mb


def cnv_mb_convergence(cell_residuals: Sequence[np.ndarray], b_avg: np.ndarray,
                       pv: np.ndarray, dt: float) -> Tuple[float, float]:
    """The scalar CNV and MB a model reports: the worst phase of each."""
    cnv, mb = cnv_mb_values(cell_residuals, b_avg, pv, dt)
    return float(cnv.max()), float(mb.max())


def well_control_residual(well_residuals: Sequence[np.ndarray],
                          control_residual: np.ndarray,
                          wells: Sequence[Any]) -> Tuple[float, float]:
    """Scaled well-closure and control residuals (MRST `checkWellConvergence`).

    A well's rate residual is scaled by a reference rate (the largest scheduled
    rate, or 1 for a bhp-only set); its control residual by 1 bar for a bhp
    control and by the target rate otherwise. A model with no wells reports
    both as zero.
    """
    if not len(wells):
        return 0.0, 0.0
    # `if rates else 1.0` guards the case where every well is bhp-controlled,
    # but not the case where rate-controlled wells all carry a zero target -
    # every producer shut in a step, say. `max` is then 0.0 and the division
    # below yields inf or nan, and since nan compares false against any
    # tolerance the step can never converge, however small the residual gets.
    # This is the same hazard the ctrl_scale comment below describes; it
    # applies here too.
    rates = [abs(w.val) for w in wells if w.type != "bhp"]
    positive = [r for r in rates if r > 1e-12]
    q_ref = max(positive) if positive else 1.0
    well = max(np.abs(np.asarray(value(r), dtype=float)).max()
               for r in well_residuals)
    well /= q_ref
    # A zero rate target is common in history schedules (a well may remain
    # open while its measured target is zero).  Dividing its floating-point
    # closure error by 1e-12 turns pressure-sized roundoff (~1e-8 m3/s in a
    # field case) into a residual of order 1e4 and makes exact convergence
    # impossible.  Use the schedule's characteristic rate for zero targets;
    # nonzero controls retain their target-relative scaling.
    ctrl_scale = np.array([
        1e5 if w.type == "bhp" else (
            abs(w.val) if abs(w.val) > 1e-12 else max(q_ref, 1.0)
        )
        for w in wells
    ])
    ctrl = float(np.max(np.abs(np.asarray(value(control_residual), float))
                        / ctrl_scale))
    return float(well), ctrl


def max_extra_residual(extra_residuals: Sequence[np.ndarray]) -> float:
    """Worst of any residuals beyond the cell/well/control set (a mechanics
    block, an aquifer equation): a plain infinity norm, unscaled."""
    vals = [np.abs(np.asarray(value(e), dtype=float)).max()
            for e in extra_residuals if np.asarray(value(e)).size]
    return max(vals) if vals else 0.0


def print_convergence_report(names: Sequence[str], values: Sequence[float],
                             tolerances: Sequence[float], iteration: int
                             ) -> None:
    """MRST's `printConvergenceReport`: one line per criterion, flagged."""
    print(f"  iteration {iteration}:")
    for name, v, tol in zip(names, values, tolerances):
        ok = "ok  " if v <= tol else "FAIL"
        print(f"    {ok} {name:>8} = {v:.3e}  (tol {tol:.1e})")


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
getConvergenceValuesCNV = cnv_mb_values
printConvergenceReport = print_convergence_report

if __name__ == "__main__":
    def test_b_avg_is_mean_reciprocal():
        b = [np.array([2.0, 4.0]), np.full(3, 0.5)]
        got = formation_volume_factor_avg(b)
        assert np.isclose(got[0], np.mean([0.5, 0.25]))
        assert np.isclose(got[1], 2.0)
        print("ok test_b_avg_is_mean_reciprocal")

    def test_cnv_mb_formula():
        """CNV is the B-weighted max of |r/pv|*dt; MB the B-weighted sum."""
        pv = np.array([10.0, 10.0])
        r = [np.array([1.0, -3.0]), np.array([2.0, 2.0])]
        b_avg = np.array([1.0, 0.5])
        dt = 2.0
        cnv, mb = cnv_mb_values(r, b_avg, pv, dt)
        # phase 0: max|r/pv| = 0.3, *dt*B = 0.6 ; sum|r|=2, /20 *dt*B = 0.2
        assert np.isclose(cnv[0], 0.6) and np.isclose(mb[0], 0.2)
        # phase 1: max|r/pv| = 0.2, *2*0.5 = 0.2 ; sum=4,/20*2*0.5 = 0.2
        assert np.isclose(cnv[1], 0.2) and np.isclose(mb[1], 0.2)
        assert np.isclose(cnv_mb_convergence(r, b_avg, pv, dt)[0], 0.6)
        print("ok test_cnv_mb_formula")

    def test_the_gas_scaling_is_what_b_avg_is_for():
        """The whole point: without B_avg a large-b phase dominates falsely.

        Give gas a b of 262 (SPE 1) and an identical *reservoir-volume* error to
        oil. Unweighted, the gas CNV is 262x the oil's; B-weighted, they are
        equal - which is the physically correct comparison."""
        pv = np.full(4, 5.0)
        # equal reservoir-volume error => surface residual scales with b
        r_oil = np.full(4, 1.0)
        r_gas = np.full(4, 262.0)
        b_avg = formation_volume_factor_avg([np.full(4, 1.0), np.full(4, 262.0)])
        cnv, _ = cnv_mb_values([r_oil, r_gas], b_avg, pv, 1.0)
        assert np.isclose(cnv[0], cnv[1], rtol=1e-12), (cnv, "should be equal")
        # and unweighted they would differ by 262x
        cnv_u, _ = cnv_mb_values([r_oil, r_gas], np.ones(2), pv, 1.0)
        assert np.isclose(cnv_u[1] / cnv_u[0], 262.0)
        print("ok test_the_gas_scaling_is_what_b_avg_is_for")

    def test_well_and_extra_residuals():
        from types import SimpleNamespace
        wells = [SimpleNamespace(type="rate", val=2e-3),
                 SimpleNamespace(type="bhp", val=150e5)]
        well_eqs = [np.array([1e-6, 0.0]), np.array([0.0, 2e-6])]
        ctrl = np.array([1e-4, 3e3])
        well, ctrl_res = well_control_residual(well_eqs, ctrl, wells)
        assert np.isclose(well, 2e-6 / 2e-3)         # scaled by max rate
        assert np.isclose(ctrl_res, max(1e-4 / 2e-3, 3e3 / 1e5))
        zero_target = [SimpleNamespace(type="rate", val=0.0),
                       SimpleNamespace(type="rate", val=2.0)]
        _, zero_ctrl = well_control_residual(
            [np.zeros(2)], np.array([1e-8, 0.0]), zero_target
        )
        assert np.isclose(zero_ctrl, 1e-8 / 2.0)
        assert well_control_residual([], np.array([]), []) == (0.0, 0.0)
        assert np.isclose(max_extra_residual([np.array([1.0, -5.0])]), 5.0)
        assert max_extra_residual([]) == 0.0
        print("ok test_well_and_extra_residuals")

    def test_every_rate_target_zero_still_converges():
        """The SPE3 failure: all rate targets zero made `q_ref` zero.

        The existing zero-target case above has one zero rate beside a nonzero
        one, so `max` is still positive and the division is safe. When *every*
        rate-controlled well carries a zero target - all producers shut for a
        step - `max` is 0.0, the scaled residual becomes inf or nan, and since
        nan compares false against any tolerance the step can never converge
        however small the residual is. SPE3 died at step 20 that way.
        """
        from types import SimpleNamespace
        all_zero = [SimpleNamespace(type="rate", val=0.0),
                    SimpleNamespace(type="rate", val=0.0)]
        well, ctrl = well_control_residual(
            [np.array([1e-9, 1e-9])], np.array([1e-9, 1e-9]), all_zero)
        assert np.isfinite(well), f"well residual is {well}"
        assert np.isfinite(ctrl), f"control residual is {ctrl}"
        # A tiny residual must read as tiny, not as inf.
        assert well < 1e-6 and ctrl < 1e-6, (well, ctrl)
        print("ok test_every_rate_target_zero_still_converges")

    def test_a_shut_well_set_falls_back_to_unit_scaling():
        """With no positive rate to scale by, the residual is its own size."""
        from types import SimpleNamespace
        wells = [SimpleNamespace(type="rate", val=0.0)]
        well, _ = well_control_residual([np.array([3e-4])], np.array([0.0]),
                                        wells)
        assert np.isclose(well, 3e-4), well
        print("ok test_a_shut_well_set_falls_back_to_unit_scaling")

    test_b_avg_is_mean_reciprocal()
    test_cnv_mb_formula()
    test_the_gas_scaling_is_what_b_avg_is_for()
    test_well_and_extra_residuals()
    test_every_rate_target_zero_still_converges()
    test_a_shut_well_set_falls_back_to_unit_scaling()
    print("\nAll tests passed!")
