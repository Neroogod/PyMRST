"""Running an ensemble of models (MRST ensemble/MRSTEnsemble).

The Monte Carlo half of the ensemble module: draw parameters, run the
model once per member, and collect the quantity of interest. What comes out
is not one answer but a distribution - which is the honest output of a
model whose inputs are not known.

Pair with `history_matching` to condition that distribution on data.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EnsembleResult:
    """The quantity of interest for every member, and its statistics."""

    parameters: np.ndarray       # (n_par, N)
    qoi: np.ndarray              # (n_obs, N)

    @property
    def mean(self) -> np.ndarray:
        return self.qoi.mean(axis=1)

    @property
    def std(self) -> np.ndarray:
        return self.qoi.std(axis=1, ddof=1)

    def percentile(self, q) -> np.ndarray:
        """P10/P50/P90 and friends, across the ensemble."""
        return np.percentile(self.qoi, q, axis=1)


def gaussian_samples(mean: np.ndarray, cov: np.ndarray, n: int,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw an (n_par, N) prior ensemble from a Gaussian."""
    rng = np.random.default_rng() if rng is None else rng
    mean = np.asarray(mean, float).ravel()
    cov = np.asarray(cov, float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    chol = np.linalg.cholesky(cov)
    return mean[:, None] + chol @ rng.standard_normal((mean.size, n))


def run_ensemble(parameters: np.ndarray,
                 simulate: Callable[[np.ndarray], np.ndarray]
                 ) -> EnsembleResult:
    """Run `simulate` once per member (MRST's simulateEnsembleMembers).

    `simulate` takes one member's parameter vector and returns its quantity
    of interest. Members are independent, which is what makes an ensemble
    embarrassingly parallel - though this runs them in sequence.
    """
    m = np.atleast_2d(np.asarray(parameters, float))
    cols = [np.asarray(simulate(m[:, k]), float).ravel()
            for k in range(m.shape[1])]
    n_obs = {c.size for c in cols}
    if len(n_obs) != 1:
        raise ValueError(f"members returned different QoI lengths: {sorted(n_obs)}")
    return EnsembleResult(parameters=m, qoi=np.column_stack(cols))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ensemble.history_matching import es_mda
    from pymrst.modules.incomp.fluid import init_single_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.state import init_res_sol
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

    NX, NY = 8, 8
    N_ZONE = 4                      # the model has 4 permeability zones

    def _zones(g):
        """Split the grid into four quadrants: the unknowns to be matched."""
        x, y = g.cells.centroids[:, 0], g.cells.centroids[:, 1]
        return ((x > 50.0).astype(int) + 2 * (y > 50.0).astype(int))

    def _simulate(log_k):
        """Well data for a given set of zone log-permeabilities (mD).

        A quarter five-spot with three producers. Observed: the three
        producer rates, the injector's bottom-hole pressure, and a
        monitoring pressure in each zone.

        The monitoring pressures are not decoration. Without them there are
        only three independent observations (the three rates must sum to the
        injection rate, so one is redundant) for four unknowns, the problem
        is underdetermined, and ES-MDA will happily fit the data with a
        permeability field that is nothing like the truth. One pressure per
        zone makes the zones identifiable.
        """
        g = compute_geometry(cart_grid([NX, NY], [100.0, 100.0]))
        zone = _zones(g)
        perm = (10.0 ** np.asarray(log_k, float))[zone] * MILLI * DARCY
        rock = make_rock(g, perm[:, None], np.full((g.cells.num, 1), 0.2))
        t = compute_trans(g, rock)
        fluid = init_single_fluid(mu=1e-3, rho=1000.0)

        prods = [NX - 1, NX * (NY - 1), g.cells.num - 1]
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=100.0 / DAY, compi=np.array([1.0]))
        for p in prods:
            w = add_well(w, g, rock, np.array([p]), well_type="bhp",
                         val=100 * BARSA, compi=np.array([1.0]))
        st = incomp_tpfa(init_res_sol(g, 100 * BARSA), g, t, fluid, wells=w)

        rates = np.array([-st.wellSol[k].flux.sum() for k in (1, 2, 3)]) * DAY
        bhp = st.wellSol[0].pressure / BARSA
        # one monitoring pressure per zone
        monitors = np.array([
            st.pressure[np.flatnonzero(zone == z)].mean() / BARSA
            for z in range(N_ZONE)])
        return np.concatenate([rates, [bhp], monitors])

    def test_run_ensemble_spreads_and_averages():
        """An ensemble of permeability fields gives a spread of rates."""
        rng = np.random.default_rng(0)
        prior = gaussian_samples(np.full(N_ZONE, 2.0),
                                 np.full(N_ZONE, 0.35 ** 2), 30, rng=rng)
        res = run_ensemble(prior, _simulate)

        assert res.qoi.shape == (8, 30)
        assert np.all(res.std > 0), "an uncertain input must give uncertain output"
        p10, p90 = res.percentile(10), res.percentile(90)
        assert np.all(p10 < res.mean) and np.all(res.mean < p90)
        # the three producer rates must always sum to the injection rate
        assert np.allclose(res.qoi[:3].sum(axis=0), 100.0, rtol=1e-6), \
            "the ensemble should conserve mass member by member"
        print(f"ok test_run_ensemble_spreads_and_averages "
              f"(producer 1 takes {res.mean[0]:.1f} +/- {res.std[0]:.1f} m3/day)")

    def test_history_match_recovers_the_permeability():
        """ES-MDA recovers zone permeabilities from well rates.

        The end-to-end test: a synthetic truth, noisy observations of it, a
        wrong prior, and a forward model that is a real (nonlinear) pressure
        solve. The posterior must fit the data *and* move toward the truth -
        fitting alone would be satisfied by any number of wrong answers.
        """
        rng = np.random.default_rng(42)

        truth = np.array([2.7, 1.6, 2.2, 1.9])       # log10 mD
        d_true = _simulate(truth)
        # rates in m3/day, pressures in bar
        sigma = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2])
        d_obs = d_true + sigma * rng.standard_normal(d_true.size)
        r = np.diag(sigma ** 2)

        # a prior that is wrong, and wrong in a correlated way
        prior = gaussian_samples(np.full(N_ZONE, 2.0),
                                 np.full(N_ZONE, 0.4 ** 2), 60, rng=rng)

        def forward(m):
            return run_ensemble(m, _simulate).qoi

        post, hist = es_mda(prior, d_obs, r, forward, n_iter=4, rng=rng)

        assert hist[-1] < 0.2 * hist[0], (
            f"the match did not improve ({hist[0]:.1f} -> {hist[-1]:.1f})")

        err_prior = np.abs(prior.mean(axis=1) - truth).mean()
        err_post = np.abs(post.mean(axis=1) - truth).mean()
        assert err_post < 0.6 * err_prior, (
            f"the posterior did not move toward the truth "
            f"(prior err {err_prior:.3f}, posterior err {err_post:.3f})")
        # and the ensemble should have tightened
        assert post.std(axis=1).mean() < prior.std(axis=1).mean()

        print(f"ok test_history_match_recovers_the_permeability\n"
              f"     truth      {np.round(truth, 2)}\n"
              f"     prior mean {np.round(prior.mean(axis=1), 2)} "
              f"(err {err_prior:.3f})\n"
              f"     posterior  {np.round(post.mean(axis=1), 2)} "
              f"(err {err_post:.3f})\n"
              f"     chi-square {hist[0]:.1f} -> {hist[-1]:.2f}")

    def test_members_must_agree_on_the_qoi_length():
        """A member that returns the wrong shape is caught, not broadcast."""
        def ragged(m):
            return np.ones(int(m[0]))

        try:
            run_ensemble(np.array([[2.0, 3.0]]), ragged)
        except ValueError as e:
            assert "different QoI lengths" in str(e)
        else:
            raise AssertionError("ragged QoI should have been rejected")
        print("ok test_members_must_agree_on_the_qoi_length")

    test_members_must_agree_on_the_qoi_length()
    test_run_ensemble_spreads_and_averages()
    test_history_match_recovers_the_permeability()
    print("\nAll tests passed!")
