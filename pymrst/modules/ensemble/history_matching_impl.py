"""Ensemble history matching (MRST ensemble/historyMatching).

An ensemble of models, each with different parameters, is run forward; the
spread of their predictions tells you how uncertain you are, and the
*correlation* between a model's parameters and its predictions tells you
how to correct the parameters so the predictions fit the data. That
correlation is estimated from the ensemble itself, which is what makes the
method derivative-free - unlike the adjoint, it never needs dJ/dm.

The analysis step (MRST's `enkf`), for parameters M (n_par, N) and
simulated observations D (n_obs, N):

    S  = (D - mean(D)) / sqrt(N - 1)                  anomalies
    W  = S' (S S' + a R)^-1                           gain, in ensemble space
    M <- M + (M - mean(M)) / sqrt(N - 1) * W (d - D)

ES-MDA (Emerick & Reynolds) runs this several times with an inflated
observation covariance a R, where the inflation factors satisfy
sum(1/a_i) = 1 - so the data is assimilated once in total, but gradually,
which keeps a nonlinear problem from over-correcting on the first pass.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def enkf_update(
    parameters: np.ndarray,          # (n_par, N)
    observations: np.ndarray,        # (n_obs,) the measured data
    simulated: np.ndarray,           # (n_obs, N)
    obs_cov: np.ndarray,             # (n_obs, n_obs) R, inflated by alpha
    perturbed: Optional[np.ndarray] = None,   # (n_obs, N), if perturbing
) -> np.ndarray:
    """One analysis step (MRST's `enkf`). Returns the updated parameters.

    Two algebraically equivalent forms are used, exactly as MRST does: when
    there are fewer observations than members the gain is built in
    observation space (n_obs x n_obs solve), otherwise in ensemble space
    (N x N solve). Whichever is smaller.
    """
    m = np.atleast_2d(np.asarray(parameters, float))
    d = np.atleast_2d(np.asarray(simulated, float))
    n_obs, n = d.shape
    assert m.shape[1] == n, "parameters and simulations must share the ensemble"
    assert n > 1, "an ensemble of one has no covariance to work with"

    r = np.atleast_2d(np.asarray(obs_cov, float))
    target = (np.asarray(observations, float).reshape(-1, 1)
              if perturbed is None else np.asarray(perturbed, float))

    s = (d - d.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)

    if n_obs <= n:
        c = s @ s.T + r                                  # (n_obs, n_obs)
        w = np.linalg.solve(c.T, s).T                    # S' inv(C) -> (N, n_obs)
    else:
        rinv_s = np.linalg.solve(r, s)                   # (n_obs, N)
        a = np.eye(n) + s.T @ rinv_s                     # (N, N)
        w = np.linalg.solve(a, rinv_s.T)                 # (N, n_obs)

    anomaly = (m - m.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
    return m + anomaly @ (w @ (target - d))


def esmda_alphas(n_iter: int) -> np.ndarray:
    """The standard inflation factors: a_i = n_iter, so sum(1/a_i) = 1."""
    return np.full(int(n_iter), float(n_iter))


def check_alphas(alphas: Sequence[float]) -> np.ndarray:
    """ES-MDA is only a valid assimilation if sum(1/alpha) = 1."""
    a = np.asarray(alphas, float).ravel()
    total = float((1.0 / a).sum())
    if not (0.999 < total < 1.001):
        raise ValueError(
            f"1/alpha must sum to 1 (the data must be assimilated exactly "
            f"once); got {total:.4f}")
    return a


def es_mda(
    parameters: np.ndarray,                    # (n_par, N) the prior ensemble
    observations: np.ndarray,                  # (n_obs,)
    obs_cov: np.ndarray,                       # (n_obs, n_obs) or (n_obs,)
    forward: Callable[[np.ndarray], np.ndarray],   # (n_par, N) -> (n_obs, N)
    alphas: Optional[Sequence[float]] = None,
    n_iter: int = 4,
    perturb_observations: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, list]:
    """Ensemble Smoother with Multiple Data Assimilation.

    `forward` runs the whole ensemble and returns its simulated observations.
    Returns the posterior parameters and the misfit history (one entry per
    iteration, plus the prior).
    """
    rng = np.random.default_rng() if rng is None else rng
    m = np.array(np.atleast_2d(parameters), float, copy=True)
    n = m.shape[1]

    d_obs = np.asarray(observations, float).ravel()
    r = np.asarray(obs_cov, float)
    if r.ndim == 1:
        r = np.diag(r)
    a = esmda_alphas(n_iter) if alphas is None else check_alphas(alphas)

    def misfit(sim):
        res = sim - d_obs[:, None]
        w = np.linalg.solve(r, res)
        return float(np.mean(np.sum(res * w, axis=0)))     # mean chi-square

    history = []
    sim = forward(m)
    history.append(misfit(sim))

    chol = np.linalg.cholesky(r)
    for alpha in a:
        if perturb_observations:
            # each member sees the data with its own noise, inflated by
            # sqrt(alpha) - this is what keeps the posterior spread honest
            noise = np.sqrt(alpha) * (chol @ rng.standard_normal((r.shape[0], n)))
            target = d_obs[:, None] + noise
        else:
            target = None

        m = enkf_update(m, d_obs, sim, alpha * r, perturbed=target)
        sim = forward(m)
        history.append(misfit(sim))

    return m, history


if __name__ == "__main__":
    def test_alphas_must_assimilate_the_data_once():
        """sum(1/alpha) = 1, or the data is counted twice."""
        assert np.allclose(esmda_alphas(4), [4, 4, 4, 4])
        assert np.isclose((1.0 / esmda_alphas(4)).sum(), 1.0)
        check_alphas([2.0, 2.0])                       # fine
        check_alphas([9.333, 7.0, 4.0, 2.0])           # Emerick's taper
        for bad in ([2.0, 2.0, 2.0], [1.0, 1.0]):
            try:
                check_alphas(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad} should have been rejected")
        print("ok test_alphas_must_assimilate_the_data_once")

    def test_recovers_the_kalman_posterior_on_a_linear_problem():
        """On a linear-Gaussian problem the answer is known in closed form.

        With d = G m + e, the exact posterior is

            m^ = m0 + C0 G' (G C0 G' + R)^-1 (d - G m0)
            C^ = C0 - C0 G' (G C0 G' + R)^-1 G C0

        A large ensemble must reproduce both the mean and the covariance.
        This is the test that says the update is the Kalman gain and not
        merely something that reduces the misfit.
        """
        rng = np.random.default_rng(0)
        n_par, n_obs, n = 3, 2, 20000

        g = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        m0 = np.array([1.0, -0.5, 2.0])
        c0 = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]])
        r = np.diag([0.25, 0.5])
        d = np.array([2.0, -1.0])

        # exact posterior
        k = c0 @ g.T @ np.linalg.inv(g @ c0 @ g.T + r)
        mean_exact = m0 + k @ (d - g @ m0)
        cov_exact = c0 - k @ g @ c0

        prior = (m0[:, None]
                 + np.linalg.cholesky(c0) @ rng.standard_normal((n_par, n)))
        sim = g @ prior
        noise = np.linalg.cholesky(r) @ rng.standard_normal((n_obs, n))
        post = enkf_update(prior, d, sim, r, perturbed=d[:, None] + noise)

        mean_got = post.mean(axis=1)
        cov_got = np.cov(post)
        assert np.abs(mean_got - mean_exact).max() < 0.03, (mean_got, mean_exact)
        assert np.abs(cov_got - cov_exact).max() < 0.05, (cov_got, cov_exact)
        print(f"ok test_recovers_the_kalman_posterior_on_a_linear_problem "
              f"(mean err {np.abs(mean_got - mean_exact).max():.3f}, "
              f"cov err {np.abs(cov_got - cov_exact).max():.3f})")

    def test_both_gain_forms_agree():
        """The two branches (n_obs <= N and n_obs > N) are the same algebra.

        MRST picks whichever solve is smaller. They must give the same answer,
        or one of the two paths is wrong and only half the cases would show it.
        """
        rng = np.random.default_rng(1)
        n_par, n = 6, 12

        for n_obs in (4, 12, 30):                # below, equal to, and above N
            m = rng.standard_normal((n_par, n))
            sim = rng.standard_normal((n_obs, n))
            d = rng.standard_normal(n_obs)
            a = rng.standard_normal((n_obs, n_obs))
            r = a @ a.T + n_obs * np.eye(n_obs)       # SPD

            # force each branch by calling the underlying algebra directly
            s = (sim - sim.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
            c = s @ s.T + r
            w_obs_space = np.linalg.solve(c.T, s).T
            rinv_s = np.linalg.solve(r, s)
            w_ens_space = np.linalg.solve(np.eye(n) + s.T @ rinv_s, rinv_s.T)
            assert np.abs(w_obs_space - w_ens_space).max() < 1e-8, (
                f"the two gain forms disagree at n_obs = {n_obs}")

            # and the function itself agrees with both
            got = enkf_update(m, d, sim, r)
            anomaly = (m - m.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
            want = m + anomaly @ (w_obs_space @ (d[:, None] - sim))
            assert np.abs(got - want).max() < 1e-8
        print("ok test_both_gain_forms_agree")

    def test_esmda_reduces_the_misfit_on_a_nonlinear_problem():
        """ES-MDA on a nonlinear forward model: the misfit must come down.

        The parameters are log-permeabilities; the observation is a rate,
        which depends on them nonlinearly. A single Gauss-linear update
        would over-shoot here; the point of the multiple, damped
        assimilations is that it does not.
        """
        rng = np.random.default_rng(3)
        n_par, n_obs, n = 3, 4, 60

        # a nonlinear "reservoir": rates fall off with the exponential of the
        # log-permeability, observed at four different drawdowns
        drawdown = np.array([1.0, 2.0, 4.0, 8.0])

        def forward(m):
            k = np.exp(m)                              # (3, N)
            total = k.sum(axis=0)                      # (N,)
            return np.sqrt(drawdown)[:, None] * total[None, :]

        m_true = np.array([0.5, -1.0, 0.2])
        d_true = forward(m_true[:, None]).ravel()
        r = np.diag((0.05 * np.abs(d_true)) ** 2)      # 5% noise
        d_obs = d_true + np.linalg.cholesky(r) @ rng.standard_normal(n_obs)

        prior = rng.normal(0.0, 1.5, size=(n_par, n))
        post, hist = es_mda(prior, d_obs, r, forward, n_iter=4, rng=rng)

        assert len(hist) == 5, hist
        assert hist[-1] < 0.05 * hist[0], (
            f"ES-MDA did not reduce the misfit ({hist[0]:.1f} -> {hist[-1]:.1f})")
        assert np.all(np.diff(hist) <= 1e-9 * hist[0]), \
            f"the misfit should fall monotonically: {hist}"
        # the ensemble should also have collapsed toward the data
        assert post.std(axis=1).mean() < prior.std(axis=1).mean()
        print(f"ok test_esmda_reduces_the_misfit_on_a_nonlinear_problem "
              f"(chi-square {hist[0]:.1f} -> {hist[-1]:.2f} over 4 iterations)")

    def test_esmda_matches_one_big_update_on_a_linear_problem():
        """On a linear model, ES-MDA in n steps equals one step. It must.

        This is the identity ES-MDA is built on: sum(1/alpha) = 1 means the
        data is assimilated exactly once. If the schedule were wrong, the
        posterior would be over- or under-confident.
        """
        rng = np.random.default_rng(7)
        n_par, n_obs, n = 4, 3, 40000
        g = rng.standard_normal((n_obs, n_par))

        def forward(m):
            return g @ m

        d = rng.standard_normal(n_obs)
        r = np.diag(np.full(n_obs, 0.3))
        prior = rng.standard_normal((n_par, n))

        one, _ = es_mda(prior, d, r, forward, alphas=[1.0],
                        rng=np.random.default_rng(11))
        many, _ = es_mda(prior, d, r, forward, n_iter=5,
                         rng=np.random.default_rng(11))

        dm = np.abs(one.mean(axis=1) - many.mean(axis=1)).max()
        dc = np.abs(np.cov(one) - np.cov(many)).max()
        assert dm < 0.02, f"means differ by {dm:.3f}"
        assert dc < 0.05, f"covariances differ by {dc:.3f}"
        print(f"ok test_esmda_matches_one_big_update_on_a_linear_problem "
              f"(mean diff {dm:.4f}, cov diff {dc:.4f})")

    test_alphas_must_assimilate_the_data_once()
    test_both_gain_forms_agree()
    test_recovers_the_kalman_posterior_on_a_linear_problem()
    test_esmda_matches_one_big_update_on_a_linear_problem()
    test_esmda_reduces_the_misfit_on_a_nonlinear_problem()
    print("\nAll tests passed!")
