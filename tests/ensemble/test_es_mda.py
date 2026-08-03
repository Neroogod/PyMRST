"""ES-MDA (F0.4 risk list).

The defect this module invites is a mismatch between two numbers that must
agree: observations are perturbed with noise scaled by `sqrt(alpha)` while the
update inflates the observation covariance by `alpha`. Scale the noise by
`alpha` instead and the perturbed data carry covariance `alpha**2 R` against an
update that assumes `alpha R` - the history matcher still runs, the misfit
still falls, and the posterior spread is simply wrong.

`es_mda` takes its random generator as an argument, which makes that testable
exactly rather than statistically: with a stub generator the perturbation is
known in closed form, so the whole assimilation can be replayed by hand through
`enkf_update` and compared bit for bit.

One expectation deliberately *not* tested here: for a linear-Gaussian problem
ES-MDA's tempered updates are often said to compose to a single update. They do
for the exact Kalman recursion, and they do not for this deterministic
per-realization form, whose anomalies transform as `(I - KG)A` and so shrink
faster than the Kalman covariance. Measured on a linear model, one assimilation
and four differ by 0.30 in the posterior mean. That is the update's known
character, not a defect, and asserting the identity would pin a false claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ensemble.history_matching.utils.enkf import enkf_update
from pymrst.modules.ensemble.history_matching.utils.es_mda import (
    check_alphas, es_mda, esmda_alphas,
)


class _OnesGenerator:
    """Stands in for a `Generator`, returning all-ones noise.

    Not a seeded generator: a seed makes a run repeatable but still leaves the
    expected answer unknown, so the test could only compare es_mda to itself.
    """

    def standard_normal(self, shape):
        return np.ones(shape)


def _linear_case(members=25, seed=1):
    rng = np.random.default_rng(seed)
    n_param, n_data = 3, 4
    operator = rng.standard_normal((n_data, n_param))
    prior = rng.standard_normal((n_param, members))
    covariance = np.diag([0.04, 0.09, 0.01, 0.05])
    truth = np.array([1.0, -2.0, 0.5])
    observations = operator @ truth + 0.1
    return operator, prior, covariance, observations


def test_a_full_assimilation_is_reproduced_by_hand():
    """The load-bearing test: four tempered updates, replayed exactly.

    Reproducing the run through `enkf_update` pins the `sqrt(alpha)` noise
    scaling against the `alpha*R` inflation, the Cholesky factor, and the order
    of the loop. Any of those wrong and the replay diverges.
    """
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    alphas = [4.0]*4

    got, _ = es_mda(prior, observations, covariance, forward, alphas=alphas,
                    perturb_observations=True, rng=_OnesGenerator())

    chol = np.linalg.cholesky(covariance)
    members = prior.shape[1]
    expected = prior.copy()
    simulated = forward(expected)
    for alpha in alphas:
        target = (observations[:, None]
                  + np.sqrt(alpha)*(chol @ np.ones((covariance.shape[0], members))))
        expected = enkf_update(expected, observations, simulated,
                               alpha*covariance, perturbed=target)
        simulated = forward(expected)

    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_a_single_assimilation_is_reproduced_by_hand():
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble

    got, _ = es_mda(prior, observations, covariance, forward, alphas=[1.0],
                    perturb_observations=True, rng=_OnesGenerator())
    chol = np.linalg.cholesky(covariance)
    target = (observations[:, None]
              + chol @ np.ones((covariance.shape[0], prior.shape[1])))
    expected = enkf_update(prior, observations, forward(prior), covariance,
                           perturbed=target)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_the_inflation_factors_assimilate_the_data_exactly_once():
    """The constraint that makes ES-MDA a Bayesian update rather than an
    arbitrary sequence of nudges. `sum(1/alpha) = 1` is the whole rule."""
    for n_iter in (1, 2, 4, 8):
        alphas = esmda_alphas(n_iter)
        assert len(alphas) == n_iter
        assert (1.0/alphas).sum() == pytest.approx(1.0, rel=1e-12)


def test_inflation_factors_that_over_or_under_assimilate_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        check_alphas([2.0, 2.0, 2.0])          # sums to 1.5: over-assimilated
    with pytest.raises(ValueError, match="sum to 1"):
        check_alphas([8.0, 8.0])               # sums to 0.25: under-assimilated
    np.testing.assert_allclose(check_alphas([4.0]*4), [4.0]*4)


def test_the_misfit_history_records_every_assimilation_and_falls():
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    _, history = es_mda(prior, observations, covariance, forward,
                        alphas=[4.0]*4, rng=np.random.default_rng(0))
    assert len(history) == 5, "prior misfit plus one per assimilation"
    assert history[-1] < history[0]


def test_uninformative_observations_leave_the_prior_almost_untouched():
    """The limit that catches an inverted covariance.

    With an enormous observation error the data say nothing and the ensemble
    should barely move. A flipped inverse makes this the case where the update
    is *largest*.
    """
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    posterior, _ = es_mda(prior, observations, covariance*1e12, forward,
                          alphas=[1.0], perturb_observations=False)
    assert np.abs(posterior - prior).max() < 1e-3*np.abs(prior).max()


def test_precise_observations_pull_the_simulated_data_onto_them():
    """The opposite limit, on an observation the model can actually reproduce.

    `_linear_case` adds a constant offset to the truth, which for 4 data and 3
    parameters lies partly outside the range of the operator - so no update can
    drive that residual to zero and a tight assertion here would be testing the
    case, not the code. The observation is made consistent instead.
    """
    operator, prior, covariance, _ = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    observations = operator @ np.array([1.0, -2.0, 0.5])

    posterior, _ = es_mda(prior, observations, covariance*1e-8, forward,
                          alphas=[1.0], perturb_observations=False)
    residual = forward(posterior) - observations[:, None]
    assert np.abs(residual).max() < 1e-4*np.abs(observations).max()


def test_the_same_generator_gives_the_same_answer():
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    first, _ = es_mda(prior, observations, covariance, forward,
                      alphas=[4.0]*4, rng=np.random.default_rng(7))
    second, _ = es_mda(prior, observations, covariance, forward,
                       alphas=[4.0]*4, rng=np.random.default_rng(7))
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_the_prior_ensemble_is_not_modified_in_place():
    """An in-place update makes a second call from the same prior meaningless,
    which an outer loop comparing settings would never notice.

    Worth being precise about what this guards. Removing the defensive `copy`
    alone does *not* break it - the loop rebinds `m` to a new array each
    assimilation and never writes through - so this test passes today for a
    reason stronger than the copy. It fails only for the mutation that actually
    matters: aliasing the input *and* writing back into it, which is what an
    in-place optimisation would look like. That was confirmed by making that
    two-part change and watching this fail.
    """
    operator, prior, covariance, observations = _linear_case()
    forward = lambda ensemble: operator @ ensemble
    untouched = prior.copy()
    es_mda(prior, observations, covariance, forward, alphas=[4.0]*4,
           rng=np.random.default_rng(3))
    np.testing.assert_array_equal(prior, untouched)
