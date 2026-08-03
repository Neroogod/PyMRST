"""Statistics and geostatistics qualification for native ADFNE15 owners."""

import numpy as np
import pytest

from pymrst.modules.adfne15 import (info, kriging, markov, stats, upscaling,
                                    variocloud, variogram, variomodel)

POINTS = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
VALUES = np.array([1., 2., 3., 4.])


def test_info_and_descriptive_statistics():
    result = info(np.array([[0., 0., 1., 0.], [0., 0., 0., 2.]]))
    assert result.count == 2
    np.testing.assert_allclose(result.centers, [[.5, 0], [0, 1]])
    np.testing.assert_allclose(result.lengths, [1, 2])
    summary = stats([1, 2, np.nan, 4])
    assert summary.count == 4 and summary.minimum == 1 and summary.maximum == 4
    np.testing.assert_allclose(summary.mean, 7/3)
    np.testing.assert_allclose(summary.standard_deviation,
                               np.std([1, 2, 4], ddof=1))


def test_variogram_cloud_binning_and_models():
    cloud = variocloud(POINTS, VALUES)
    assert cloud.distances.shape == cloud.semivariances.shape == (6,)
    np.testing.assert_allclose(cloud.variance, np.var(VALUES, ddof=1))
    experimental = variogram(cloud.distances, cloud.semivariances,
                             cloud.variance, minimum_lag_factor=.5)
    assert experimental.counts.sum() <= 6
    lags = np.array([0., .5, 1., 2.])
    np.testing.assert_allclose(
        variomodel({"name":"sph","nugget":0,"sill":2,"range":1}, lags),
        [0, 1.375, 2, 2])
    np.testing.assert_allclose(
        variomodel({"name":"lin","nugget":1,"slope":2}, lags), 1+2*lags)


def test_ordinary_kriging_reproduces_observations_on_matching_grid():
    result = kriging(
        POINTS, VALUES,
        model={"name":"sph","nugget":0,"sill":2,"range":2},
        grid_shape=(2, 2))
    np.testing.assert_allclose(result.estimates, [[1, 2], [3, 4]], atol=1e-12)
    np.testing.assert_allclose(result.errors, 0, atol=1e-12)


def test_markov_and_recursive_upscaling():
    chain = markov([[1, 0], [0, 1]], 1, 10,
                   rng=np.random.default_rng(4))
    np.testing.assert_array_equal(chain, np.ones(10, dtype=int))
    values = np.arange(1, 17, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(upscaling(values, "arithmetic", 2),
                               [[3.5, 5.5], [11.5, 13.5]])
    np.testing.assert_allclose(upscaling(np.full((4, 4), 2.), "harmonic", 1),
                               [[2.]])
    with pytest.raises(ValueError, match="powers of two"):
        upscaling(np.ones((3, 4)))
