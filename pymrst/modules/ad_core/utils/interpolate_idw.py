"""Inverse-distance-weighted interpolation (MRST `interpolateIDW`).

A scattered-data interpolant: the value at a query point is a weighted average
of the sample values, each weighted by `1 / distance^order`, so nearer samples
dominate. It needs no grid or triangulation - the reason MRST uses it for
scattered inputs (e.g. well trajectories, unstructured property clouds) where a
`griddedInterpolant` does not apply.

`NaN` samples are treated as missing: they drop out of a column's weighting, and
if every sample for a column is missing the query falls back to a plain mean.
"""

import numpy as np


def interpolate_idw(x: np.ndarray, f: np.ndarray, xq: np.ndarray,
                    order: float = 2.0) -> np.ndarray:
    """Interpolate the sample values `f` at `x` onto the query points `xq`.

    `x` is `(ns, nx)` sample coordinates, `f` is `(ns, nf)` sample values, `xq`
    is `(nq, nx)` query coordinates; returns `(nq, nf)`. `order` is the
    distance exponent (2 is the usual inverse-square weighting).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    f = np.atleast_2d(np.asarray(f, dtype=float))
    xq = np.atleast_2d(np.asarray(xq, dtype=float))
    ns, nx = x.shape
    nf = f.shape[1]
    nq = xq.shape[0]
    assert f.shape[0] == ns
    assert xq.shape[1] == nx

    v = np.full((nq, nf), np.nan)
    for i in range(nq):
        d = x - xq[i, :]
        dist = np.maximum(np.sum(np.abs(d) ** order, axis=1), 1e-20)
        wi = 1.0 / dist
        W = np.repeat(wi[:, None], nf, axis=1)
        fj = f.copy()
        for j in range(nf):
            disabled = np.isnan(fj[:, j])
            W[disabled, j] = 0.0
            fj[disabled, j] = 0.0
            s = W[:, j].sum()
            if s < 1e-18:
                W[:, j] = 1.0 / ns
            else:
                W[:, j] = W[:, j] / s
        v[i, :] = np.sum(W * fj, axis=0)
    return v


if __name__ == "__main__":
    def test_query_on_a_sample_returns_that_sample():
        """A query point sitting on a sample gets ~that sample's value: its
        distance is clamped to 1e-20, so its weight dominates."""
        x = np.array([[0.0], [1.0], [2.0]])
        f = np.array([[10.0], [20.0], [30.0]])
        v = interpolate_idw(x, f, np.array([[1.0]]), order=2.0)
        assert np.isclose(v[0, 0], 20.0, atol=1e-6), v
        print("ok test_query_on_a_sample_returns_that_sample")

    def test_midpoint_of_two_equal_samples_is_symmetric_average():
        """Halfway between two samples the weights are equal, so the value is
        their mean regardless of the exponent."""
        x = np.array([[0.0], [2.0]])
        f = np.array([[0.0], [100.0]])
        v = interpolate_idw(x, f, np.array([[1.0]]), order=2.0)
        assert np.isclose(v[0, 0], 50.0), v
        print("ok test_midpoint_of_two_equal_samples_is_symmetric_average")

    def test_nan_sample_is_ignored():
        """A NaN sample drops out of the weighting; the result uses the rest."""
        x = np.array([[0.0], [1.0], [2.0]])
        f = np.array([[10.0], [np.nan], [30.0]])
        v = interpolate_idw(x, f, np.array([[1.0]]), order=2.0)
        assert np.isfinite(v[0, 0])
        assert np.isclose(v[0, 0], 20.0), v   # equidistant from 10 and 30
        print("ok test_nan_sample_is_ignored")

    test_query_on_a_sample_returns_that_sample()
    test_midpoint_of_two_equal_samples_is_symmetric_average()
    test_nan_sample_is_ignored()
    print("\nAll tests passed!")
