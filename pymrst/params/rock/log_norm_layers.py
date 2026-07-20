"""Layered log-normal permeability fields, ported from MRST `logNormLayers`."""

from typing import Optional, Sequence

import numpy as np

from pymrst.params.rock.gaussian_field import smooth_field


def log_norm_layers(
    dims: Sequence[int],
    mean_perm: Optional[Sequence[float]] = None,
    indices: Optional[Sequence[int]] = None,
    sigma: float = 2.0,
    a: float = 0.6,
    b: float = 2.0,
    size: Sequence[int] = (9, 3, 3),
    std: float = 4.5,
    rng: Optional[np.random.Generator] = None,
    fields: Optional[Sequence[np.ndarray]] = None,
    shifts: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return a layered log-normal permeability field in MRST cell order."""
    dims = tuple(int(d) for d in np.atleast_1d(dims))
    assert len(dims) == 3, "log_norm_layers needs a 3D grid"
    nx, ny, nz = dims
    rng = np.random.default_rng() if rng is None else rng
    bounds = _layer_bounds(nz, mean_perm, indices)
    layer_means = _layer_means(mean_perm)

    perm = np.zeros((nx, ny, nz))
    for i in range(bounds.size - 1):
        n_layers = int(bounds[i + 1] - bounds[i])
        raw = _layer_draw(nx, ny, n_layers, i, rng, fields)
        shift = float(shifts[i]) if shifts is not None else float(rng.standard_normal())
        smoothed = smooth_field(raw - a * shift, size, std)
        layer = np.exp(b + sigma * smoothed)[:, :, 1 : n_layers + 1]
        if layer_means is not None:
            layer = layer_means[i] * layer / layer.mean()
        perm[:, :, bounds[i] : bounds[i + 1]] = layer
    return perm.ravel(order="F")


def _layer_bounds(
    nz: int,
    mean_perm: Optional[Sequence[float]],
    indices: Optional[Sequence[int]],
) -> np.ndarray:
    if indices is not None:
        bounds = np.asarray(indices, dtype=np.int64).ravel()
    elif mean_perm is not None and np.ndim(mean_perm) > 0:
        n_layers = len(mean_perm)
        assert n_layers <= nz, f"{n_layers} layers asked for in {nz} model layers"
        bounds = np.linspace(0, nz, n_layers + 1).astype(np.int64)
    else:
        bounds = np.arange(nz + 1)
    assert bounds[0] == 0 and bounds[-1] == nz, (
        f"layer boundaries must run 0..{nz}, got {bounds}"
    )
    return bounds


def _layer_means(mean_perm: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if mean_perm is None or np.ndim(mean_perm) == 0:
        return None
    return np.asarray(mean_perm, dtype=np.float64).ravel()


def _layer_draw(
    nx: int,
    ny: int,
    n_layers: int,
    layer_index: int,
    rng: np.random.Generator,
    fields: Optional[Sequence[np.ndarray]],
) -> np.ndarray:
    if fields is not None:
        return np.asarray(fields[layer_index], float).reshape(nx, ny, n_layers + 2)
    return rng.standard_normal((nx, ny, n_layers + 2))


if __name__ == "__main__":

    def test_log_norm_layers_is_layered_and_lognormal() -> None:
        rng = np.random.default_rng(3)
        nx, ny, nz = 30, 25, 6
        perm = log_norm_layers(
            (nx, ny, nz), mean_perm=[100.0, 800.0, 40.0], rng=rng
        ).reshape((nx, ny, nz), order="F")
        assert np.all(perm > 0)
        assert np.isclose(perm[:, :, 0].mean(), 100.0, rtol=0.02)
        assert np.isclose(perm[:, :, -1].mean(), 40.0, rtol=0.05)
        assert perm.max() / perm.min() > 10.0
        print("ok test_log_norm_layers_is_layered_and_lognormal")

    def test_log_norm_layers_reuses_a_given_draw() -> None:
        rng = np.random.default_rng(5)
        nx, ny, nz = 12, 10, 4
        fields = [rng.standard_normal((nx, ny, nz + 2))]
        shifts = [0.25]
        first = log_norm_layers((nx, ny, nz), indices=[0, nz], fields=fields, shifts=shifts)
        second = log_norm_layers((nx, ny, nz), indices=[0, nz], fields=fields, shifts=shifts)
        assert np.array_equal(first, second)
        print("ok test_log_norm_layers_reuses_a_given_draw")

    test_log_norm_layers_is_layered_and_lognormal()
    test_log_norm_layers_reuses_a_given_draw()
    print("\nAll tests passed!")

