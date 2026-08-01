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
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return a layered log-normal permeability field in MRST cell order."""
    dims = tuple(int(d) for d in np.atleast_1d(dims))
    if len(dims) == 2:
        dims += (1,)
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise ValueError("log_norm_layers needs two or three positive dimensions")
    for name, value in (("sigma", sigma), ("a", a), ("b", b), ("std", std)):
        if np.asarray(value).size != 1:
            raise ValueError(f"{name} must be scalar")
    if float(std) <= 0:
        raise ValueError("std must be positive")
    nx, ny, nz = dims
    rng = np.random.default_rng() if rng is None else rng
    bounds = _layer_bounds(nz, mean_perm, indices, rng)
    layer_means = _layer_means(mean_perm)
    count = bounds.size-1
    if layer_means is not None and layer_means.size != count:
        raise ValueError("one mean permeability is required per layer")
    if fields is not None and len(fields) != count:
        raise ValueError("fields must contain one draw per layer")
    if shifts is not None and len(shifts) != count:
        raise ValueError("shifts must contain one value per layer")

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
    result = perm.ravel(order="F")
    return (result, bounds) if return_indices else result


def _layer_bounds(
    nz: int,
    mean_perm: Optional[Sequence[float]],
    indices: Optional[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    if indices is not None:
        bounds = np.asarray(indices, dtype=np.int64).ravel()
    elif mean_perm is None:
        # MRST's no-M call takes its nargin<2 branch: one permeability
        # realization per model layer.
        bounds = np.arange(nz+1, dtype=np.int64)
    else:
        values = np.asarray(mean_perm)
        if values.ndim == 0:
            requested = int(values)
            if float(values) != requested:
                raise ValueError("number of layers must be an integer")
            requested = 1 if requested < 0 else requested
        else:
            requested = values.size
        if requested < 1 or requested > nz:
            raise ValueError(f"{requested} layers requested for {nz} model layers")
        if requested == 1:
            bounds = np.array([0, nz], dtype=np.int64)
        elif requested == nz:
            bounds = np.arange(nz+1, dtype=np.int64)
        else:
            internal = np.sort(rng.choice(
                np.arange(1, nz), size=requested-1, replace=False
            ))
            bounds = np.concatenate([[0], internal, [nz]])
    if (bounds.size < 2 or bounds[0] != 0 or bounds[-1] != nz
            or np.any(np.diff(bounds) <= 0)):
        raise ValueError(f"layer boundaries must increase from 0 to {nz}")
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
        value = np.asarray(fields[layer_index], float)
        expected = (nx, ny, n_layers+2)
        if value.shape != expected:
            raise ValueError(
                f"field {layer_index} must have shape {expected}, got {value.shape}"
            )
        return value
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
