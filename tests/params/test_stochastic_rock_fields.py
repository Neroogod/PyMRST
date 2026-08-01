import numpy as np
import pytest

from pymrst.params.rock.gaussian_field import gaussian_field, gaussian_kernel
from pymrst.params.rock.log_norm_layers import log_norm_layers


@pytest.mark.parametrize("dims,size", [
    ((17,), 5), ((12, 9), [5, 3, 7]), ((8, 7, 6), [3, 5, 3]),
])
def test_gaussian_field_dimensions_kernel_forms_and_range(dims, size):
    result = gaussian_field(dims, vals=[.1, .4], size=size,
                            rng=np.random.default_rng(4))
    assert result.shape == dims
    assert np.isclose(result.min(), .1)
    assert np.isclose(result.max(), .4)


def test_gaussian_field_replays_draw_and_mrst_constant_behavior():
    draw = np.arange(63, dtype=float).reshape(9, 7)
    first = gaussian_field(draw.shape, size=3, field=draw)
    second = gaussian_field(draw.shape, size=[3, 3, 3], field=draw)
    np.testing.assert_allclose(first, second)
    assert np.all(np.isnan(gaussian_field((5, 4), field=np.ones((5, 4)))))


@pytest.mark.parametrize("kwargs, message", [
    ({"dims": ()}, "dims"),
    ({"dims": (3, 3, 3, 3)}, "dims"),
    ({"dims": (3, 3), "vals": [1]}, "vals"),
    ({"dims": (3, 3), "size": 4}, "odd"),
    ({"dims": (3, 3, 3), "size": [3, 3]}, "cover"),
    ({"dims": (3, 3), "std": 0}, "std"),
])
def test_gaussian_field_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gaussian_field(**kwargs)
    with pytest.raises(ValueError, match="shape"):
        gaussian_field((3, 3), field=np.ones((9,)))


def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel([3, 5, 7], 1.5)
    assert kernel.shape == (3, 5, 7)
    assert np.isclose(kernel.sum(), 1.0)


def test_log_norm_default_uses_every_model_layer_and_returns_boundaries():
    values, bounds = log_norm_layers(
        (8, 7, 4), rng=np.random.default_rng(2), return_indices=True
    )
    assert values.shape == (8*7*4,)
    np.testing.assert_array_equal(bounds, [0, 1, 2, 3, 4])


def test_log_norm_scalar_layer_count_and_vector_means():
    _, bounds = log_norm_layers(
        (8, 7, 5), mean_perm=3, rng=np.random.default_rng(3),
        return_indices=True,
    )
    assert bounds.size == 4 and bounds[0] == 0 and bounds[-1] == 5
    values, mean_bounds = log_norm_layers(
        (12, 10, 5), mean_perm=[10, 100, 30], indices=[0, 1, 4, 5],
        rng=np.random.default_rng(5), return_indices=True,
    )
    cube = values.reshape((12, 10, 5), order="F")
    for mean, lo, hi in zip([10, 100, 30], mean_bounds[:-1], mean_bounds[1:]):
        assert np.isclose(cube[:, :, lo:hi].mean(), mean)


def test_log_norm_two_dimensional_convenience_and_injected_draws():
    field = np.ones((5, 4, 3))
    first = log_norm_layers(
        (5, 4), indices=[0, 1], fields=[field], shifts=[.2]
    )
    second = log_norm_layers(
        (5, 4, 1), indices=[0, 1], fields=[field], shifts=[.2]
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("kwargs, message", [
    ({"dims": (3,)}, "dimensions"),
    ({"dims": (3, 3, 2), "indices": [0, 0, 2]}, "increase"),
    ({"dims": (3, 3, 2), "indices": [0, 1]}, "increase"),
    ({"dims": (3, 3, 2), "mean_perm": 3}, "requested"),
    ({"dims": (3, 3, 2), "mean_perm": [1, 2], "indices": [0, 2]}, "one mean"),
    ({"dims": (3, 3, 2), "indices": [0, 2], "fields": []}, "one draw"),
])
def test_log_norm_validates_layers_and_draws(kwargs, message):
    with pytest.raises(ValueError, match=message):
        log_norm_layers(**kwargs)
