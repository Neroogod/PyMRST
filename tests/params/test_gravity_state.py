import numpy as np
import pytest

from pymrst.params import current_gravity, gravity, gravity_vector, isolated_gravity


@pytest.fixture(autouse=True)
def reset_gravity():
    gravity("reset")
    yield
    gravity("reset")


def test_reset_is_off_with_standard_z_direction():
    np.testing.assert_array_equal(gravity(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(gravity("on"), gravity_vector(3))


def test_string_controls_apply_left_to_right():
    np.testing.assert_allclose(gravity("x", "on"), [9.80665, 0.0, 0.0])
    np.testing.assert_array_equal(gravity("off", "y"), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(gravity("yes"), [0.0, 9.80665, 0.0])
    np.testing.assert_array_equal(gravity("reset", "true", "z"), gravity_vector(3))


def test_boolean_scalar_and_explicit_vectors_match_mrst_state_rules():
    gravity(True)
    np.testing.assert_allclose(gravity(2.5), [0.0, 0.0, 2.5])
    np.testing.assert_allclose(gravity([1.0, 2.0]), [1.0, 2.0, 0.0])
    np.testing.assert_allclose(gravity(np.array([1.0, 2.0, -3.0])), [1.0, 2.0, -3.0])
    np.testing.assert_array_equal(gravity(False, [4.0, 5.0]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(gravity(True), [4.0, 5.0, 0.0])


def test_zero_direction_rejects_subsequent_scalar_strength():
    gravity([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        gravity(4.0)


def test_current_gravity_returns_copy_and_dimension_slice():
    gravity("x", "on")
    value = current_gravity(2)
    value[:] = 0.0
    np.testing.assert_allclose(current_gravity(), [9.80665, 0.0, 0.0])
    with pytest.raises(ValueError, match="griddim"):
        current_gravity(4)


def test_isolated_gravity_restores_state_even_after_error():
    gravity("x", "on")
    with pytest.raises(RuntimeError):
        with isolated_gravity("reset", "y", "on") as temporary:
            np.testing.assert_allclose(temporary, [0.0, 9.80665, 0.0])
            raise RuntimeError("stop")
    np.testing.assert_allclose(gravity(), [9.80665, 0.0, 0.0])


def test_invalid_controls_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        gravity("down")
    with pytest.raises(ValueError, match="two or three"):
        gravity([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(TypeError, match="real number"):
        gravity(1.0j)
