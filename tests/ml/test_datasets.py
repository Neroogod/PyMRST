"""Saved-state pair dataset (F0.4 risk list).

The whole purpose of this class is the *pairing*: item `i` must be
(state at step i, state at step i+1). Get that wrong - return the same step
twice, or shift both by one - and nothing raises. Training proceeds, the loss
falls, and the model learns the identity map or a step it was never asked to
predict. There is no downstream check that catches it, which is why the pairing
is asserted here against values chosen so every step is distinguishable.

The off-by-one at the end is the same story: `len` must be `n_steps - 1`,
because the final state has no successor to be paired with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymrst.ml.datasets import SavedStateDataset

FIELDS = ("pressure", "sW")


def _saved(tmp_path, steps=5, cells=4, with_time=True, **extra) -> Path:
    """Values encode their own step, so a mispaired item is unmistakable."""
    arrays = {
        "pressure": np.array([[100.0 + step for _ in range(cells)]
                              for step in range(steps)]),
        "sW": np.array([[0.1*step for _ in range(cells)] for step in range(steps)]),
    }
    if with_time:
        arrays["time"] = np.arange(steps, dtype=float)*10.0
    arrays.update(extra)
    path = tmp_path/"states.npz"
    np.savez(path, **arrays)
    return path


def _dataset(tmp_path, **kwargs):
    return SavedStateDataset(_saved(tmp_path), input_fields=FIELDS, **kwargs)


def test_an_item_pairs_a_step_with_the_next_one(tmp_path):
    """The property the class exists for."""
    data = _dataset(tmp_path)
    for index in range(len(data)):
        x, y, _ = data[index]
        np.testing.assert_allclose(x[0], 100.0 + index)
        np.testing.assert_allclose(y[0], 100.0 + index + 1)
        np.testing.assert_allclose(x[1], 0.1*index)
        np.testing.assert_allclose(y[1], 0.1*(index + 1))


def test_the_length_is_one_less_than_the_number_of_steps(tmp_path):
    """The last state has no successor, so it is not a training pair."""
    data = _dataset(tmp_path)
    assert data.n_steps == 5
    assert len(data) == 4


def test_reading_past_the_end_raises_rather_than_wrapping(tmp_path):
    """Backed by numpy, so this half of the guard is belt-and-braces.

    Removing the explicit bounds check leaves this green: a too-large index
    reaches `self.data[name][index]` and numpy raises `IndexError` itself. The
    guard earns its place on the *negative* side instead, where numpy would
    happily accept the index - see the next test.
    """
    data = _dataset(tmp_path)
    with pytest.raises(IndexError):
        data[len(data)]
    with pytest.raises(IndexError):
        data[99]


def test_negative_indices_count_from_the_end(tmp_path):
    data = _dataset(tmp_path)
    # The load-bearing half of the bounds check. Without it, `data[-9]` becomes
    # `data[-5]` after the offset, which numpy accepts as a valid negative
    # index and silently returns a real-looking pair from the wrong end -
    # confirmed by removing the check and reading pressure 100.0 back.
    last_x, last_y, _ = data[-1]
    same_x, same_y, _ = data[len(data) - 1]
    np.testing.assert_array_equal(last_x, same_x)
    np.testing.assert_array_equal(last_y, same_y)
    with pytest.raises(IndexError):
        data[-len(data) - 1]


def test_the_stacked_fields_keep_their_declared_order(tmp_path):
    """Row order is the model's input contract; swapping it silently feeds
    saturation where pressure is expected."""
    data = SavedStateDataset(_saved(tmp_path), input_fields=("sW", "pressure"))
    x, _, _ = data[2]
    np.testing.assert_allclose(x[0], 0.1*2)
    np.testing.assert_allclose(x[1], 100.0 + 2)


def test_targets_default_to_the_input_fields(tmp_path):
    data = _dataset(tmp_path)
    assert data.target_fields == FIELDS


def test_targets_may_differ_from_inputs(tmp_path):
    data = SavedStateDataset(_saved(tmp_path), input_fields=("pressure", "sW"),
                             target_fields=("pressure",))
    x, y, _ = data[0]
    assert x.shape[0] == 2 and y.shape[0] == 1
    np.testing.assert_allclose(y[0], 101.0)


def test_the_metadata_reports_both_times_from_the_file(tmp_path):
    data = _dataset(tmp_path)
    _, _, meta = data[2]
    assert meta["step"] == 2
    assert meta["time"] == pytest.approx(20.0)
    assert meta["target_time"] == pytest.approx(30.0)


def test_a_file_without_times_falls_back_to_step_numbers(tmp_path):
    data = SavedStateDataset(_saved(tmp_path, with_time=False),
                             input_fields=FIELDS)
    _, _, meta = data[1]
    assert meta["time"] == 1 and meta["target_time"] == 2


def test_a_missing_field_is_refused_by_name(tmp_path):
    with pytest.raises(KeyError, match="rs"):
        SavedStateDataset(_saved(tmp_path), input_fields=("pressure", "rs"))


def test_a_field_that_is_not_time_by_cell_is_refused(tmp_path):
    path = _saved(tmp_path, flat=np.arange(4.0))
    with pytest.raises(ValueError, match="time-by-cell"):
        SavedStateDataset(path, input_fields=("pressure", "flat"))


def test_the_requested_dtype_is_applied(tmp_path):
    data = _dataset(tmp_path, dtype="float32")
    x, y, _ = data[0]
    assert x.dtype == np.float32 and y.dtype == np.float32


def test_grid_metadata_is_selected_by_prefix(tmp_path):
    path = _saved(tmp_path, **{"grid.cells.num": np.array([4]),
                               "rock.poro": np.full(4, 0.2),
                               "well.cells": np.array([0, 3]),
                               "unrelated": np.zeros(3)})
    data = SavedStateDataset(path, input_fields=FIELDS)
    metadata = data.grid_metadata()
    assert set(metadata) == {"grid.cells.num", "rock.poro", "well.cells"}


def test_the_dataset_owns_its_arrays_after_the_file_is_closed(tmp_path):
    """Pins the outcome; the `.copy()` that appears to produce it is redundant.

    `np.load` without `mmap_mode` materialises each array on access inside the
    `with` block, so dropping the copy leaves this green. It is kept as a
    regression guard for the day someone adds `mmap_mode` for large files,
    where a dataset holding views into a closed handle would fail only once
    training starts.
    """
    path = _saved(tmp_path)
    data = SavedStateDataset(path, input_fields=FIELDS)
    path.unlink()
    x, y, _ = data[0]
    np.testing.assert_allclose(x[0], 100.0)
