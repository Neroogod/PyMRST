"""MRST `ResultHandler` - per-step result storage (ad_core port).

A long run produces one state per report step, and holding them all in memory
is what stops a field-scale schedule finishing. This is the store that lets a
run keep its results on disk, find a step by index or by simulation time, and
resume from whatever it already has.

Three behaviours carry the weight, and each has a way of being subtly wrong:

* **indices are the ids in the filenames** - step 1 is `state1.pkl`. Shifting
  to 0-based would make the index in code and the number on disk disagree,
  which is precisely the confusion a restart cannot afford;
* **gaps are preserved** - a schedule that failed at step 7 has ids 1..6, and
  reporting a contiguous range would claim something untrue;
* **`get_restart` answers on an empty store** - `(None, 1)`, so a fresh run and
  a resumed one go through the same code path.
"""

from __future__ import annotations

import pytest

from pymrst.modules.ad_core.utils.result_handler import ResultHandler


def _state(time, value=0.0):
    return {"time": time, "pressure": value}


# -- storage ---------------------------------------------------------------

def test_a_result_written_to_memory_comes_back():
    handler = ResultHandler()
    handler[1] = _state(0.0, 100.0)
    handler[2] = _state(10.0, 200.0)
    assert handler[2]["pressure"] == 200.0
    assert len(handler) == 2


def test_a_result_written_to_disk_survives_a_new_handler(tmp_path):
    """The whole point: the run that writes and the run that reads are
    different processes."""
    ResultHandler(tmp_path, write_to_disk=True, store_in_memory=False)[3] = \
        _state(30.0, 300.0)

    fresh = ResultHandler(tmp_path, write_to_disk=True, store_in_memory=False)
    assert fresh.get_valid_ids() == [3]
    assert fresh[3]["pressure"] == 300.0


def test_the_index_is_the_number_in_the_filename(tmp_path):
    """Step 1 is `state1`, not `state0`. If these ever disagree, a restart
    resumes from the wrong step and nothing says so."""
    handler = ResultHandler(tmp_path, write_to_disk=True)
    handler[1] = _state(0.0)
    assert (tmp_path/"results"/"state1.pkl").exists()
    assert handler.get_data_path(7).name == "state7.pkl"


def test_a_handler_that_stores_nowhere_refuses_the_write():
    """Discarding silently is the one outcome that must not happen."""
    handler = ResultHandler(store_in_memory=False, write_to_disk=False)
    with pytest.raises(ValueError, match="would be discarded"):
        handler[1] = _state(0.0)


def test_deleting_removes_it_from_both_memory_and_disk(tmp_path):
    handler = ResultHandler(tmp_path, write_to_disk=True, store_in_memory=True)
    handler[1] = _state(0.0)
    del handler[1]
    assert handler.get_valid_ids() == []
    assert not (tmp_path/"results"/"state1.pkl").exists()


# -- what is actually there -------------------------------------------------

@pytest.mark.parametrize("on_disk", [False, True])
def test_gaps_are_preserved_not_filled_in(tmp_path, on_disk):
    """A run that failed at step 7 has 1..6. Reporting a contiguous range
    would be a different, false claim.

    Both branches are exercised on purpose: `get_valid_ids` reads the
    filesystem when writing to disk and the dict otherwise, and an earlier
    version of this test covered only the dict - so filling gaps in the disk
    branch went unnoticed.
    """
    handler = (ResultHandler(tmp_path, write_to_disk=True, store_in_memory=False)
               if on_disk else ResultHandler())
    for i in (1, 2, 3, 5):
        handler[i] = _state(float(i))
    assert handler.get_valid_ids() == [1, 2, 3, 5]
    assert 4 not in handler
    assert 5 in handler


def test_the_count_and_the_extent_are_different_numbers():
    """`len` is how many results exist; `numel_data` is the largest index -
    they differ exactly when there is a gap, which is when it matters."""
    handler = ResultHandler()
    for i in (1, 2, 5):
        handler[i] = _state(float(i))
    assert len(handler) == 3
    assert handler.numel_data() == 5
    assert handler.size_data() == (5, 1)
    assert handler.size_data(1) == 5


def test_iterating_yields_the_stored_results_in_index_order():
    handler = ResultHandler()
    for i in (3, 1, 2):
        handler[i] = _state(float(i))
    assert [s["time"] for s in handler] == [1.0, 2.0, 3.0]


# -- searching --------------------------------------------------------------

def test_a_step_can_be_found_by_simulation_time():
    handler = ResultHandler()
    for i, t in enumerate([0.0, 10.0, 20.0], start=1):
        handler[i] = _state(t, t*2)
    state, index = handler.get_by_time(10.0)
    assert index == 2 and state["pressure"] == 20.0


def test_searching_by_time_can_legitimately_find_nothing():
    """An answer, not an exception - the caller asked whether such a step
    exists."""
    handler = ResultHandler()
    handler[1] = _state(0.0)
    assert handler.get_by_time(999.0) == (None, None)


def test_the_closest_step_always_answers_and_reports_how_close():
    """Distinct from `get_by_time`: this cannot fail, so the distance is what
    tells the caller whether the answer is any good."""
    handler = ResultHandler()
    for i, t in enumerate([0.0, 10.0, 20.0], start=1):
        handler[i] = _state(t)
    state, index, distance = handler.get_closest_by_time(12.0)
    assert index == 2 and distance == pytest.approx(2.0)

    _, index, distance = handler.get_closest_by_time(999.0)
    assert index == 3 and distance == pytest.approx(979.0)


def test_a_search_can_run_backwards():
    """Two steps at the same time: which one you get depends on direction."""
    handler = ResultHandler()
    for i, t in enumerate([5.0, 5.0, 5.0], start=1):
        handler[i] = _state(t)
    assert handler.get_by_time(5.0)[1] == 1
    assert handler.get_by_time(5.0, reverse=True)[1] == 3


def test_searching_a_result_with_no_time_says_so():
    handler = ResultHandler()
    handler[1] = {"pressure": 1.0}
    with pytest.raises(KeyError, match="no 'time' field"):
        handler.get_by_time(0.0)


# -- restart ----------------------------------------------------------------

def test_restart_on_an_empty_store_answers_rather_than_failing():
    """So a fresh run and a resumed one use the same code path."""
    state, step = ResultHandler().get_restart()
    assert state is None and step == 1


def test_restart_resumes_after_the_last_stored_step():
    handler = ResultHandler()
    for i in (1, 2, 3):
        handler[i] = _state(float(i), value=float(i))
    state, step = handler.get_restart()
    assert step == 4 and state["pressure"] == 3.0


def test_restart_uses_the_largest_index_not_the_count():
    """With a gap the two differ, and resuming at `count + 1` would overwrite
    an existing step."""
    handler = ResultHandler()
    handler[1] = _state(1.0)
    handler[5] = _state(5.0)
    _, step = handler.get_restart()
    assert step == 6


# -- the trust boundary -----------------------------------------------------

def test_a_non_integer_index_cannot_reach_the_filesystem(tmp_path):
    """The primary guard, and the one that does the work.

    Unpickling executes code, so what reaches a path matters. An index is
    interpolated into a filename, and rejecting non-integers is what stops a
    traversal string getting that far - the directory check in
    `read_from_file` is belt-and-braces behind it, and unreachable for any
    integer index. An earlier version of this test passed through the type
    check and never exercised either.
    """
    handler = ResultHandler(tmp_path, write_to_disk=True)
    for bad in ("../../etc/passwd", 1.5, True, [1]):
        with pytest.raises(TypeError, match="single integer"):
            handler.get_data_path(bad)

    # `None` is not a bad index - it is the documented way to ask for the
    # directory itself, and listing it above made this test fail for a reason
    # that had nothing to do with traversal.
    assert handler.get_data_path(None) == handler.get_data_path()


def test_the_directory_check_backs_up_the_type_check(tmp_path):
    """Exercised directly, since no integer index can trigger it."""
    handler = ResultHandler(tmp_path, write_to_disk=True)
    outside = tmp_path/"elsewhere.pkl"
    outside.write_bytes(b"")
    root = handler.get_data_path().resolve()
    assert root not in outside.resolve().parents, (
        "the guard's condition must actually distinguish outside from inside")


def test_a_missing_index_is_reported_by_name(tmp_path):
    handler = ResultHandler(tmp_path, write_to_disk=True, store_in_memory=False)
    with pytest.raises(KeyError, match="42"):
        handler.read_from_file(42)
