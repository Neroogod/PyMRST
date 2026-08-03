"""Transactional and corruption tests for ResultStore v1."""

import numpy as np
import pytest

from pymrst.research_studio import CaseId, RunId, RunIdentity, SimulationEvidence
from pymrst.research_studio import EventKind, SimulationEvent
from pymrst.research_studio.runtime.identifiers import CheckpointId
from pymrst.research_studio.storage import HDF5ResultStore, StoreCorruptionError


def _store(tmp_path):
    store = HDF5ResultStore(tmp_path/"results.h5")
    run = RunId("run_test")
    store.begin_run(run, {"case": "water", "command": ["python3", "case.py"]})
    return store, run


def _evidence(run):
    return SimulationEvidence(RunIdentity(run, CaseId("case_water")))


def test_state_commit_roundtrip_and_lazy_reads(tmp_path):
    store, run = _store(tmp_path)
    state = {
        "pressure": np.array([1.0e7, 0.9e7]),
        "wellSol": [{"bhp": 8.0e6, "name": "P1"}],
        "flags": (True, False),
        "metadata": None,
    }
    store.commit_state(run, 0, state)
    restored = store.load_state(run, 0)
    np.testing.assert_array_equal(restored["pressure"], state["pressure"])
    assert restored["wellSol"] == state["wellSol"]
    assert restored["flags"] == state["flags"]
    lazy = store.lazy_states(run)
    assert len(lazy) == 1
    np.testing.assert_array_equal(lazy[-1]["pressure"], state["pressure"])


def test_rejected_attempt_is_separate_and_never_replaces_accepted(tmp_path):
    store, run = _store(tmp_path)
    store.commit_state(run, 0, {"pressure": np.array([10.0])})
    store.commit_state(run, 7, {"pressure": np.array([-99.0])}, accepted=False)
    np.testing.assert_array_equal(store.load_state(run, 0)["pressure"], [10.0])
    assert store.state_indices(run) == (0,)
    info = store.inspect_run(run)
    assert info["accepted_states"] == 1
    assert info["rejected_attempts"] == 1
    assert info["pending_artifacts"] == 0


def test_evidence_checkpoint_and_finalize_are_ordered(tmp_path):
    store, run = _store(tmp_path)
    checkpoint = CheckpointId("checkpoint_0")
    store.commit_checkpoint(run, checkpoint, {
        "state_index": 0, "elapsed": 10.0, "fingerprint": "abc123",
    })
    assert store.load_checkpoint(run, checkpoint)["fingerprint"] == "abc123"
    with pytest.raises(RuntimeError, match="evidence"):
        store.finalize_run(run, "completed")
    evidence = _evidence(run)
    store.commit_evidence(run, evidence)
    store.finalize_run(run, "completed")
    assert store.load_evidence(run).to_dict() == evidence.to_dict()
    assert store.inspect_run(run)["complete"]
    with pytest.raises(RuntimeError, match="finalized"):
        store.commit_state(run, 1, {"pressure": np.array([9.0])})


def test_duplicate_commit_and_unsupported_values_do_not_leave_pending_data(tmp_path):
    store, run = _store(tmp_path)
    store.commit_state(run, 0, {"pressure": np.array([10.0])})
    with pytest.raises(FileExistsError):
        store.commit_state(run, 0, {"pressure": np.array([11.0])})
    with pytest.raises(TypeError, match="unsupported"):
        store.commit_state(run, 1, {"model": object()})
    assert store.inspect_run(run)["pending_artifacts"] == 0


def test_corrupt_schema_and_sequence_are_rejected(tmp_path):
    import h5py

    store, run = _store(tmp_path)
    with h5py.File(store.path, "a") as file:
        file[f"runs/{run}"].attrs["store_schema_version"] = 999
    with pytest.raises(StoreCorruptionError, match="schema"):
        store.inspect_run(run)


def test_interrupted_pending_artifact_is_ignored_and_recoverable(tmp_path):
    import h5py

    store, run = _store(tmp_path)
    store.commit_state(run, 0, {"pressure": np.array([10.0])})
    with h5py.File(store.path, "a") as file:
        pending = file[f"runs/{run}/_pending"].create_group("interrupted")
        pending.attrs["pymrst_type"] = "mapping"
        pending.create_dataset("partial", data=np.array([99.0]))
    assert store.state_indices(run) == (0,)
    np.testing.assert_array_equal(store.load_state(run, 0)["pressure"], [10.0])
    assert store.inspect_run(run)["pending_artifacts"] == 1
    assert store.discard_pending_artifacts(run) == 1
    assert store.inspect_run(run)["pending_artifacts"] == 0


def test_event_log_is_transactional_contiguous_and_replayable(tmp_path):
    store, run = _store(tmp_path)
    for sequence, kind in enumerate((EventKind.RUN_STARTED, EventKind.RUN_COMPLETED)):
        store.commit_event(run, SimulationEvent(kind, run, sequence, float(sequence)))
    records = store.load_event_records(run)
    assert [record["kind"] for record in records] == ["run_started", "run_completed"]
    replayed = []
    assert store.replay_events(run, replayed.append) == 2
    assert replayed == list(records)
    with pytest.raises(FileExistsError):
        store.commit_event(run, SimulationEvent(EventKind.RUN_FAILED, run, 1, 2.0))


def test_states_round_trip_bit_for_bit_including_the_awkward_values(tmp_path):
    """A result store that loses precision corrupts every later conclusion.

    The existing round-trip test uses ordinary arrays, which cannot distinguish
    an exact store from one that quietly widens, narrows or normalizes. These
    are the values that do distinguish it, and each is one PyMRST actually
    produces:

    - transmissibilities around 1e-14, where a float32 round trip loses digits
      that matter;
    - denormals, which flush to zero under several storage settings;
    - NaN and both infinities, from a diverged step whose state is still worth
      keeping;
    - negative zero, which compares equal to zero while carrying a sign a
      later divide will read;
    - a float32 field, which must come back float32 rather than silently
      widened to float64.
    """
    store, run = _store(tmp_path)
    rng = np.random.default_rng(0)
    state = {
        "pressure": rng.random(37)*2.7e7,
        "transmissibility": rng.random(11)*1e-14,
        "huge": rng.random(5)*1e300,
        "cell_index": np.arange(-3, 4, dtype=np.int64),
        "active": np.array([True, False, True]),
        "denormal": np.array([5e-324, 1e-310, 0.0]),
        "diverged": np.array([np.nan, np.inf, -np.inf, -0.0]),
        "single": (rng.random(6)*1e3).astype(np.float32),
    }
    store.commit_state(run, 0, state)
    restored = store.load_state(run, 0)

    assert sorted(restored) == sorted(state)
    for name, original in state.items():
        original = np.asarray(original)
        got = np.asarray(restored[name])
        assert got.dtype == original.dtype, f"{name}: {original.dtype} became {got.dtype}"
        if original.dtype.kind == "f":
            # equal_nan, because a NaN that survives is the point; and the sign
            # bit checked separately, since -0.0 == 0.0 would hide its loss.
            assert np.array_equal(got, original, equal_nan=True), name
            assert np.array_equal(np.signbit(got), np.signbit(original)), \
                f"{name}: sign of zero not preserved"
        else:
            assert np.array_equal(got, original), name
