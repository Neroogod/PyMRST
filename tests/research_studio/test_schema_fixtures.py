"""Golden-fixture tests for the persistent schemas (plan section 5.2).

Every other schema test in this suite builds a record in memory and reads it
back, which proves the round trip is self-consistent and nothing about stored
data. A schema change that breaks reading files written by an earlier version
passes all of them.

The fixtures in `fixtures/` are files that today's code actually wrote. They
are read-only history: a later version must either load them or refuse them
with a clear error, and either outcome has to be a deliberate decision rather
than a surprise in production.

Section 5.2 asks for four things per persistent schema. Where one does not
apply yet, that is stated rather than skipped:

* **source and target fixtures** - only v1 exists, so a fixture is its own
  target; a v2 fixture belongs here the day a v2 exists;
* **forward migration test** - no migration exists to test. The test below
  asserts that, so it starts failing the moment a second version appears
  without one;
* **rejection of unsupported versions** - tested;
* **corruption and truncation** - tested.
"""

import json
from pathlib import Path

import pytest

from pymrst.research_studio.evidence.report import (
    EVIDENCE_SCHEMA_VERSION, SimulationEvidence,
)
from pymrst.research_studio.storage.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, CheckpointCompatibilityError, CheckpointRecord,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_stored_evidence_v1_still_loads():
    """The fixture is a file, not a fresh object. That is the whole point."""
    data = _load("evidence_v1.json")
    evidence = SimulationEvidence.from_dict(data)
    assert evidence.run_identity.run_id == "run_golden_v1"
    assert evidence.convergence["iterations"] == [5, 4, 4]
    assert evidence.conservation["water"][2] == pytest.approx(3.4e-14)
    # Re-serialising must reproduce the stored file exactly, or the fixture has
    # silently stopped describing what the code writes.
    assert evidence.to_dict() == data


def test_stored_checkpoint_v1_still_loads():
    data = _load("checkpoint_v1.json")
    record = CheckpointRecord.from_dict(data)
    assert record.completed_report_steps == 2
    assert record.state["pressure"] == [2.7e7, 2.7e7]
    assert record.to_dict() == data


def test_fixtures_match_the_current_schema_version():
    """A version bump must come with a fixture for the new version.

    Without this, bumping `EVIDENCE_SCHEMA_VERSION` would leave the v1 fixture
    failing to load and the only signal would be a confusing assertion in the
    tests above.
    """
    assert _load("evidence_v1.json")["schema_version"] == EVIDENCE_SCHEMA_VERSION, (
        "evidence schema version changed; add a fixture for the new version and "
        "a migration, then keep this v1 fixture as the source case"
    )
    assert _load("checkpoint_v1.json")["schema_version"] == CHECKPOINT_SCHEMA_VERSION, (
        "checkpoint schema version changed; same requirement"
    )


def test_no_migrations_exist_yet_and_that_is_recorded():
    """Both schemas are at v1, so there is nothing to migrate.

    This is an assertion rather than an omission: the day a second version
    lands, this fails and points at the missing migration fixtures that section
    5.2 requires.
    """
    assert EVIDENCE_SCHEMA_VERSION == 1 and CHECKPOINT_SCHEMA_VERSION == 1, (
        "a schema moved past v1: section 5.2 now requires source and target "
        "fixtures, a forward migration test, and documented information loss"
    )


def test_a_future_version_is_refused_not_guessed():
    """Reading a newer file must fail loudly rather than partially succeed."""
    future = _load("evidence_v1.json") | {"schema_version": EVIDENCE_SCHEMA_VERSION + 1}
    with pytest.raises(ValueError, match="unsupported"):
        SimulationEvidence.from_dict(future)

    future_checkpoint = _load("checkpoint_v1.json")
    future_checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    with pytest.raises(CheckpointCompatibilityError, match="unsupported"):
        CheckpointRecord.from_dict(future_checkpoint)


@pytest.mark.parametrize("dropped", ["state", "schedule_fingerprint",
                                     "completed_report_steps", "code_fingerprint"])
def test_a_truncated_checkpoint_is_refused(dropped):
    """A field lost in transit must not read as a valid restart point.

    Silently defaulting a missing `state` or `completed_report_steps` would
    resume from the wrong place, which is worse than refusing to resume.
    """
    data = _load("checkpoint_v1.json")
    del data[dropped]
    with pytest.raises(CheckpointCompatibilityError, match="invalid checkpoint fields"):
        CheckpointRecord.from_dict(data)


def test_an_unknown_field_is_refused():
    """Extra fields mean the writer knew something this reader does not."""
    data = _load("checkpoint_v1.json") | {"invented_later": 1}
    with pytest.raises(CheckpointCompatibilityError, match="invalid checkpoint fields"):
        CheckpointRecord.from_dict(data)
