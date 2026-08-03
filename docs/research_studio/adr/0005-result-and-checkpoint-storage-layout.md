# ADR 0005: result and checkpoint storage layout

Status: accepted, recording established practice.

This records a decision the code already embodies and the measurements now
justify, in the manner of ADRs 0003 and 0004. It is not a policy invented here.
The four topics still outstanding in F0.2 — compatibility wrappers, component
and capability registration, experiment identity and hashing, and schema
versioning beyond v1 — remain genuine choices nobody has made, and are
deliberately not settled by this document.

## Decision

One HDF5 file per run store, holding four kinds of thing under a run:

* **states**, one dataset group per report step, written only when accepted;
* **checkpoints**, addressed by `CheckpointId`, each a self-describing record;
* **evidence**, one versioned JSON-safe document;
* **events**, an append-only contiguous sequence.

Accepted and pending artifacts are separated, so an interrupted run leaves
recoverable state rather than a half-written result that reads as complete.

## Why one file rather than a directory tree

A run is the unit that gets copied, archived and attached to a report. A single
file cannot be partially moved, and it cannot lose its checkpoints on the way to
a colleague. The counter-argument — that a directory allows independent
concurrent writers — does not apply: a run has one writer by construction, and
concurrency across runs is concurrency across files.

The one place this is deliberately not followed is the background child process,
which writes `events.jsonl` and `run_manifest.json` as plain files beside its
work directory. That code is replica-layer and the covenant forbids it importing
Research Studio, so it cannot use the store at all. Standard-library JSON Lines
is the compromise: readable by the extension, requiring nothing of it.

## Why the checkpoint is a record, not a file offset

A checkpoint carries the accepted state, elapsed time, completed report steps,
control index, and four fingerprints — schedule, model, solver and code. It is
enough to reconstruct a restart on its own, and it refuses to restore into a
model or solver it does not recognise.

The alternative, storing a position and re-deriving the rest, makes restart
correctness depend on the schedule object being identical at load time. That is
exactly the assumption that breaks quietly a release later.

## What the measurements say

Recorded by `scripts/freeze_characterization.py --perf`:

* a checkpoint is **3–129 kB** and commits in **0.8–1.0 ms**, and the commit
  time does not move with the payload. Norne's 129 kB checkpoint commits in the
  same time as a 3 kB one, 43 times smaller.
* a whole run's store is **242 kB–1010 kB** for the cases measured.
* session with store and per-report checkpointing costs **+0% to +46%** over
  the bare solve, the high figure on a 42 ms run and the low one on a 7.2 s
  run.

The consequence for cadence: **checkpoint frequency is bounded by store growth,
not by commit latency.** Committing every report step is affordable on any run
long enough to care about restarting, and the reason to checkpoint less often
is disk, not time. On very short runs the fixed session cost dominates
everything, which is an argument against instrumenting them rather than against
this layout.

## Fidelity

States round-trip bit for bit. This is checked on the values that distinguish an
exact store from one that quietly normalizes — transmissibilities at 1e-14,
denormals, NaN and both infinities, negative zero, and a float32 field that must
not widen — with dtype, exact equality and the sign bit asserted separately,
because `-0.0 == 0.0` would hide the last one. See
`tests/research_studio/test_hdf5_store.py`.

A store that loses precision corrupts every conclusion drawn from it afterwards,
and does so silently. That is why this is a layout decision and not an
implementation detail.

## Consequences

* `h5py` is required for the store and is declared as the `studio` extra, not
  a base dependency. Without it the store is unavailable and everything else
  still runs.
* Adding a fifth artifact kind means adding a group, not a file format.
* Any future change to what a checkpoint carries is a schema version bump under
  ADR 0004's rule, and needs the migration test that rule requires.
