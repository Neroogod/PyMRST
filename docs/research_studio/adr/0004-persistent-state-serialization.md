# ADR 0004: persistent state serialization

Status: accepted, recording established practice.

## Decision

Durable data is JSON-compatible values and numpy arrays. No pickles, no
arbitrary Python objects, in any artifact that outlives a process: evidence,
checkpoints, failure bundles, characterization baselines.

## Why not pickle

A pickle binds stored data to the class layout that wrote it. Every rename or
field change silently invalidates history, and the failure appears at load time
in a later release rather than at write time. It is also an execution vector for
a file that arrives from elsewhere, which failure bundles are meant to do.

## What each store does with arrays

* **HDF5** writes numpy arrays as datasets and reads them back as arrays. This
  matters more than it sounds: an attempt to test resume by storing state as
  `.tolist()` produced lists on restore and a model that died on `list - float`.
  That was a wrong test, not a wrong store - but it shows the round-trip shape
  is worth stating rather than assuming.
* **Failure bundles** encode arrays as `.npy` members inside the archive with
  `allow_pickle=False`, and checksum every member.
* **Characterization baselines** are plain JSON with arrays flattened to lists,
  because they are meant to stay readable when the code that produced them no
  longer imports.

## Schema versioning

Every persistent schema carries an integer `schema_version` and refuses an
unknown one rather than guessing. Section 5.2 also requires golden fixtures and
migration tests; `tests/research_studio/test_schema_fixtures.py` holds files
this code actually wrote, and asserts that no schema has moved past v1 - so a
second version cannot land without the migration the plan asks for.

## The rule that keeps this honest

A record that stores nothing compares equal to anything. `record()` in the
characterization harness refuses to write a case with no state fields, and
refuses one with neither solver decisions nor metrics, because both conditions
produced baselines that reported success while testing nothing.
