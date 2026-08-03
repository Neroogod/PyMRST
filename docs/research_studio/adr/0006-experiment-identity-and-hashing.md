# ADR 0006: experiment identity and hashing

Status: accepted.

## Context

An experiment needs a stable scientific identity, while execution caching needs
to answer a narrower question: whether two definitions request the same
simulation. Machine-specific dataset paths must not make otherwise identical
experiments differ, but silently changing dataset contents must.

## Decision

ExperimentDefinition v1 has two SHA-256 identities over strict canonical JSON:

- `experiment_id` covers the full scientific definition, including hypothesis,
  description, validation requirements, outputs and publication artifacts;
- `execution_fingerprint` covers only simulation-affecting content: case,
  model, schedule, explicit variants, dataset content identities, execution
  profile and random seed.

A dataset contributes its logical name, SHA-256 digest, size and media type.
Its source path is persisted as provenance but excluded from both hashes.

Canonical JSON permits only null, strings, booleans, integers, finite floats,
string-keyed mappings and sequences. Mapping keys are sorted, insignificant
whitespace is removed and negative zero is normalized. NaN, infinities,
arbitrary Python objects and implicit object serialization are refused.

Experiment and member identifiers contain the complete digest rather than a
truncated prefix. Explicit variant member identity includes the experiment ID,
ordinal, name, comparison group, parameter patch and component selection.

## Consequences

- Moving an unchanged dataset does not invalidate identity or execution cache.
- Changing dataset bytes invalidates both identities.
- Editing the hypothesis changes scientific identity but may reuse an identical
  execution result.
- Variant ordering is meaningful and changes member identities.
- Content-address collisions are checked and refused by the registry.
- Cartesian and sampled expansion remain M3.2 work; M3.1 supports named explicit
  variants only.
