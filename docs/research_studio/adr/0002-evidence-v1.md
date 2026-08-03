# ADR 0002: SimulationEvidence v1

Status: accepted for initial implementation.

The first persistent envelope uses dataclasses and strict JSON-compatible
values. It stores identifiers and numerical evidence, never live model or
solver objects and never arbitrary pickles. Version 1 rejects unknown envelope
fields, unsupported versions and non-finite numbers. Migrations will be
explicit functions backed by source/target fixtures; silently accepting a
future schema is forbidden.

Events are immutable, monotonically sequenced records. Their top-level payload
is read-only to handlers. Deep JSON validation occurs before persistence.
