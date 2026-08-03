# ADR 0001: additive session boundary

Status: accepted for initial implementation.

Research Studio owns orchestration, identifiers, evidence and persistence. The
MRST-shaped model and solver owners retain equations, convergence decisions,
state updates, controls and default return contracts. Integration is opt-in
through adapters or observation hooks. There is no reverse dependency from an
MRST module to browser, experiment, ML or external-simulator packages.

This boundary permits specialized packed, adjoint and optimization adapters
without pretending they share identical execution semantics.
