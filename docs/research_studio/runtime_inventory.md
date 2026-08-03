# Research Studio runtime inventory

This inventory is the Foundation 0 baseline. Research Studio is additive: none
of these entry points currently imports or depends on it.

The canonical general-purpose numerical owner is `NonLinearSolver`. Its
`solve_ministep`, timestep-cutting and schedule methods own Newton decisions.
`simulate_schedule_ad` is a narrower two-phase legacy schedule loop with a
different return contract. Packed simulation adds persistence and restart
around a problem-owned simulator setup. Compositional examples contain a
separate example-oriented runner. Adjoint and optimization forward simulations
are intentionally specialized and must not be forced through a forward-only
session kernel.

The machine-readable classification is in `runtime_entrypoints.csv`; the call
relationships are in `runtime_call_graph.mmd`.

## Compatibility baseline

- No existing entry point changes its default parameters or return value.
- Initial integration will be opt-in and observation-only.
- Numerical state remains owned by the existing model and solver.
- Persistent evidence cannot contain live model, solver or arbitrary Python
  objects.
- Background processes require their own run manifest and event stream; an
  in-memory event bus cannot be shared across a process boundary.

## Progress-output owners

Direct progress output currently exists in `simulate_schedule_ad`,
`NonLinearSolver`, packed/background runners and several example drivers.
Foundation 1 will first reproduce useful messages from structured events;
existing output will only be removed after exact compatibility tests exist.
