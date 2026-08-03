# ADR 0003: the observation hook contract

Status: accepted, recording established practice.

This ADR does not choose a design. It writes down the one already implemented
in three numerical owners and demonstrated by evidence, so the fourth
integration does not have to rediscover it.

## The contract

A numerical owner that wants to be observable takes a keyword-only `observer`
parameter defaulting to `None`, and calls it as `observer(event, payload)`.

* **No import of Research Studio.** The parameter is a plain callable. This is
  what keeps section 2.1's covenant true: `tests/research_studio/test_runtime_inventory.py`
  fails if any inventoried entry point imports `research_studio` at module
  scope.
* **Payload is copied scalars and plain containers only.** Never a state, a
  model, or an array the solver still holds. The guarantee is not that
  mutations are ignored; it is that there is nothing live to mutate.
  `freeze_characterization --hostile` asserts the payload surface stays within
  `{bool, int, float, str, dict, NoneType}` and fails if live state ever
  reaches a handler.
* **Handler exceptions are swallowed.** A reporting fault must never become a
  simulation fault.
* **Event names are shared across owners.** `report_step_started`,
  `ministep_started`, `step_accepted`, `step_rejected`, `timestep_cut`, so a
  handler written against one owner reads another without special cases. An
  owner may add fields, not rename shared ones: the packed runner adds
  `restarted_from`, which is the thing it uniquely decides.
* **The default changes nothing.** With `observer=None` the function behaves
  exactly as before, which section 2.1 requires of every replica-layer API.

## Why a hook per owner rather than one wrapper

`NonLinearSolver` is the canonical owner, but it is not the only one. The
fixed-stress splitting loop, `simulate_schedule_ad` and the packed runner each
keep their own Newton iteration or their own persistence, and none constructs a
`NonLinearSolver`. A single wrapper around the canonical solver reaches none of
them: before this contract was applied, three of seven characterization cases
produced no events at all and the gate could not be evidenced on those paths.

## What "evidenced" means for this contract

A passive recorder proves only that a well-behaved handler is harmless. Each
integration is instead checked with a handler that mutates every payload value,
clears nested containers and raises on some events, and the run must be
bit-identical to a baseline recorded without it. The check also asserts the
hook actually fired: without that, an integration that silently failed to
attach would report success.

Current evidence: 6 of 7 characterization cases unchanged under 3221 attempted
mutations and 228 exceptions across 688 events, plus
`test_the_observer_hook_is_observational` for the packed runner, which no
characterization case reaches.
