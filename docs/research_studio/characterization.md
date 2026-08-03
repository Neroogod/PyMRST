# Foundation 0.3: the characterization baseline

This is the specification for F0.3. Nothing here is frozen yet; the ledger row
in `implementation.csv` stays `planned` until the artifacts below exist.

## Why this package is the keystone

Four Milestone 1 work packages cannot be accepted without it, and each is
blocked on the same missing thing rather than on any missing code:

| package | what it needs from F0.3 |
|---|---|
| M1.3 canonical nonlinear integration | a before/after comparison proving the `observer` hook changes no solver decision |
| M1.4 SimulationSession | baseline states and reports to compare each integrated entry point against |
| M1.6 checkpoint and restart | an uninterrupted run to prove resume equivalence against |
| M1.9 failure analysis | real residual histories, phase-114 and Norne specifically, to classify |

Milestone 1 is therefore built but not gated. The gap is not code; it is the
evidence that the code preserves existing behaviour.

## The work is assembly, not creation

The plan reads as though seven cases must be built from nothing.
`examples/validation/` already holds **236 MATLAB-validated reference `.mat`
files**, and five of the seven named families are covered:

| plan's case | existing reference | validator |
|---|---|---|
| single-phase water | `ref_p64_run2d_water_fixedstress.mat` | `validate_phase64.py` |
| immiscible oil-water | `ref_p65_run2d_oilwater_fixedstress.mat` | `validate_phase65.py` |
| black-oil SPE1 | `ref_p67_run2d_blackoil_fixedstress.mat`, 13 SPE1 validators | `validate_phase67.py` |
| compositional, natural and overall | `ref_p114_spe1_native_compositional.mat`, 11 more | `validate_phase114.py` |
| fixed-stress and fully coupled poromechanics | p64 / p65 / p67 above | as above |
| **one packed problem** | **none** | — |
| **one adjoint/optimization workflow** | **none** | — |

So F0.3 is: *designate* five existing cases as the frozen baseline and record
what the plan asks for around them, then add the two that are genuinely
missing. That is days of assembly, not weeks of new simulation.

## What "frozen" means here

Section 6 of the plan lists what each baseline case records. Restated as
concrete fields, so a harness can produce them and a reviewer can check them:

- **final state** — pressure, saturations, compositions where present;
- **selected intermediate states** — at minimum the first accepted step and the
  midpoint, so a divergence is locatable in time rather than only at the end;
- **report structure** — report-step and ministep counts, and their nesting;
- **solver decisions** — iterations per ministep, timestep cuts and their
  causes, achieved residuals per named equation;
- **controls and well histories** — rates, BHP, and every control switch with
  the step it occurred on;
- **conservation** — where an adapter exists; recorded as absent otherwise, not
  silently skipped;
- **runtime and peak memory** — as a budget for section 16, not as a pass
  criterion;
- **hashes** — of the input deck or case constructor, of the reference `.mat`,
  and of the PyMRST commit;
- **the exact command** that reproduces the case.

## Two rules that decide whether this is worth anything

**Record what the solver decided, not only what it produced.** Two runs can
agree on every final state and disagree on iteration counts, cut points and
achieved residuals. An integration that changes solver decisions has changed
behaviour even when the answer matches, and a state-only baseline cannot see
it. This is what M1.3's gate is actually asking about.

**A baseline must be able to fail.** A characterization case that passes
because it compares a value to itself, or because its tolerance is wider than
the effect being guarded, is worse than no case: it reports safety it has not
established. Each frozen case needs a recorded demonstration that perturbing
the thing it guards makes it fail. Two concrete traps already met in this
repository:

- transmissibilities are around 1e-14 in SI, and `np.isclose`'s default
  `atol=1e-8` calls any two of them equal — every comparison must be relative
  with `atol=0`;
- a self-test that runs green in a terminal can block forever under the
  subprocess harness, so "it passed when I ran it" is not evidence about how it
  behaves where it actually runs.

## Order

1. Freeze the five covered families from their existing references, states and
   solver decisions both.
2. Add the packed-problem case; `simulate_packed_problem` is already classified
   in `runtime_entrypoints.csv` as a specialized session adapter.
3. Add the adjoint/optimization case; three `simulate_forward` variants are
   classified as intentionally specialized and must not be forced through the
   forward-only session kernel.
4. Only then re-open M1.3, M1.4, M1.6 and M1.9 for acceptance.

## Status

Specification only. No case is frozen, no harness is written, and
`implementation.csv` records F0.3 as `planned` accordingly.
