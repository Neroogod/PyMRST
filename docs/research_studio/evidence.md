# The evidence layer: how it is checked, and what the checks are worth

Documentation for Milestones 1.4 through 1.10 — the session, the store,
checkpoints, conservation, invariants, failure analysis and failure bundles.
Package status lives in `implementation.csv`; this explains how the evidence
behind those rows is produced and how to read it.

## Running the checks

Every gate is a mode of one script, and each runs over the frozen
characterization cases:

```
python3 scripts/freeze_characterization.py --check       # F0.3
python3 scripts/freeze_characterization.py --observed    # M1.3
python3 scripts/freeze_characterization.py --hostile     # M1.2
python3 scripts/freeze_characterization.py --resume      # M1.6
python3 scripts/freeze_characterization.py --bundle      # M1.10
python3 scripts/freeze_characterization.py --classify    # M1.9
python3 scripts/freeze_characterization.py --conserve    # M1.7
python3 scripts/freeze_characterization.py --invariants  # M1.8
python3 scripts/freeze_characterization.py --perf        # section 16
```

Each accepts case names to narrow the run (`--check p64_water`). Baselines are
recorded with `--freeze` and live in `characterization/` as JSON — never
pickles, per ADR 0002, so a baseline stays readable when the code that produced
it no longer imports.

## What each mode covers, and why the rest is uncovered

| mode | gate | passing | not applicable |
|---|---|---|---|
| `--check` | F0.3 | 8 | 0 |
| `--observed` | M1.3 | 7 | 1 |
| `--hostile` | M1.2 | 7 | 1 |
| `--resume` | M1.6 | 5 | 3 |
| `--bundle` | M1.10 | 5 | 3 |
| `--classify` | M1.9 | 17 | 2 |
| `--conserve` | M1.7 | 4 | 4 |
| `--invariants` | M1.8 | 12 | 1 |
| `--perf` | section 16 | 8 | 0 |

**A case reported `n/a` is not an untested claim.** Every exclusion prints its
own reason, because a generic message invites someone to do work that is either
impossible or forbidden. The recurring ones:

- `packed_compositional` constructs no instrumented owner, so the observation
  hook has nothing to attach to. Its runner is also a deprecated duplicate per
  F0.1, which is why residual histories and conservation are absent too.
- Three of the eight cases build their schedule internally or run their own
  splitting loop, so a session cannot drive them — that bounds `--resume` and
  `--bundle` at five.
- `p65_oilwater` cannot be conserved as configured: its own `PoreVolume` state
  function is negative in 74 of 80 cells. Not a porting defect — it matches its
  MATLAB reference to 1.7e-09 on pressure.
- `blackoil_one_phase` runs two independent settings whose states are
  concatenated, so consecutive states straddle a seam. Its `WaterModel` fits
  `WaterConservationAdapter` exactly; the state sequence is the problem.

## Two disciplines these numbers depend on

**Every check is verified by making it fail.** Not once as a ritual — each
detector, adapter and comparison has had the specific defect it guards
reintroduced, and must fail then and pass on restore. This rule was earned:
three separate versions of this harness passed while testing nothing, recording
no state fields, comparing empty decision lists, and attaching an observer that
never fired.

Two practical traps in doing this:

- **Anchor uniqueness.** A source mutation matched by text must occur exactly
  once. A mutation written to falsify `--resume` matched an identical line in
  `check_bundle` first, so `--resume` kept passing and looked vacuous on all
  five cases. It was not. Both call sites now share `_checkpoint_at`.
- **Stale bytecode.** These mutations are often byte-length preserving (`>` to
  `<`, `[0]` to `[1]`), and so is the restore. Python reused a cached `.pyc`
  carrying the previous mutation. Clear `__pycache__` between mutations.

**Numbers are measured, not asserted.** `--conserve` and `--perf` report
figures and fail only on gross violations, because section 16 sets budgets
*after* Foundation 0 measurement. Choosing a threshold now would decide the
question instead of answering it. What the measurements say so far:

- conservation imbalance is at machine precision on the fully implicit cases
  — 9.89e-17 compositional, 1.74e-16 water, 9.72e-08 Norne oil-water — and
  4.17e-06 on fixed-stress water. That single figure is what a budget has to
  account for, and its cause is **not established**. It is invariant under
  every tolerance the model has: `tol_mb`, `tol_cnv`, `splitting_tol` across
  six orders, and removing the post-convergence mechanics re-solve. So it is
  structural rather than a convergence error, and that is the whole of what is
  known.

  The most useful thing measured about it is a ratio: the raw imbalance is a
  constant **rate** of 2.895e-09 m³/s, identical across steps of different
  length, and that is about **a quarter of the well rate**. The fraction is
  stable under grid refinement (0.246–0.251 across nx 10–40, ny 2–8) and under
  compressibility (0.222–0.267 while c_fluid/c_rock spans 0.125–0.75), so a
  roughly fixed quarter of injected volume does not reach the inventory and it
  is not a discretization artifact. Two tempting readings of the quarter are
  already excluded: it is not `1/ny`, and it is not `c_fluid/c_rock` — both
  matched the default case exactly and neither survived varying it.

  Five causes have been tested and refuted: poromechanical coupling, operator
  splitting, the Newton tolerances, the post-convergence mechanics re-solve,
  and reading `FixedStressPoreVolume` with the wrong fixed-stress terms. Two of
  them were written down as fact before being tested. The record now says "not
  known" and lists what is ruled out, which is worth more than a sixth guess.

  This entry previously read that the spread across families was ten orders of
  magnitude and reflected the poromechanical coupling. It did not. It was an
  artifact of `WaterConservationAdapter` looking for `b_w` on a fluid that
  indexes b per phase, finding nothing, and silently using `b = 1` where the
  model's own b is 1.0000001. Taking b from the state-function graph — what
  the residual is actually assembled from — moved the fully coupled case from
  5.25e-07 to 1.74e-16. What ruled out the alternatives: tightening `tol_mb`
  by two orders moved the number not at all, and the raw imbalance was exactly
  proportional to `dt`, which means a constant rate discrepancy rather than
  accumulated error.

  The same three signals still hold for the fixed-stress case, and no cause has
  been found for it. Ruling something out is a result; naming a cause without
  an experiment is not.
- observation costs nothing measurable: −6% to +6% against spreads of 0–21%,
  on runs from 42 ms to 7.3 s.
- a checkpoint is 3–129 kB and commits in 0.8–1.0 ms regardless of size, so
  restart cadence is bounded by store growth rather than commit latency.
- the observe profile — invariants and conservation, evaluated once per
  accepted state — costs −1% to +1% on top of the session that already stores
  and checkpoints. Both sit on the critical path, so a session configured
  without them measures the one arrangement that avoids the cost.

A sign that cannot happen is the most useful error signal here. Attaching an
observer cannot make a run faster and adding a store cannot either; two
readings of *negative* overhead both turned out to be a solver substituted for
another, not a real speedup.

## Per-package notes

**M1.4 SimulationSession.** Six of eleven runtime entry points carry the
additive observation hook; a seventh, `ad_blackoil.examples.utils.run_schedule`,
needs none of its own because it takes a solver and delegates every step to it.
The remaining four are excluded by design — one deprecated duplicate and three
`simulate_forward` workflows the plan forbids forcing through a forward-only
session kernel. Numerical characterization runs MRST's 2D poromechanics case
(80 cells, 210 displacement degrees of freedom) natively and through the
session, requiring every state field bit-identical.

**M1.5 result store.** States round-trip bit for bit, checked on the values
that distinguish an exact store from one that quietly normalizes:
transmissibilities at 1e-14, denormals, NaN and both infinities, negative zero,
and a float32 field that must not widen. The sign bit is asserted separately —
`-0.0 == 0.0` would hide its loss.

**M1.6 checkpoint and restart.** Resume from a mid-schedule checkpoint
reproduces the uninterrupted run exactly, each case through the driver it
really uses. Note what this cannot check: the uninterrupted, prefix and resumed
runs all go through the same callable, so any driver makes the comparison
self-consistent. Using each case's real driver is a statement about what is
being characterized, not a defence the check enforces.

**M1.7 conservation.** Four model families measured. Adapters are chosen per
family rather than inferred, because conservation semantics are a property of
the model, not something to guess globally.

The shrinkage factor comes from the model's state-function graph before any
fluid's `b_w`, and that order is the whole lesson of this package. A
poromechanics `PhaseFluid` has no `b_w`; the previous lookup found nothing,
silently used `b = 1`, and produced a plausible imbalance computed from a
property the model never supplied. It stood as recorded evidence of
poromechanical coupling until someone asked what the number was actually made
of. Both branches of the lookup are now pinned by tests — the fallback path
had no coverage at all, and disabling it broke nothing.

**M1.8 invariants.** Observe and strict profiles are evidenced as distinct —
a framework where both behave identically would pass a naive check.

**M1.9 failure analysis.** Every classifier the work package names has a
negative control built to be specific: the oscillating history improves 99%
across its last window so it cannot also register as a stall. Real-failure
validation is a Norne solve starved to a three-iteration cap. Pursuing it found
that `solve_timestep` built a fresh report on both its cut-success and failure
exits and left `residuals` empty — so the one path failure analysis exists for
was the one path carrying no evidence.

**M1.10 failure bundles.** Bundles are written from real runs, reloaded from
disk and replayed without executing the earlier schedule. No pickles: arrays
are stored as `.npy` members with per-member checksums.

## Capabilities added after the first review

Five packages were built on top of the evidence layer above. Each is additive
and optional, and each is verified the same way — by reintroducing the defect
it guards.

**`pymrst doctor` (F0.5).** What the installation can actually do: ten
capabilities, CUDA, writable directories, each missing one naming the exact
`pip install` that fixes it. Capability probing uses `find_spec` rather than
`import`, so reporting that torch exists does not pay for loading it and a
broken optional package cannot take the diagnosis down. `pymrst datasets`
reports provenance and shows terms before fetching; several datasets are not
redistributable and are refused rather than downloaded.

**VTK/ParaView export (M3.5).** `write_vtk_series` emits a `.pvd` collection
over per-step `.vtu` files indexed by simulation time, wells as a polyline
`.vtp`, with the polyhedral cell array built once rather than per step.
`read_vtk_series` exists so the round trip is verified rather than claimed —
values compare **bit-identical**, not within a tolerance, because `allclose`
would accept a format that silently narrowed to float32.

**Grid inspection (M2.11).** `validate()` returns ordered findings rather than
a boolean, splitting *invalid* — the discretization cannot be trusted — from
*unusual*, which is legal and probably unintended. `explain_connection(a, b)`
gives a reason rather than a negative: sharing no face is a different
situation from lying in a separate flow region, and the second is what makes a
well drain a pocket. Read-only, asserted by comparing arrays before and after.

**Diagnostic narrative (M1.11).** Turns M1.9's classification into a stated
cause and actions naming the setting and its current value. Every statement
carries the evidence fields it was built from, and a test asserts every number
in the prose appears in that evidence. A language model may rephrase a
finished narrative; it may not originate one.

**Dimensional safety (M2.12).** `BARSA` is `100000.0` and `DARCY` is
`9.87e-13` — both plain floats, indistinguishable once passed. The caller
supplies the dimension, which is what makes this opt-in *by construction*
rather than by flag: a plain float passes through untouched and every existing
call site is unaffected without anyone having to be careful.

## Architecture decisions

`adr/` holds five accepted records. Three of them — the observation hook
contract, persistent state serialization, and the result and checkpoint
storage layout — record decisions the code already embodies and the
evidence demonstrates. Four topics remain deliberately unwritten:
compatibility wrappers, component and capability registration, experiment
identity and hashing, and schema versioning beyond v1. Those are genuine
choices nobody has made, and an ADR written unilaterally would decide them
by default rather than record them.

## The covenant

The replica layer must not depend on Research Studio. This is checked, not
assumed:

```
grep -rl research_studio pymrst/modules pymrst/gridprocessing pymrst/solvers
```

must return nothing. Where replica-layer code needs to emit evidence — the
background runner's child process writes `events.jsonl` and
`run_manifest.json` — it does so with the standard library alone. The extension
reads those files without either side importing the other.
