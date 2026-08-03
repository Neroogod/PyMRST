# PyMRST Research Studio: implementation plan

> **Status.** Machine-readable in `docs/research_studio/implementation.csv`
> (48 work packages, the columns and status vocabulary of section 18). As of
> the last update: **2 accepted, 36 validating, 6 implementing, 4 planned**.
> Counts are read out of the CSV, not maintained by hand - the previous line
> had drifted six packages behind it.
>
> Eight packages were added after review, in two groups.
>
> *Interoperability:* VTK/ParaView export (M3.5, already largely built
> underneath), ECLIPSE result ingestion (M2.6, whose original wording assumed
> readers that do not exist), CMG ingestion (M2.10) and outbound deck/result
> export (M3.8 - the first *writer* in the plan, and the one package defined
> by what it refuses).
>
> *Usability and safety:* diagnostic narrative (M1.11, on top of M1.9's
> finished classifier), grid inspection (M2.11), dimensional safety (M2.12)
> and a portable case definition (M3.9), plus packaging and environment
> reporting (F0.5). Two proposals were considered and declined, with reasons
> recorded in section 2.1 rather than left implicit.
>
> One review proposed building the characterization baseline before any
> performance work. That baseline **already exists** - it is F0.3, and the
> harness below is it - so the sequencing constraint it was protecting is
> satisfied, and the risky assembly work is less blocked than that review
> assumed. What the baseline does not yet record is listed in the F0.3
> addendum, and the Jacobian structural hash is the item to add before any
> sparse-assembly change.
>
> Six packages moved off `planned` after that review and now carry evidence:
> F0.5 packaging (`pymrst doctor`), M3.5 VTK/ParaView export, M2.11 grid
> inspection, M1.11 diagnostic narrative and M2.12 dimensional safety, plus
> the Jacobian structural signature the F0.3 addendum called for - the
> prerequisite before any sparse-assembly work, since a rewrite that alters
> the pattern rather than the values was invisible to every previous check.
>
> All ten Milestone 1 packages and three of four Foundation 0 packages carry
> evidence. F0.1 and M2.5 are *accepted*: section
> 18's rule is that a package is not accepted until its stated artifacts and
> tests exist, and every package below records precisely which paths its
> evidence does **not** reach.
>
> Foundation 0.3, the characterization baseline, turned out to be the keystone
> - M1.3, M1.4, M1.6 and M1.9 were all blocked on it - and smaller than the
> plan assumes, because five of its seven case families already had
> MATLAB-validated references. It is now eight frozen cases driving nine
> gate checks through `scripts/freeze_characterization.py`:
>
> | mode | gate | evidence |
> |---|---|---|
> | `--check` | F0.3 | 8 cases frozen, every one shown falsifiable |
> | `--observed` | M1.3 | 7 of 8 unchanged under observation |
> | `--hostile` | M1.2 | bit-identical under 3221 mutations, 228 exceptions |
> | `--resume` | M1.6 | exact on 5 of 8, each via its own driver |
> | `--bundle` | M1.10 | reloaded from disk and replayed on 5 of 8 |
> | `--classify` | M1.9 | 7 of 8; 8 synthetic controls + 1 real induced failure |
> | `--conserve` | M1.7 | 4 of 8, four model families; measured, not asserted |
> | `--invariants` | M1.8 | noise and impossibility distinguished |
> | `--perf` | §16 | observation within noise; store +4% to +47% |
>
> How to run these, what each covers, and why each uncovered case is
> uncovered: `docs/research_studio/evidence.md`.
>
> Every one of those checks was verified by making it fail first. That rule
> earned its place: three separate versions of this harness passed while
> testing nothing - recording no state fields, comparing empty decision lists,
> and attaching an observer that never fired.
>
> The same rule applies to the *explanations* recorded here, which is a
> distinct discipline and was learned later. Two entries in this ledger have
> been withdrawn after the cheap confirming check was finally run: M1.7's
> blocker attributed p65_oilwater's imbalance of 1.6 to two adapters
> disagreeing about where a fluid lives, and the real cause was a pore volume
> that is negative in 74 of 80 cells - the adapters now agree and the figure is
> unchanged. F0.4's coverage ranking surfaced three diagnostics "ports" that
> are one file copied three times. Writing the tests that package asks for
> would have certified the duplication rather than found it.
>
> A second instance of the same pattern turned up while closing the adjoint
> gradient gap: `optimization/utils/compute_gradient_adjoint_ad.py`
> re-implements the backward sweep already in `optimization/adjoint.py`, so
> `pymrst.modules.optimization.compute_gradient_adjoint` and the function at
> the MRST path are different objects. It is milder than the diagnostics trio -
> both share one result type, and the two copies agree numerically - and it is
> left as an owner decision for the same reason: deduplicating is a change to
> the replica layer, not an addition to it. The drift is pinned by a test that
> fails the day one copy is fixed without the other.

## 1. Purpose

This document converts `new_improvement_v3.md` into an implementation program
for PyMRST Research Studio and the Simulation Evidence Platform.

The work is intended to be robust, architecture-first and internally validated.
It is not organized around an early public release. Public benchmarks and user
interfaces should be generated from the same runtime, evidence and result
contracts after those contracts are mature.

The target end state is:

> Every model is interchangeable, every result carries evidence, every failure
> is diagnosable, every long run is resumable, and every research conclusion
> can be reproduced.

### 1.1 Implementation status

The first additive foundation is implemented under
`pymrst/research_studio/`:

- Foundation 0 execution-path inventory, machine-readable entrypoint ledger,
  call graph and initial architecture decisions;
- stable typed identifiers;
- immutable lifecycle events and synchronous handler-isolating event bus;
- flushed JSON Lines event output;
- strict, JSON-safe `SimulationEvidence` schema version 1;
- an observation-only `SimulationSession` adapter for the canonical schedule
  solver, with cancellation at safe outer boundaries and unchanged native
  state/report return objects;
- transactional HDF5 ResultStore v1 with accepted/rejected separation, lazy
  reads, versioned evidence, checkpoints, completion markers and interrupted
  pending-artifact recovery, using no Python object pickles;
- report-boundary Checkpoint v1 for bare timestep schedules, including accepted
  elapsed time, schedule/model/solver/code fingerprints, strict compatibility
  rejection, source lineage, branch construction and exact uninterrupted-versus-
  resumed nonlinear characterization;
- structured-schedule restart across control-period changes, with complete
  control-data fingerprinting, preserved control indices and restoration of
  the model-side prior control before native schedule continuation; and
- an optional dependency-free `NonLinearSolver` observation hook for live
  nonlinear residual, linear-solve and timestep-cut facts. Observer failures
  are isolated, existing observers are composed/restored, and native/enabled/
  failing-observer runs are required to return identical numerical states and
  iteration histories; and
- stable live report/ministep/attempt correlation, including retry indices for
  timestep cuts, plus transactional HDF5 event append, contiguous-sequence
  corruption checks and deterministic event replay; and
- generator-based canonical schedule execution that preserves the historical
  `simulate_schedule` return contract while allowing immediate accepted-state
  and checkpoint commits. Safe-boundary cancellation finalizes cleanly, and an
  abrupt process-style interruption leaves an unfinalized source run whose last
  committed checkpoint resumes exactly in a new run; and
- explicit water, immiscible oil-water and live black-oil surface-component
  conservation adapters, including free plus dissolved gas, signed well/
  boundary/source terms, pressure-dependent or state-provided pore volume,
  normalized imbalance and regional aggregation; compositional conservation on
  the equation assembler's EOS-component mass basis, including phase-split
  invariance and signed per-perforation component streams; poromechanics fluid
  conservation through deformation-dependent effective pore volume, with
  displacement and volumetric-strain coupling diagnostics kept separate from
  mass imbalance; committed-state analytic-aquifer accounting for both
  Fetkovich remaining-volume and Carter-Tracy cumulative-influx histories,
  while facility/well rates remain on their native signed component basis;
  plus immutable
  observe/strict/reference invariant profiles for finite and bounded fields;
  and
- evidence-only failure analysis for linear/non-finite failure, divergence,
  oscillation, stall, residual floors, control chattering and max-iteration
  exhaustion, including convergence-rate estimates and worst residual owner/
  iteration. Failed sessions attach the classification automatically; and
- ReproducibilityManifest v1 and portable failure bundles with input SHA-256
  lineage, command/environment/package metadata, code fingerprints, the newest
  accepted checkpoint, failing-step data, non-pickled NumPy arrays, member
  checksums, atomic archive publication and independent loading. Requested
  failed sessions emit `FailureBundleWritten` before finalization. Safe replay
  reconstructs data only, requires caller-owned executable model/solver/
  schedule objects, verifies input lineage and model/solver/schedule/code
  fingerprints, and only then invokes the native solver.

The characterization test runs the real `NonLinearSolver` both natively and
through the session and requires exact states, iteration counts and timestep
cuts. Research Studio is not imported by any MRST-shaped numerical owner.
Live correlation, durable replay and report-boundary interruption recovery are
implemented. Mid-Newton interruption remains unsupported until a recoverable
linear/nonlinear state contract exists. Water, oil-water, black-oil and
compositional and poromechanics conservation plus the invariant framework and
residual-history failure analysis are implemented. Analytic aquifer and
facility/well rate accounting plus failure-bundle replay compatibility are
also covered. Remaining Milestone 1 work is deeper deterministic recurrence
certification across representative nonlinear failures and the intentionally
deferred mid-Newton recovery contract.

Milestone 2 has started with a typed normalized comparison engine. It supports
strictly validated report-time interpolation, well/entity-name alignment,
explicit unit and sign conversion, combined absolute/relative/scale-normalized
tolerances, worst timestep/entity localization, active/global-cell alignment,
explicit phase/component aliases and deterministic regional sum/mean/weighted
aggregation. Ambiguous conventions, duplicate or missing identities, uncovered
time ranges and non-finite values fail explicitly. EvidenceGraph v1 records
the claim, mappings, independence, shared kernels, convergence, exceptions and
metrics for each validation edge; graphs persist inside SimulationEvidence and
round-trip through the HDF5 ResultStore.

M2.1's general field gap is now closed by `NamedFieldSeries`: every non-time
axis has a stable name and identities, candidate axes may arrive in another
order, and each axis is independently mapped before report-time interpolation.
The first real M2.2 graph is
`docs/research_studio/qualification/phase64_evidence_graph.json`, regenerated
from MATLAB R2021b/MRST 2024a and a fresh PyMRST Phase 64 run. Its eight edges
cover pressure, free and full displacement, volumetric strain, tensor strain,
effective stress, injection rate and BHP; every edge passes, and each declares
partial rather than independent evidence because PyMRST translates MRST's
formulation.

M2.3 now has its first four-node cross-formulation matrix. Fresh MATLAB/MRST
references cover fixed-stress (Phase 64) and fully coupled (Phase 63), each
connected to its corresponding PyMRST run by eight quantity edges. Eight more
edges compare the two PyMRST formulations, and a solver-work edge records
fixed-stress outer iterations `[5, 6]` against fully coupled Newton iterations
`[2, 2]` without pretending those counts have identical semantics. The two
PyMRST formulations differ by at most 3.391 Pa in pressure and exactly zero in
injection rate; the fully coupled PyMRST/MRST pressure difference is 0.002826
Pa. Cross-formulation edges declare `shared_kernel`, while MRST edges declare
`partial` independence.

The second M2.3 matrix qualifies natural variables against overall composition
on the two-step native SPE1 compositional case. It is another four-node graph:
each MATLAB/MRST formulation is connected to its matching PyMRST formulation,
and six named quantities connect the two PyMRST formulations. The 18 numerical
edges cover pressure, pseudo-component mole fractions, water/oil/gas
saturations, well BHP, oil rate and gas rate; a nineteenth edge reports solver
work. All pass. The PyMRST formulations differ by only `1.045e-05 Pa` in cell
pressure, `1.410e-14` in composition, `5.773e-15` in saturation and `0.03054
Pa` in well BHP, with oil rate at roundoff and gas-rate difference exactly
zero. Both use Newton iterations `[3, 4]`, compared with MATLAB's `[4, 7]` for
overall composition and `[9, 10]` for natural variables. The persisted artifact
is `docs/research_studio/qualification/compositional_natural_overall_evidence_graph.json`.

The third M2.3 matrix qualifies black-oil time integration on the first two
SPE1 report steps. It has six nodes (MATLAB/MRST and PyMRST for FIM, explicit
transport and AIM) and 36 edges: 21 same-method MRST comparisons, 14 PyMRST
FIM-versus-explicit/AIM comparisons and one solver-work edge. Explicit
well-name mapping handles MRST's injector/producer order against PyMRST's
producer/injector order. The first graph exposed two real porting defects: AIM
ran its CFL estimator before MRST's first-step-implicit check, and the existing
native `ComponentMobility`, `FaceComponentMobility`, and
`PermeabilityPotentialGradient` ports were not registered in the default
state-function graph, so explicit hydrocarbon transport could not freeze the
same owner as MRST. Fixing both moves explicit PyMRST/MRST pressure error from
`114746 Pa` to `0.04794 Pa` and saturation error from `0.01002` to
`7.572e-10`. FIM and AIM now both remain inside `37.78 Pa` in cell pressure;
all three methods share the same strict MRST tolerance policy. This is distinct
from the expected FIM-versus-explicit truncation difference, measured at
`219107 Pa` (`2.191 bar`) after both implementations match their MRST
counterparts. All methods cross the same report times without cuts. PyMRST uses
iterations `[3, 5]` for all three; MATLAB uses `[2, 4]`, `[3, 4]`, and `[2, 4]`.
Both AIM implementations now start fully implicit and then select 2/300 cells.
The artifact is
`docs/research_studio/qualification/blackoil_time_integration_evidence_graph.json`.

The fourth M2.3 matrix qualifies sparse direct, GMRES+ILU, and CPR+GMRES on
the same two-step fully implicit SPE1 system. `LinearSolveDiagnostic` now
propagates solver identity, convergence, Krylov iterations, true absolute and
relative residuals, wall time and degrees of freedom through every Newton
correction and through cut/adaptive timestep aggregation. This also exposed
and fixed a call-contract defect that made the base `GMRES_ILUSolverAD`
unusable through `NonLinearSolver` even though CPR's override worked. All three
solvers take nonlinear iterations `[3, 5]`; direct records 8 factorizations,
GMRES 98 Krylov iterations and CPR 48. Their maximum true relative linear
residuals are `3.108e-15`, `4.072e-11`, and `9.442e-09`. GMRES and CPR final
pressures agree with direct within `0.004252 Pa` and `0.003931 Pa`. The 22-edge
artifact is
`docs/research_studio/qualification/blackoil_linear_solver_evidence_graph.json`.

The fifth M2.3 matrix qualifies automatic derivatives against independently
perturbed values. Its six accepted edges separate the full oil-water residual,
reservoir mass rows, well closure rows, well controls, source/boundary fluxes,
and a four-control adjoint objective gradient. Mixed-unit variables are
perturbed in dimensionless coordinates; central differences at two step sizes
are Richardson-refined, and deterministic coordinate and dense directions are
recorded. The full residual's maximum relative directional error is
`3.501e-09`; reservoir, well-closure and control rows are all below
`3.501e-09`. The adjoint gradient differs from four full nonlinear forward
reruns by at most `1.382e-08` relative. This work found two genuine owner-code
defects in pressure-boundary differentiation: fixed outside pressure retained
the cell-pressure Jacobian, cancelling `d(dp)/dp_cell`, and an inactive
reservoir-flux branch attempted `numpy.where` over AD operands. Both are fixed
and pinned by a focused regression. Finite-difference edges declare partial
independence because they differentiate values without AD operators but still
share the residual-value implementation. The artifact is
`docs/research_studio/qualification/ad_finite_difference_evidence_graph.json`.

The sixth and final M2.3 matrix qualifies physical Peng--Robinson flash against
Victory's beta-2.2 seven-fold conditional flash ensemble on the existing
digest-matched 10,000-state PyMRST/MATLAB-MRST run. It contains 12 nodes and 29
edges: nine PyMRST/MRST physical-output comparisons, seven ensemble output
gates, seven individual-fold characterizations, three physical-safety edges,
phase-decision parity, inference speed and ensemble-uncertainty signal. PyMRST
and MRST pass all physical parity gates and make identical phase decisions.
Victory inference is `20.16x` faster than the batched physical PyMRST flash,
but that number excludes stability analysis because the network receives
reference-derived phase flags. The ensemble fails production gates for six of
seven outputs and fails physical safety: material-balance RMSE is `6.386e-03`
and log-fugacity RMSE is `3.522`. Consequently the experiment is complete but
the surrogate is explicitly **not accepted** as a production flash or a
stability classifier. The graph also carries a pressure/phase/uncertainty
error atlas and content hashes for all source artifacts. The artifact is
`docs/research_studio/qualification/victory_surrogate_evidence_graph.json`.

## 2. Governing implementation principles

1. **Preserve numerical behavior before improving architecture.** Add
   characterization evidence before modifying an execution path.
2. **One source of evidence.** Console output, reports, comparisons, notebooks
   and browser views consume structured events and evidence; they do not
   reconstruct it independently.
3. **Explicit model semantics.** Conservation, units, phases, components and
   controls are supplied through model adapters, never guessed globally.
4. **Compatibility through delegation.** Existing public simulation functions
   remain available and delegate to the new runtime where their semantics fit.
5. **Specialized methods remain specialized.** Adjoint, optimization and packed
   workflows implement common protocols without being forced through an
   unsuitable forward-simulation kernel.
6. **Persistence is transactional.** An interruption must not corrupt accepted
   states or evidence.
7. **Schemas are versioned from their first persistent release.** Stored
   results, reports, checkpoints and experiments require migration tests.
8. **Validation is continuous and initially private.** Internal benchmark
   evidence protects development without creating premature release pressure.
9. **No silent numerical policy changes.** PyMRST may diagnose and recommend;
   it does not silently loosen tolerances, change controls or replace physics.
10. **Every capability has a completion gate.** Filename presence or a partial
    demonstration is not completion.

### 2.1 MRST replica preservation covenant

Research Studio is an additive PyMRST extension. It must not redefine what it
means for PyMRST to be a Python replica of MRST.

The following rules are non-negotiable:

1. MRST-equivalent module directories, owner files, class/function names and
   public entry points remain available in their MRST-shaped locations.
2. MRST equations, discretizations, state transitions, control semantics,
   default tolerances, timestep policies and option defaults are not changed by
   Research Studio.
3. The default call to an existing PyMRST/MRST-compatible API retains its
   existing return contract and numerical behavior.
4. Evidence collection is observational. Event handlers, stores, renderers and
   diagnostics cannot modify solver/model state.
5. Checkpointing and result streaming persist accepted behavior; they do not
   create a different numerical formulation.
6. Alternative solvers, formulations, validation profiles, surrogates and
   automatic recommendations are explicit opt-in choices.
7. A learned component never silently replaces an MRST physics component.
8. Research Studio files do not count as substitutes for missing MRST owners in
   module parity audits.
9. Every integration change must pass the affected MRST parity fixtures and a
   before/after characterization comparison with the extension disabled and
   enabled in observation-only mode.
10. If faithful MRST behavior and an improved/physically corrected behavior
    differ, both are named explicitly. The MRST-compatible path remains
    available, and the improved path is an extension profile rather than a
    silent correction.

The architecture therefore has two compatible layers:

```text
PyMRST MRST Replica Layer
  MRST-shaped modules, equations, defaults, examples and compatibility APIs
                           │
                    explicit adapters/hooks
                           │
PyMRST Research Studio Layer
  evidence, persistence, validation, experiments, surrogates and UI
```

Research Studio may observe, orchestrate and compare the replica layer. It may
offer new opt-in capabilities around it. It must not make the replica layer
dependent on the browser, experiment engine, external simulators, ML frameworks
or optional result formats.

### 2.0 The covenant applies to every capability in this plan

Section 2's first principle is that these capabilities are **additive**. It is
restated here because the packages added by later review - interoperability,
diagnostics, grid inspection, dimensional safety, packaging, performance work -
are the ones most likely to erode it, each for a different reason:

- **Default off, or default inert.** A user who does not opt in gets
  bit-identical results and the same exceptions as before. Where a capability
  has a cost or a behaviour, it has a switch.
- **No replica-layer module imports the extension.** Checked, not assumed:
  `grep -rl research_studio pymrst/modules pymrst/gridprocessing pymrst/solvers`
  returns nothing today, and that is a standing condition rather than a
  historical fact.
- **Inspection is read-only.** M2.11 may interrogate a grid; it may not
  normalize, repair or annotate one in place.
- **Additive is not the same as harmless.** A capability that *refuses* input
  the port currently accepts has changed the port, even though it added no
  feature. M2.12 is the live example and is opt-in for exactly this reason.
- **Performance work is the sharpest case**, because it touches the numerics
  directly: any change to accumulation order or sparse representation must be
  quantified against the F0.3 baseline before adoption, not after.

The parity is what makes everything above it worth having. A capability that
compromises it destroys the thing it was meant to enhance.

### 2.1 Considered and not adopted

**A backend abstraction over NumPy, JAX, PyTorch and CuPy.** Proposed as an
architectural layer so the same model could execute on any array library, for
differentiability and GPU support.

**Deferred with a stated trigger, not rejected.** Revisit only if Foundation-0
profiling shows the Python assembly loop dominating runtime *after* the Tier 1
fixes in M4.5 - canonical CSR construction, sparsity-pattern reuse and
vectorized duplicate reduction. At that point it becomes an evidence-driven
decision rather than an architectural preference, and M3.9's portable case
definition would make a second execution path far more tractable than it is
today. Until that profile exists the question is unanswerable, and the
capabilities usually cited for it have cheaper routes (below).

Not adopted *now*, because in this codebase it is a rewrite rather than a
layer, and the surface is measurable: `ADI` is a value vector plus an explicit
list of sparse Jacobian blocks, and `.jac` is reached into directly at **325
sites across 85 files**, with 68 files constructing variables through
`init_variables_adi`. JAX and Torch expose a trace, not a materialized
Jacobian list, so those are not porting friction - they are places assuming a
different AD representation.
PyMRST carries its own automatic-differentiation system under
`pymrst/utils/adi/`, and every model, state function and residual assembly is
written against it. Switching array backend therefore means replacing the AD
layer, which means rewriting every model that depends on it - and the
proposal itself concedes that backends cannot be treated as identical, so the
end state is two partially-complete simulators to maintain and benchmark
against each other.

The tractable version of the same goal is narrow rather than universal: keep
the physics on the existing AD system, and expose *specific* components -
surrogate closures, learned constitutive relations, hot kernels - to torch or
JAX at a declared boundary. That is what M4.2's surrogate contract and M4.3's
guarded runtime already describe, and `pymrst/ml/` is positioned for it.

Three of the four capabilities usually cited for a full abstraction already
have cheap routes:

- **training a network end-to-end through the physics** - the strongest
  argument - needs only the existing adjoint wrapped as a
  `torch.autograd.Function`: forward on PyMRST's solver, backward on
  `compute_gradient_adjoint`, composed with any network by autograd. One
  adapter, not 85 files;
- **GPU** belongs behind `LinearSolverAD`, which is already an interface, and
  is solver-scope - M4.6;
- **ensemble vectorization** is M3.3's distribution problem, and reservoir
  ensembles are usually memory-bound rather than launch-bound.

Only **JIT-compiling the residual** has no cheap substitute, and whether it
matters is exactly the Tier 0 profiling question above.

**Repositioning away from MRST parity.** Proposed on the grounds that matching
a mature toolbox feature-for-feature is an unwinnable race.

Not adopted, because the parity *is* the credibility mechanism. A new
reservoir simulator has no reason to be trusted; this one reproduces MRST to
1.7e-09 on pressure on a Norne case, carries MATLAB-validated references for
its case families, and records in a parity ledger which MRST files are ported
and which are not. That is not a legacy obligation to escape - it is the only
asset that makes the additive layers above worth anything. The architecture
already expresses the right relationship: parity is the trust layer, and
everything in this plan sits on top of it without being allowed to modify it.

## 3. Proposed package ownership

Research Studio capabilities are PyMRST extensions and should not be forced
into MRST module-layout directories. Keep them visibly separate under one
extension namespace:

```text
pymrst/
  research_studio/
    runtime/
      session.py
      protocols.py
      identifiers.py
      controls.py
      cancellation.py
      compatibility.py

    evidence/
      report.py
      events.py
      manifest.py
      convergence.py
      conservation.py
      invariants.py
      failure_analysis.py
      failure_bundle.py
      deck_lineage.py
      renderers/

    storage/
      result_store.py
      checkpoint.py
      hdf5_store.py
      zarr_store.py
      migrations/

    validation/
      graph.py
      comparison.py
      mappings.py
      analytic/
      matlab/
      opm/
      properties/
      metamorphic/
      fault_injection/
      benchmarks/

    experiments/
      definition.py
      registry.py
      variants.py
      execution.py
      ensemble.py
      branching.py
      comparison.py
      publishing.py

    components/
      registry.py
      capabilities.py
      schemas.py
      surrogate.py
      fallback.py

    studio/
      notebook/
      browser/
      reports/
```

Existing numerical owners remain where they are. Integration occurs through
small optional protocols and adapters rather than moving MRST ports or making
their equations depend on Research Studio. Stable convenience imports may be
added later, but implementation ownership remains under `research_studio/`.

## 4. Core contracts

### 4.1 Stable identifiers

Every event and artifact must carry stable identifiers:

```python
RunId
CaseId
ExperimentId
MemberId
ReportStepId
MinistepId
AttemptId
CheckpointId
```

Identifiers must be serializable strings, unique inside their scope and stable
across restart.

### 4.2 Simulation session

```python
session = SimulationSession(
    model=model,
    initial_state=state0,
    schedule=schedule,
    nonlinear_solver=solver,
    result_store=store,
    evidence_profile="strict",
)

result = session.run()
```

`SimulationSession` owns orchestration, not numerical equations. It coordinates:

- report/ministep lifecycle;
- accepted and rejected transitions;
- control application;
- event dispatch;
- cancellation;
- evidence accumulation;
- checkpoint commits;
- streaming output;
- restart and replay metadata.

### 4.3 Simulation result

```python
SimulationResult
├── final_state
├── states                 # lazy sequence where possible
├── report
├── evidence
├── store
└── status
```

### 4.4 Evidence schema

```python
SimulationEvidence
├── schema_version
├── run_identity
├── manifest
├── input_lineage
├── timestep_reports
├── convergence
├── conservation
├── invariants
├── deck_coverage
├── failures
├── warnings
├── timings
├── comparisons
└── checkpoints
```

Use typed dataclasses or similarly explicit models. Avoid placing model or
solver objects inside persistent report structures.

### 4.5 Event protocol

At minimum:

```text
RunStarted
RunCompleted
RunFailed
ReportStepStarted
MinistepStarted
NonlinearIterationCompleted
LinearSolveCompleted
ControlSwitched
StepAccepted
StepRejected
TimestepCut
InvariantViolated
CheckpointCommitted
FailureBundleWritten
```

Events are immutable records. Handlers may render output or persist logs but
must not change solver state.

### 4.6 Result store

```python
class ResultStore(Protocol):
    def begin_run(...): ...
    def commit_state(...): ...
    def commit_evidence(...): ...
    def commit_checkpoint(...): ...
    def load_state(...): ...
    def load_checkpoint(...): ...
    def finalize_run(...): ...
```

The HDF5 implementation should be first because optional `h5py` support already
exists. Zarr/xarray support follows behind the stable protocol.

### 4.7 Model evidence adapters

Models opt into evidence features explicitly:

```python
class ConservationAdapter(Protocol):
    component_names: tuple[str, ...]
    def inventory(self, state): ...
    def well_terms(self, state, dt): ...
    def boundary_terms(self, state, dt): ...
    def source_terms(self, state, dt): ...

class InvariantProvider(Protocol):
    def evaluate_invariants(self, state, context): ...
```

Adapters must declare surface/reservoir bases, units, sign conventions, active
cells and phase/component ordering.

## 5. Compatibility and migration strategy

### 5.1 Existing entry points

Before modifying code, inventory at least:

- `simulate_schedule_ad`;
- packed-problem foreground/background/standalone paths;
- module-specific `run_schedule` functions;
- adjoint `simulate_forward` functions;
- optimization simulation adapters;
- compositional example drivers;
- field/example orchestration wrappers.

Classify each as:

1. direct runtime delegate;
2. specialized session adapter;
3. high-level example wrapper;
4. deprecated duplicate;
5. intentionally independent numerical workflow.

Do not change return values during the first integration. Compatibility wrappers
may expose the new result through an optional flag or retained `last_result`.

### 5.2 Schema versioning

Persistent schemas use semantic integer versions:

```text
SimulationEvidence v1
Checkpoint v1
ResultStore v1
ExperimentDefinition v1
FailureBundle v1
```

Every migration requires:

- source and target fixtures;
- forward migration test;
- rejection test for unsupported versions;
- corruption/truncation test;
- documentation of information loss.

### 5.3 Breaking-change policy

During private development, breaking changes to Research Studio's new internal
schemas are allowed only when:

- an architecture decision record explains the change;
- all stored internal fixtures are migrated or explicitly retired;
- characterization/parity gates pass;
- affected compatibility wrappers are updated.

This policy does not authorize silent breaking changes to MRST-compatible
module APIs or numerical behavior. Any unavoidable replica-layer compatibility
change requires an explicit migration path, parity evidence and user approval
before implementation.

## 6. Foundation 0: architecture and characterization

### Objective

Create the design and regression baseline required for cross-cutting runtime
work.

### Work package F0.1: execution-path inventory

Tasks:

1. Use AST/import analysis and targeted runtime tracing to inventory simulation
   entry points.
2. Produce a call graph showing schedule, timestep, nonlinear, linear, update,
   control and persistence ownership.
3. Mark duplicated versus intentionally specialized paths.
4. Record current return contracts and state/report structures.
5. Identify all direct `print()` progress/reporting paths.

Artifacts:

```text
docs/research_studio/runtime_inventory.md
docs/research_studio/runtime_call_graph.mmd
docs/research_studio/runtime_entrypoints.csv
```

### Work package F0.2: architecture decisions

Create ADRs for:

- session orchestration;
- event delivery;
- evidence ownership;
- result/checkpoint storage;
- schema/versioning policy;
- state serialization;
- compatibility wrappers;
- component and capability registration;
- experiment identity and hashing.

### Work package F0.3: characterization baseline

Select representative cases:

- single-phase water;
- immiscible oil-water;
- black-oil SPE1;
- compositional natural and overall variables;
- fixed-stress and fully coupled poromechanics;
- one packed problem;
- one adjoint/optimization workflow.

Record:

- final and selected intermediate states;
- report/ministep structure;
- solver decisions and cuts;
- controls and well histories;
- conservation where already available;
- runtime and memory;
- hashes and exact commands.

#### F0.3 addendum: what the baseline does not yet record

The frozen baselines capture accepted timesteps, cuts, Newton iterations,
residual histories, states, well rates and BHP, material-balance imbalance,
objective values, adjoint gradients, runtime and peak memory. Three things are
missing, and the first two matter specifically for any change to sparse
assembly or accumulation order:

- **linear iterations per nonlinear iteration.** Newton counts alone cannot
  distinguish a change that made the linear system harder from one that did
  not touch it. Larger than it looks: `GMRES_ILUSolverAD` counts them already,
  but `LinearSolverAD.solve` returns only the increment, `_solve_linear`
  discards the rest and `StepReport` has no field for it - so this is a return
  contract change across the linear solvers, the nonlinear solver and the
  report, in core replica code.
- ~~**Jacobian dimensions, nonzero count and a structural hash.**~~ **Done.**
  Recorded per case and compared by `--check`: rows, columns, nonzeros and a
  sha256 over `indices` and `indptr` only, never `data`, so it is orthogonal
  to the numerical comparison rather than a duplicate of it. Available on 3 of
  8 cases; the rest record *why not*, since a fixed-stress model solving flow
  and mechanics in separate systems has no single Jacobian to sign. Verified
  against a matrix differing by one explicit stored zero - numerically
  identical, structurally distinguished - while a within-row index permutation
  hashes the same, because it is the same matrix.
- **an engineering-invariant tier.** Comparison today is structural (shapes,
  decisions) and numerical (relative, `atol=0`). A third tier of quantities an
  engineer would actually defend - breakthrough time, cumulative production
  and injection, leakage fraction, maximum pressure, constraint satisfaction -
  fails differently from a per-cell tolerance and catches changes that are
  numerically small and practically wrong.

The baseline is a behavioural safety net, not a claim that current results are
ideal. It deliberately preserves present behaviour, quirks included, until a
change is reviewed on purpose.

### Work package F0.4: risk-ranked coverage

1. Inventory substantive modules without focused tests.
2. Score numerical risk, dependency centrality, size and use frequency.
3. Add tests to the highest-risk group.
4. Record defects and convert them into reusable properties or metamorphic
   tests.

### Foundation 0 gate

- Every supported execution path has an owner and classification.
- Baseline cases regenerate deterministically within documented tolerance.
- Core runtime behavior has characterization protection.
- ADRs resolve state, report, event and persistence ownership.
- No Milestone 1 work begins with an unresolved persistent schema decision.

### Work package F0.5: packaging and environment reporting

Make the first five minutes work, and make the environment legible.

`pip install pymrst` already works - the distribution installs and imports.
What does not exist is any way to ask what the installation can actually *do*:
there is no `[project.scripts]` entry, so there is no `pymrst` command at all.

Required:

- a console entry point, and `pymrst doctor` reporting capability rather than
  version soup: core install, sparse solver, optional AMG and PETSc, plotting
  backend, torch and CUDA, writable cache and output directories;
- capability status separated from *data* status. "Installed correctly" and
  "benchmark data downloaded" are different states and must not share a line;
- `pymrst datasets list` and `pymrst datasets fetch <name>`, over the existing
  `utils.dataset_manager`, which already downloads and resolves paths;
- **a licence and redistribution field on `DatasetInfo`, presented before a
  download proceeds.** The dataclass currently carries `website`, `source`,
  `instructions` and `note`, and nothing about terms. No benchmark deck should
  be bundled or fetched until its redistribution rights are recorded - this is
  the one item here with release risk rather than convenience value.

Placed in Foundation 0 because it is a precondition for anyone outside this
repository existing: contributors, testers and reviewers all need an
installation they can verify before any of the milestones above matter.

## 7. Milestone 1: unified runtime and evidence engine

### Objective

Make all supported simulation paths observable, persistent and diagnosable
through common protocols.

### Work package M1.1: identifiers and schemas

Implement:

- stable identifiers;
- evidence dataclasses;
- timestep, convergence and failure reports;
- JSON serialization;
- strict schema validation at persistence boundaries;
- unit tests and golden schema fixtures.

### Work package M1.2: event system

Implement:

- synchronous in-process event bus first;
- handler isolation and error policy;
- console renderer preserving current useful output;
- JSON Lines handler;
- progress handler;
- tests proving handlers cannot change numerical decisions.

### Work package M1.3: canonical nonlinear integration

Instrument `NonLinearSolver` without changing its mathematics:

- named residual histories;
- requested/achieved tolerance;
- update/relaxation/line-search information;
- linear-solver result;
- timestep attempts and cuts;
- state/control transition identifiers.

Run characterization and MATLAB/parity gates after every integration step.

### Work package M1.4: SimulationSession

Implement orchestration for the canonical schedule path, then integrate each
entry-point class from Foundation 0.

For every integration:

1. add compatibility tests;
2. run baseline cases before/after;
3. compare solver decisions and numerical states;
4. verify event completeness;
5. record any intentional difference in an ADR.

### Work package M1.5: transactional HDF5 result store

Requirements:

- append accepted states;
- separate rejected attempts;
- atomic completion markers;
- lazy state reads;
- evidence storage;
- interruption tests;
- corruption detection;
- schema version storage;
- no dependence on Python object pickling for durable data.

### Work package M1.6: checkpoint and restart

Persist:

- state and facility/well solution;
- model-control state;
- schedule/report position;
- accepted elapsed time;
- solver configuration;
- evidence and identifiers;
- input/code fingerprint.

Test:

- interruption after each lifecycle stage;
- resume equivalence to uninterrupted simulation;
- failed compatibility checks;
- schedule continuation;
- branch from a checkpoint;
- rejected trial never replacing accepted state.

### Work package M1.7: conservation adapters

Implementation order:

1. water;
2. immiscible oil-water;
3. black-oil;
4. compositional;
5. poromechanics;
6. aquifers/facilities;
7. other model families.

For each adapter:

- document component basis and signs;
- compare with direct accumulation equations;
- test closed and source-driven systems;
- test wells/boundaries;
- test compressible rock where applicable;
- test regional aggregation;
- validate against an independent hand calculation or reference.

### Work package M1.8: invariant framework

Implement validation profiles and model-provided checks. Tests must distinguish
floating-point noise from physical impossibility and verify observe/strict/
reference behavior.

### Work package M1.9: failure analysis

Implement classifiers using residual and event histories:

- convergence-rate analysis;
- stall/floor detection;
- oscillation detection;
- control-chattering detection;
- divergence and non-finite detection;
- linear-failure classification;
- worst owner/location reporting.

Validate using real historical failures and synthetic fault injection.

### Work package M1.11: diagnostic narrative

Turn M1.9's classification into an explanation an engineer can act on.

M1.9 already produces the structured half: nine failure kinds, the worst
residual and where it occurred, convergence rate, control-switch counts. What
it does not produce is a statement of what went wrong. "Newton failed at step
42" and "stall detected, worst residual CNV in cell 18,420" are both true and
neither tells anyone what to change.

The narrative must be **derived from the recorded diagnosis, never generated
from a log**. That is the whole design constraint: an explanation invented by
a language model reading solver output is a plausible story about a
simulation, and indistinguishable from a correct one. Required:

- a deterministic mapping from classification plus state evidence to a stated
  cause - saturation change per step, the pressure and PVT range the step
  crossed, the control that switched, the cell that carried the worst
  residual;
- suggested actions drawn from the parameters that actually govern the
  diagnosed cause, each naming the setting and its current value;
- every number in the narrative traceable to a field of the evidence record,
  so the explanation can be checked against the run rather than trusted;
- a machine-readable form alongside the prose, since the narrative is also
  what an agent or a report generator consumes.

An optional language-model layer may rephrase a completed narrative for
readability. It may not originate one, and the ungenerated form must remain
available.

### Work package M1.10: reproducibility and failure bundles

Implement manifests and minimal bundles. A bundle must be independently
loadable and reproduce the failing step without earlier schedule execution.

### Milestone 1 gate

- Supported entry points emit compatible evidence.
- Baseline numerical results remain within certified tolerance.
- Accepted states survive later failure and process interruption.
- Resume matches uninterrupted simulation.
- Primary model-family conservation adapters pass.
- Known phase-114 and Norne residual behaviors are classified correctly.
- Every statement in a diagnostic narrative resolves to a recorded evidence
  field, and no narrative is produced for a run with no diagnosis.
- Persistent schemas pass migration, corruption and compatibility tests.
- Full ad-core, affected module and parity suites pass.

#### Where the gate stands

Assessed against evidence rather than intent. Milestone 2 is ordered behind
this gate, so what blocks it is worth stating precisely.

| criterion | status |
|---|---|
| entry points emit compatible evidence | **met for what is in scope** - 6 of 11 carry the hook, a 7th is covered by delegation, and the remaining 4 are excluded by design |
| baselines within certified tolerance | **met** - `--check` exact on 8 of 8 |
| accepted states survive failure and interruption | **met** - abrupt-interruption and cancellation paths resume from the last committed checkpoint |
| resume matches uninterrupted simulation | **met for what a session can drive** - exact on 5 of 8; the other 3 build their schedule internally |
| conservation adapters pass | **not met** - 4 families measured, but "pass" needs a tolerance and section 16 sets budgets only after this measurement. The figures now exist; the budget does not |
| phase-114 and Norne classified | **met** - both covered, plus 8 synthetic controls and one real induced failure |
| schemas pass migration, corruption, compatibility | **partly met** - corruption and compatibility are tested; migration is untestable while only v1 exists, and a fixture test asserts no schema has moved past v1 so a second version cannot land without it |
| full suites pass | **met** - 0 failures across the project suite |

Two things stand between here and Milestone 2, and only one is work:

1. **A conservation budget.** This is a decision, not an implementation, and
   it now turns on one number rather than an unexplained spread. Three of the
   four families conserve at or near machine precision (9.89e-17 compositional,
   1.74e-16 water fully coupled, 9.72e-08 Norne oil-water); fixed-stress water
   sits at 4.17e-06 for reasons not established. It is invariant under every
   tolerance the model has, so it is structural, but two proposed causes have
   been refuted by experiment. A budget drawn above it today would be a number
   chosen to pass rather than a bound anyone can defend.
2. **A second schema version**, which cannot be exercised until something
   actually needs one.

M2.1 is already `implementing`, ahead of this gate, which the plan orders
otherwise.

## 8. Milestone 2: validation laboratory

### Objective

Provide multiple explicit evidence classes for numerical correctness.

### Work package M2.1: normalized comparison engine

Support:

- report-time interpolation/alignment;
- active/global cell mapping;
- well-name alignment;
- phase/component mapping;
- unit and sign conversions;
- scalar, field and history metrics;
- absolute/relative/scale-normalized tolerances;
- worst location and timestep reporting.

### Work package M2.2: evidence graph

Represent nodes and edges for:

- PyMRST formulations;
- MRST;
- OPM Flow;
- analytic/manufactured solutions;
- finite-difference derivatives;
- invariants and conservation.

Edges record independence, shared kernels, mappings, convergence quality and
known exceptions.

### Work package M2.3: cross-formulation cases

Implement at minimum:

- natural versus overall compositional;
- fully implicit versus sequential/AIM where supported;
- fully coupled versus fixed-stress mechanics;
- direct versus iterative linear solve;
- AD versus finite-difference Jacobian/gradients;
- physics versus surrogate components.

Implemented qualification matrices:

- fixed-stress versus fully coupled mechanics, with both MRST counterparts;
- natural variables versus overall composition, with both MRST counterparts.
- black-oil FIM versus explicit transport versus AIM, with all three MRST
  counterparts.
- sparse direct versus GMRES+ILU versus CPR+GMRES, with MRST direct as the
  external anchor.

M2.3's six planned formulation families are now implemented. Acceptance is
per edge: completing the physics-versus-surrogate matrix does not turn its
failed surrogate safety gates into positive evidence.

### Work package M2.4: analytic/manufactured registry

The canonical Research Studio hierarchy is recorded in
`docs/research_studio/package_layout.md`. During M2.4 the initially flat
`validation/analytic.py` and `validation/analytic_benchmarks.py` files were
relocated into `validation/analytic/contracts.py` and
`validation/benchmarks/foundation.py`. This keeps registry contracts cheap to
import and puts executable numerical adapters behind an explicit benchmark
package. A structural regression test prevents the flat layout from returning.

Initial registry:

- hydrostatic equilibrium;
- 1D diffusion;
- radial pressure transient;
- Buckley-Leverett displacement under declared assumptions;
- Terzaghi consolidation;
- Mandel poromechanics;
- elasticity patch tests;
- flash equilibrium identities;
- small accumulation/flux balances.

Every case declares assumptions, domain, variables, units and expected
convergence order.

The M2.4 foundation is implemented as an explicit `AnalyticRegistry` with
immutable case specifications, refinement levels, normed errors, invariants,
acceptance policies and observed-order calculation. Importing Research Studio
does not execute benchmarks. The registry artifact now contains all nine
planned cases and thirty-nine independently supported edges:

- the incompressible TPFA gravity column reproduces
  `p(z) = p_top + rho*g*z` over four grids with maximum pressure error
  `2.049e-07 Pa` and maximum absolute face flux `2.003e-18 m^3/s`;
- constant-coefficient 1D pressure diffusion uses cell-centred TPFA and
  backward Euler with `dt proportional to dx^2`. Across 10, 20, 40 and 80
  cells its observed orders are `2.006`, `2.009` and `2.014` in L1, L2 and
  Linf. Finest-grid L2 error is `9.820e-05`; accumulated storage agrees with
  integrated net boundary flux within `3.19e-12`, and the discrete maximum
  principle holds;
- the radial pressure transient uses the exact axisymmetric eigenmode
  `J0(alpha*r)*exp(-D*alpha^2*t)` on annular control volumes. Its observed L1
  and L2 orders are `1.956` and `1.940`; finest L2 error is `3.541e-06` and
  storage/boundary-flux imbalance is `4.67e-13`;
- equal-mobility Buckley--Leverett displacement runs PyMRST's production TPFA
  pressure and explicit transport solvers over 50--400 cells. The
  shock-limited L1 order is `0.793`, finest L1 error is `4.192e-03`, and the
  water front is within `1.618e-03` of the Welge position. Injected and stored
  water differ by at most `6.22e-15 m^3`, with bounded saturations.
- Terzaghi consolidation reuses the existing MRST-layout poromechanics
  workflow. Pressure and settlement converge at orders `1.023` and `1.001`;
  finest relative maximum errors are `9.380e-04` and `2.959e-04`. Excess
  pressure remains within drained/undrained limits and settlement is monotone;
- Mandel reuses the fully coupled moving-plate owner. Maximum pressure error
  converges at order `1.227`, reaching `8.741e-03` on the finest grid. The
  centre pressure peaks at `1.0965` times its initial value, certifying the
  Mandel--Cryer effect that a one-way or lagged coupling misses. Pressure is
  uniform through thickness within `1.08e-08`, and prescribed plate force is
  maintained within `4.07e-09` relative;
- VEM elasticity patch tests exercise six affine fields on regular and
  distorted 2D grids and twelve fields in 3D. Maximum relative displacement
  error is `4.441e-16`, normalized rigid-mode energy is `4.541e-17`, and
  stiffness asymmetry is `8.650e-17`;
- the Peng--Robinson ensemble contains 168 explicitly two-phase states across
  three batch sizes. Overall material balance closes within `1.198e-13`, both
  phase compositions close within `2.220e-16`, and maximum liquid/vapour log
  fugacity mismatch is `4.253e-12`;
- the manufactured accumulation/flux family exercises 3, 9 and 33-cell TPFA
  chains through the production ADI operators. Local residual and analytic
  Jacobian-entry error are exactly zero; closed-domain global balance is
  `7.445e-24`, and every AD sparsity pattern matches the independently
  assembled tridiagonal matrix.

The analytic solutions do not call MRST or reuse PyMRST residual kernels, so
these graph edges are classified as independent evidence. The persisted graph
is `docs/research_studio/qualification/analytic_foundation_evidence_graph.json`.
The planned M2.4 family is complete; the package remains `validating` until
the evidence and tolerances receive acceptance review.

### Work package M2.5: MRST adapter

Standardize fixture generation and direct execution:

- version and module capture;
- input/output mapping;
- solver settings and achieved convergence;
- dataset hash;
- quantity-specific comparison configuration;
- immutable reference and regeneration command.

M2.5 is implemented under `validation/matlab/` as a shell-free adapter with
strict versioned schemas. `MatlabFixtureSpec` declares the case, generator,
MRST root and module set, input artifacts, solver settings, result mappings
and quantity-specific tolerances. `MatlabFixtureManifest` records SHA-256
digests for the generator, reference and every simulator input alongside the
MATLAB release, inferred MRST version, exact batch command and generation
timestamp. Validation refuses missing manifests, changed reference payloads,
changed generator sources, changed decks, altered mappings/tolerances, module
drift and solver-setting drift.

The first real consumer is the two-step SPE1 FIM/explicit/AIM matrix. It was
regenerated through the adapter with MATLAB R2021b and MRST 2024a, including
the `BENCH_SPE1.DATA` digest and six loaded modules. Its manifest is
`examples/validation/ref_time_integration_spe1.mat.manifest.json`; the
qualification artifact embeds that manifest, explicit time/cell/well/phase
mappings, achieved iteration/ministep/AIM-partition histories and all declared
tolerances. The regenerated matrix passes all 36 evidence edges. Unit tests
mutate the reference, generator and deck independently and verify that each
change is rejected as stale.

The second consumer is a direct poromechanics equation fixture. It uses only
`ad-core` and `ad-mechanics`, hashes MRST's `equationsPoroMechanics.m` owner,
and records an assembly-only AD solver policy. The scaled displacement
residual agrees within `4.441e-16`; the displacement and pressure Jacobian
blocks agree entry-for-entry. Its three-edge graph and signed manifest are
persisted as
`docs/research_studio/qualification/poromechanics_equations_evidence_graph.json`
and `scripts/matlab/ad_mechanics/ref_equations_poro_mechanics.mat.manifest.json`.

M2.5 is accepted: two physically and structurally distinct consumers now use
the same execution, fixture, provenance, mapping and stale-input contracts;
both were regenerated with installed MATLAB and pass their declared evidence.

### Work package M2.6: OPM Flow adapter and ECLIPSE result ingestion

Implement ECLIPSE binary readers for EGRID, INIT, UNRST and UNSMRY. Record
supported keyword/model overlap and known conventions explicitly.

**The original wording said "promote and extend ECLIPSE readers"; a first
revision of this section replaced it on the grounds that no such readers
exist. That was wrong, and checking took a minute.**
`pymrst/modules/ad_core/utils/read_summary_local.py` already reads
SMSPEC/UNSMRY pairs, and carries the part that is actually fiddly: the Fortran
unformatted record layer, the big-endian INTE/REAL/DOUB/LOGI map, CHAR width
handling and chunked-block reassembly. Its `_record` and `_fields` helpers are
format-agnostic and are what the remaining readers should be built on.

So the true state is: ECLIPSE *decks* load - RUNSPEC, GRID, PROPS, REGIONS,
PVT and relative-permeability tables via `read_eclipse_deck`, with Norne, Egg,
SPE1, SPE9 and SPE10 all going through it - and *summary output* loads.
**EGRID, INIT and UNRST do not**, so a case another simulator ran can be set
up here and its well curves compared, but its grid, static properties and
solution arrays cannot be loaded. That is the gap, and it is narrower than a
from-scratch reader.

Required:

- EGRID geometry, including corner-point and faulted grids, mapped onto the
  same grid contract `read_eclipse_deck` already produces;
- INIT static properties, with unit conversion recorded rather than assumed;
- UNRST restart/solution arrays, indexed by report step;
- UNSMRY summary vectors, aligned to well and group names;
- an explicit statement of which keywords are read, which are ignored, and
  which cause a refusal - a reader that silently skips a keyword it does not
  understand produces a case that is quietly not the one on disk.

Reading is deliberately permissive about keywords it does not support, and
must say so per file. That asymmetry with M3.8, which cannot be permissive,
is the reason the two are separate packages.

**The per-file statement is built**, in `eclipse/coverage.py`, and it is the
half of permissiveness that makes it safe. `keyword_support()` says what the
reader understands in general; `keyword_coverage(path)` answers the question a
user actually has - *for this file, what gets used and what gets dropped?* -
because a case built from a file full of unrecognised keywords loads cleanly
and is quietly not the case on disk.

Keywords are classified `used`, `read_but_unused`, `not_implemented` and
`unrecognised`. The last two are kept apart on purpose: NNC and LGR are gaps
this package knows it has and can name, while an unrecognised keyword is one it
knows nothing about, and the second should worry a reader more. `describe_gaps`
reports what is lost rather than a bare keyword. So **NNC and LGR remain
unread, but are no longer silently dropped** - a smaller claim than reading
them, and stated as such.

The threshold lesson recurred a third time and was applied. An earlier version
counted `FILEHEAD` and `INTEHEAD` as dropped, which marked every valid file
incomplete - including the ones this package writes itself. Metadata carries no
physics, so `complete` excludes it; a flag that fires on a correct file trains
the reader to ignore it, exactly as 493 transmissibility warnings did.

Writing the report also corrected a factual error in `keyword_support()`:
`SEQNUM` was listed as `read_but_unused`, when `read_restart` splits report
steps on it and a reader ignoring it merges a whole simulation into one step.
The keyword the restart reader most depends on was documented as unused.

**NNCs are now read** (`eclipse/nnc.py`), and the design point is where they
go. A non-neighbour connection is a flow path between cells that share no face
- across a fault throw, or between the parts an LGR splits - and on a faulted
model a large share of the cross-fault flow travels through them, so a case
imported without them is a *sealed* version of the model somebody built.

They are read into their own object and **not attached to the grid**. `Grid` is
MRST-shaped and has no place for a connection that is not a face; extending it
would put Research Studio state into the replica layer and break the covenant
that makes the parity worth anything. The consequence is a genuine limitation
rather than a detail, so it is stated plainly instead of buried: **an imported
case's NNCs do not flow into the standard solvers.** What they support is
inspection - that they exist, how many, which cells they join, what
transmissibility they carry - which is the difference between knowing a model
is sealed and wondering why it is.

Two decisions worth recording. ECLIPSE writes `NNC1`/`NNC2` as one-based
*global* indices; that is assumed, checked rather than trusted - every index
must land on an active cell - and selectable, with **no automatic detection**,
because on an all-active grid the two conventions are indistinguishable and a
reader that guessed would be right on every test model and wrong on real ones.
And records are concatenated rather than first-wins, since a large model splits
`NNC1` across several records and keeping the first would drop most of the
connections while appearing to work.

The coverage report gained a classification for this rather than stretching an
existing one: calling NNC `used` would claim the imported case contains those
connections, and `not_implemented` would hide that the data is reachable. It is
`available_separately`, and a file carrying NNCs is still reported **incomplete**,
because the case is sealed across them.

**Folding NNCs into the solvers is deferred** (owner decision, 2026-08-03, to
revisit). It needs a connection structure `Grid` does not have, and adding one
is a change to the replica layer rather than an addition to it. The reader and
the `available_separately` classification stand; nothing downstream should
treat solver integration as pending work.

### Work package M2.10: CMG adapter

Ingest CMG IMEX, GEM and STARS cases and results on the same contract as
M2.6, so a CMG case is a comparison reference like any other.

Nothing in PyMRST reads any CMG format today. This is a larger job than M2.6
and should be scoped after it: the ECLIPSE formats are openly documented and
already have an in-tree deck reader to align against, while CMG's are not,
so the supported subset has to be established from real cases rather than
from a specification.

Required:

- dataset-driven scoping: the supported subset is whatever a named set of
  real CMG cases needs, recorded as that list rather than as an intention;
- keyword coverage and refusal reporting, as in M2.6;
- unit-system capture, since CMG field/SI conventions differ from ECLIPSE's;
- a comparison configuration for the quantities that map cleanly, and an
  explicit list of those that do not.

### Work package M2.7: property/metamorphic/fault tests

Build reusable generators and transformations for flash, grids, units,
ordering, mechanics rotations, timestep transformations and injected faults.

### Work package M2.8: deck coverage and lineage

Instrument keyword lifecycle from parse through verification. Add strict
failure for parsed-but-unapplied keywords without accepted disposition.

### Work package M2.9: internal benchmark suite

Profiles:

- `quick`: deterministic CI-sized subset;
- `standard`: main numerical qualification;
- `field`: SPE10/Norne/large cases;
- `research`: formulation, surrogate or solver studies.

Store reports internally until public release is desired.

### Work package M2.11: grid inspection and quality reporting

Make a grid answer questions about itself before a simulation is attempted.

PyMRST already builds Cartesian, corner-point, faulted, PEBI and EDFM grids,
and `compute_geometry` computes what the discretization needs. What is missing
is the ability to interrogate one: a user with a field grid that produces a
strange answer has no way to ask the grid what is wrong with it.

Required:

- `validate()` returning a structured report, not a boolean: negative or
  near-zero cell volumes, twisted or non-convex cells, inconsistent face
  orientation, disconnected active regions, pore volumes below a stated
  threshold, extreme transmissibility ratios between neighbours, fault
  connections that do not pair, wells perforating inactive or pinched cells;
- `explain_connection(a, b)` returning why two cells are or are not connected,
  and through which faces and transmissibilities;
- `inspect_cell(i)` returning geometry, neighbours, properties and the cell's
  rank in any recorded residual;
- a severity distinction between *invalid* - the discretization cannot be
  trusted - and *unusual*, which is a warning about a grid that is legal and
  probably not intended.

Findings are reported against cell and face indices that match what the
plotting and export layers use, so a reported cell can be located.

**The transmissibility check is done, and the design was the work.** Measured on
a healthy Norne mini: interior face transmissibility spans 5e-301 to 3.6e-11,
and 493 interior faces sit more than four orders below the median. Those are
the sealed kz of the layer-4 shale - geology the case setup announces in its
own header - so a plain ratio threshold reports 493 warnings on a correct grid
and trains the reader to ignore the check. It separates `T = 0` exactly,
which is deliberate and reported at INFO, from `T` small but non-zero against
its neighbours, which is a possible geometry problem. On Norne that turns 493
warnings into one informational line and leaves the warning channel clean. Two supporting facts: MRST's
`compute_trans` returns *half*-face values, so a face transmissibility is the
harmonic average of two; and `g.faces.tag == 1` marks a fault face, which is
what the unpaired-fault check needs.

**Twisted cells and face orientation are done**, in
`pymrst/research_studio/inspection/geometry.py`, and they complete the required
list above. Both are anchored to a consequence rather than to an aesthetic:

- a *twisted* cell is one whose centroid falls on the wrong side of one of its
  own faces, which is exactly `compute_trans` with the permeability removed.
  The port's MRST-faithful default is `fix_negative=True`, which **replaces the
  resulting negative transmissibility with its absolute value** and warns to
  stderr. Measured on a deliberately spiked corner-point cell: 2 of 110
  half-transmissibilities negative, the returned array exactly `abs()` of the
  raw one. The defect is therefore repaired before the user sees anything, and
  a test asserts that identity so the justification cannot quietly stop being
  true. The check reports the same cells the negative transmissibilities land
  in - asserted, not assumed;
- a *reversed* face normal breaks the convention that normals point from
  neighbour 0 to neighbour 1. Nothing crashes; flow through that face simply
  runs backwards and reads downstream as a plausible wrong answer.

**Thresholds again set so a healthy real grid is silent.** Norne mini's most
oblique half-face is at `cos = 0.489`, about 61 degrees, and is legitimate
corner-point geometry, so the warning sits at 0.2 and Norne reports nothing.
The error condition is a sign change and needs no threshold at all.

Writing this exposed a third defect class and a bad test. A grid whose
`cells.faces` and `faces.neighbors` disagree makes `compute_trans` take its
`else` branch and silently flip the normal - so every projection computed from
it has an untrustworthy sign. That is now its own finding, checked first, and
the geometry checks decline to run rather than report a twist that is really an
inconsistency. The existing disconnected-region test had been building exactly
such an impossible grid by blanking both neighbour entries by hand; it now
deactivates a cell instead, which is how a real model becomes disconnected.

### Work package M2.12: dimensional safety

Refuse quantities whose dimensions do not match what an argument means.

`pymrst.utils.units` already defines 53 unit symbols and a unit-system helper,
so a user can *write* `300*BARSA`. Nothing checks that what arrives at a
compressibility argument has dimensions of one-over-pressure. Reservoir inputs
mix field and SI conventions - psi, barsa, mD, Darcy, reservoir barrels,
stock-tank barrels - and a dimensionally wrong input produces a plausible
number, not an error.

**Opt-in, and off by default.** This is the one package in the plan that can
break the covenant while looking additive: a check that refuses input PyMRST
accepts today *changes the behaviour of the port*, even though it adds no
feature. A user who has not asked for dimensional checking must get the same
result and the same exceptions as before, including for input that is
dimensionally wrong. Enabling it is a deliberate act - a context manager, a
constructor flag or an environment setting - never a default.

Required:

- dimensions attached at the public model-construction boundary, where a user
  supplies data, rather than through the numerical core - the inner loops must
  stay plain arrays;
- a no-op path when disabled, evidenced the way the observation hook is: a
  characterization case run with checking off must be bit-identical to one run
  before the feature existed;
- an error naming the expectation and what arrived: *expected compressibility
  [1/pressure], received permeability [length^2]*;
- a declared internal base system, with conversion at the boundary only, so
  the core has one convention and one place where it is established;
- input and display units chosen by the user, independent of the internal one;
- coverage measured as the fraction of public entry points whose arguments
  carry a declared dimension, so the gap is visible rather than assumed
  closed.

This is deliberately scoped to the boundary. Threading units through every
operator would slow the core and duplicate what the AD layer already carries.

### Milestone 2 gate

- Comparisons expose mappings, convergence quality and independence.
- Known bad/unconverged references are detected.
- Guardrails catch injected indexing, unit, topology and convention faults.
- A grid with an injected defect - negative volume, unpaired fault connection,
  disconnected region - is reported by `validate()` with the offending index.
- A dimensionally wrong argument at a public entry point is refused, naming
  both the expected and the received dimension.
- Analytic cases meet error/order expectations.
- MRST/OPM comparisons are reproducible for supported overlap.
- ECLIPSE and CMG cases load with their supported keyword set stated, and an
  unsupported keyword is refused rather than skipped.
- Benchmark reports regenerate from manifests.

## 9. Milestone 3: experiment and result platform

### Objective

Make controlled research studies, ensembles and scenario comparisons first-
class, persistent workflows.

### Work package M3.1: experiment schema and registry

Define:

- hypothesis/description;
- base case;
- parameter and component variants;
- datasets;
- execution profile;
- validation requirements;
- outputs and publication artifacts.

Experiment identity must derive from canonicalized definitions and input hashes.

M3.1 is implemented under `research_studio/experiments/`. Versioned immutable
contracts cover the hypothesis and description, content-addressed base case,
model and schedule owners, named explicit variants, datasets, execution
profile, validation policies, requested outputs and publication artifacts.
Only strict JSON-safe values are accepted; non-finite floats, arbitrary Python
objects and non-string mapping keys are refused rather than serialized by
accident.

ADR 0006 separates scientific identity from execution caching. The complete
definition produces a full-length SHA-256 `ExperimentId`, while the execution
fingerprint excludes prose, validation and publication presentation. Dataset
paths are retained as provenance but excluded from identity; logical dataset
name, SHA-256, size and media type are included. Consequently, moving an
unchanged deck preserves identity, changing its bytes invalidates identity,
and revising a hypothesis changes the scientific record without forcing an
identical simulation to rerun.

`ExperimentRegistry` is versioned, atomically persisted, idempotent for an
identical definition, refuses name ambiguity and checks content-address
collisions. Named explicit variants receive deterministic `MemberId` values
and retain their comparison group, parameter paths and component choices.
Cartesian and sampled expansion are provided by M3.2.

The first persisted registry artifact is
`docs/research_studio/experiments/registry.json`. It references the actual
SPE1 deck (`88756d94...35ebf`), produces experiment
`experiment_06a027...1e643`, execution fingerprint `eac319...aad53`, and
three deterministic FIM/explicit/AIM members. M3.1 remains `validating` until
a second case family uses the schema and the execution layer consumes a member
coordinate without translation glue.

### Work package M3.2: variant matrix

Support Cartesian, explicit and sampled variants. Every result retains its
variant coordinate and comparison group.

M3.2 is implemented in `experiments/variants.py` without coupling definition
to execution. `ExplicitVariantMatrix` materializes named alternatives;
`CartesianVariantMatrix` forms a deterministic product of parameter paths and
component slots; `SampledVariantMatrix` supports uniform, log-uniform, normal
and discrete-choice parameters. Duplicate axes, invalid distribution bounds,
empty choices, nonpositive standard deviations and unsupported sampling
algorithms are refused.

Sampling uses the recorded `pymrst_pcg64_uniform_box_muller_v1` algorithm and
an explicit nonnegative seed. Normal draws use the specified Box--Muller
transform rather than delegating to an unstated distribution implementation.
The matrix identity includes the base execution fingerprint and canonical
matrix specification. Each materialized definition receives its own scientific
experiment ID, and every member retains ordinal, variant name, comparison
group, exact parameter patch and component selection. Persisted artifacts
recompute and verify the matrix ID and all member coordinates when loaded.

The repository artifact
`docs/research_studio/experiments/variant_matrices.json` contains three real
SPE1 studies: the 3-member explicit time-integration matrix, a 6-member
Cartesian nonlinear/linear-solver matrix, and an 8-member seeded permeability
and well-rate uncertainty ensemble. M3.2 remains `validating` until M3.3
executes these coordinates and proves that stored results retain them through
failure, resume and aggregation.

### Work package M3.3: ensemble execution

Implement local process parallelism first:

- bounded workers;
- deterministic member IDs/seeds;
- independent result stores;
- member checkpoints;
- failure isolation;
- resume without repeating completed members;
- memory-aware aggregation.

Distributed adapters follow behind the same worker protocol.

**Local execution is implemented and validating.**
`experiments/execution.py` defines a serializable member-runner boundary and
one atomic, independent store per coordinate. `experiments/ensemble.py` caps
the process pool by both the requested worker count and available memory,
derives deterministic seeds from experiment seed plus member identity,
isolates failures, returns failed checkpoints to retries, skips completed
members on resume, and aggregates scalar records without loading field arrays.
The persisted SPE1 matrix ran as three actual OS processes with zero failures;
the qualification artifact records the process backend rather than silently
substituting a thread pool. A sandbox-only sequential fallback exists for
hosts that prohibit POSIX process semaphores and is named in the report.

### Work package M3.4: scenario comparison

Implement field, region, well, production, convergence, conservation and
performance comparisons with time/space alignment.

**Implemented and validating.** `experiments/comparison.py` compares named
quantities from every normalized result category, interpolates candidate time
histories onto reference report times, accepts explicit entity mappings, and
keeps performance comparisons separate from physics acceptance. The real
SPE1 evidence compares FIM with AIM and explicit transport using declared
cross-formulation engineering tolerances and records worst value, time and
entity for every field and well quantity. These limits are not presented as
MRST-parity limits; they answer whether deliberately different integration
methods remain within the experiment's engineering envelope.

### Work package M3.5: result formats

Implement behind `ResultStore`/export protocols:

- Zarr;
- xarray;
- VTK, written to disk as a ParaView-readable case (see below);
- Parquet/CSV summaries;
- lazy region/field/timestep access.

Round-trip tests must quantify any precision or metadata loss.

**All listed formats are implemented and validating.** A member store supports
lazy category/field/timestep reads without loading other arrays.
`NormalizedResultDataset` maps the same store to xarray with lineage attrs and
Zarr; `tabular.py` emits matching long-form CSV and Parquet summaries. In the
real three-member SPE1 qualification every Zarr array reloaded bit-identically,
CSV and Parquet metadata matched exactly, and CSV numeric text incurred at
most one floating-point ULP at the value scale (the measured worst absolute
loss was 7.451e-9 Pa). The artifact records that loss rather than calling CSV
lossless.

`pymrst.research_studio.export.write_vtk_series`
writes a `.pvd` collection over per-step `.vtu` files indexed by simulation
time, wells as a polyline `.vtp`, with the polyhedral cell array built once
rather than per step, and a `read_vtk_series` that exists so the round trip is
verified rather than claimed - values compare bit-identical, not within a
tolerance. Exercised on Norne mini's 605 corner-point cells. The same export
package now owns every planned format so numerical owners remain independent
of optional research and visualization dependencies.

The original note, retained because it was the reason this was reachable:
`pymrst/plotting/_pyvista_grid.py` already converts any PyMRST grid -.
Cartesian, corner-point with faults, VE top surfaces - into a VTK unstructured
grid of polyhedral cells, which is the part that is hard to get right. What is
missing is writing it: nothing in the tree emits a `.vtu`, `.vtk` or `.pvd`
file. Required:

- one unstructured-grid file per report step, carrying the cell data a caller
  names rather than a fixed set;
- a `.pvd` collection indexing those files against simulation time, so
  ParaView opens a schedule rather than a snapshot;
- wells as a separate polyline dataset, since ParaView cannot infer them from
  cell data;
- geometry written once and referenced, not repeated per step, for field-scale
  grids;
- a round trip that reads the written files back and requires the cell values
  to be bit-identical, so "quantify precision loss" is answered with zero
  rather than with a tolerance.

This depends only on the grid conversion and `ResultStore`, both of which
exist, so it can be built ahead of the rest of Milestone 3.

### Work package M3.6: branching

Allow an experiment to branch from a compatible checkpoint with changed
controls, properties or components while retaining ancestry and comparison
links.

**Implemented and validating.** `experiments/branching.py` selects an exact
parent report state from an independent M3.3 store, verifies required state
fields and parent identity, hashes the branch definition and changed inputs
into a deterministic child identity, and records the parent, branch point,
controls, properties, parameters and components in the child result. The
integrated qualification continues the same real SPE1 checkpoint as both FIM
and AIM and compares those children through M3.4. Wrong parents, missing state
variables, invalid report indices and child times preceding the branch point
are hard failures.

### Work package M3.7: research package generator

Generate manifest, environment, cases, evidence, comparisons, figures, tables,
report and reproduction scripts from the experiment registry.

**Implemented and validating.** `experiments/publishing.py` produces the full
directory layout, captures the environment, copies explicitly classified
artifacts, writes an executable reproduction recipe and content-addresses all
scientific inputs. `package_manifest.json` inventories every byte; verification
detects additions, removals and mutation. The generated M3 qualification
package was executed from inside its own directory and reconstructed and ran
its portable SPE1 case successfully.

### Work package M3.8: outbound deck and result export

Emit a case PyMRST set up, or ran, in a form ECLIPSE or CMG will execute.

Every external-simulator package before this one is a *reader*. Writing is not
the mirror image of reading, and the difference is the whole design problem:
a reader may ignore a keyword it does not support and still produce a usable
case, but a writer that omits one produces a deck the other simulator either
rejects or - the case that matters - runs as a different problem without
complaint.

So this package is defined by refusal rather than by coverage:

- a declared supported subset, and a **hard failure** when a case uses
  anything outside it. Emitting a partial deck is not an option this package
  offers;
- deck export for the model families that already round-trip on input, so the
  claim is anchored to cases PyMRST demonstrably represents;
- result export - UNRST-shaped solution arrays and UNSMRY-shaped summary
  vectors - so a PyMRST run can be opened in tooling built around those;
- unit and convention capture written into the output, never left implicit;
- provenance: the emitted case carries the `ReproducibilityManifest` of the
  run that produced it.

**Round-trip is the acceptance evidence, and it must close both ways.** A deck
written from a case and read back must reconstruct the same model to the
tolerance M2.1 defines; and where an external simulator is available, running
the emitted deck must reproduce PyMRST's own result for that case within the
comparison configuration M2.5 and M2.6 establish. Without the second half this
package can only demonstrate that PyMRST agrees with itself.

Scoped after M2.6 and M2.10 deliberately: the supported subset for writing
cannot be settled before reading has established what the overlap actually is.

**Addendum - the ECLIPSE half is built.**
`pymrst/research_studio/eclipse/export_case.py`, reusing M2.6's record layer
rather than adding a second writer. `write_eclipse_case` emits EGRID, and
INIT/UNRST when there is data for them.

The refusals above are implemented as refusals, not warnings: a state field
with no ECLIPSE keyword, a rock with porosity but no permeability, an array
whose length is not the cell count, and a mismatched times/states pair each
raise `UnsupportedForExport`. The corner-point description is a *required
argument* rather than reconstructed from the processed grid - that
reconstruction is lossy across faults, and a silently approximated fault is
precisely the failure this package exists to prevent. `export_support()`
returns the supported set and, more usefully, the unsupported one.

The round trip closes on Norne mini: 605 active cells, rock and three report
steps written out and read back matching to 1e-6. Seven mutations were
reintroduced and all seven were caught, including omitting the `SEQNUM` step
boundary - the defect M2.6's own docstring warns about - which collapses three
report steps into one.

**What this does not yet deliver, against the requirements above.** The list
matters more than the part that works:

- it writes a *result set*, not a *deck*. Wells, schedule, PVT and relative
  permeability are not written, so an exported case is loadable elsewhere but
  not runnable. `export_support()["not_written"]` says so explicitly rather
  than leaving it to be discovered downstream;
- the CMG half is not started, and is blocked on M2.10;
- provenance - emitting the run's `ReproducibilityManifest` with the case - is
  not done;
- **the second half of acceptance is untried, not passed.** No external
  simulator exists in this tree, so "PyMRST agrees with itself" is exactly and
  only what the evidence shows. The test asserting the round trip says this in
  its own docstring, so a later reader cannot mistake it for more.

**Further implementation in the combined M3.6--M3.9 pass.** Every binary
result export can now carry its `ReproducibilityManifest`. A strict
single-phase-water `.DATA` subset writes RUNSPEC, GRID, PROPS, SOLUTION,
SUMMARY and SCHEDULE with an explicit METRIC or FIELD declaration, PVTW,
pressure and TSTEP data. Wells, multiphase physics, relative permeability,
aquifers, regions, faults, NNC and LGR are refused before a file is created.
This makes the supported subset structurally runnable, but does not change the
acceptance boundary above: no independent ECLIPSE executable has run it, and
CMG remains blocked on M2.10.

### Work package M3.9: portable case definition

A `ReservoirCase` that fully describes a run, independent of how it executes.

Grid, properties, physics, discretization, facilities, schedule and solver
settings are currently entangled in constructed Python objects: a case exists
only as the script that built it. That is why the same case cannot be run from
a command line, a browser or an API without rewriting it each time, and why
M3.1's experiment identity has to hash a definition that does not yet exist as
an artifact.

Required:

- a declarative case object covering grid source, properties, physics choice
  and phases, discretization, facilities, schedule and solver configuration;
- serialization to YAML or JSON that a human can read and edit, and that
  round-trips to the same constructed model;
- a schema version, under ADR 0004's rule, and refusal of an unknown one;
- a content hash over the canonicalized definition, which is what M3.1's
  experiment identity should derive from;
- construction of the same case from an ingested ECLIPSE or CMG deck (M2.6,
  M2.10), so an external case becomes a first-class PyMRST case rather than a
  one-way import;
- explicit separation of what the case *is* from how it *runs* - device,
  precision and parallelism belong to the execution profile, not here.

The round trip is the acceptance evidence: a case written out and read back
must produce a model that reproduces the original run bit-identically under
`--check`. Anything the format cannot express must be refused at write time
rather than dropped silently, on the same rule as M3.8.

**Implemented and validating.** `research_studio/case/` defines schema-v1
`ReservoirCase`, content-addressed source artifacts, strict component owners,
all seven scientific sections and JSON persistence. Execution-only device,
precision and parallelism keys are rejected from the case schema. A serialized
two-step SPE1 FIM case reconstructs and reruns with bit-identical time,
pressure, saturation, solution ratio, well rates and iteration history.
EGRID/INIT/UNRST sources can be promoted into a portable case and are hash
verified before construction; a changed source is refused. CMG construction
will use the same boundary when M2.10 exists.

### Milestone 3 gate

- Interrupted ensembles resume without repeating accepted work.
- Large histories operate lazily/out of core.
- Scenario alignment is correct for cells, regions, times and wells.
- Branch ancestry and changed inputs are explicit.
- A case serialized and reloaded reproduces its original run bit-identically,
  and an inexpressible case is refused at write time.
- Research packages reproduce generated figures and tables.
- Result-format round trips meet declared tolerance.
- A VTK export opens in ParaView as a time series and reads back bit-identical.
- An emitted deck round-trips to the same model, and a case outside the
  declared supported subset is refused rather than partially written.

## 10. Milestone 4: hybrid research platform

### Objective

Support safe, measurable substitution of numerical components and advanced
optimization/research workflows.

### Work package M4.1: capability registry

Models, solvers and components declare phases, physics, derivatives, restart,
device and schema capabilities. Validate combinations before execution.

**Implemented and validating.** `research_studio/components/registry.py`
declares versioned owners and their phases, physics, derivative modes, restart,
devices, precision, schemas, features, dependencies and conflicts. Validation
returns every incompatibility before resolving or constructing a numerical
owner. The initial registry resolves the real MRST-compatible black-oil model,
SciPy direct solver and NumPy/SciPy sparse AD backend.

### Work package M4.2: surrogate contract

Define:

- input/output schema;
- units and composition basis;
- validity domain;
- training-data fingerprint;
- uncertainty/OOD mechanism;
- invariant requirements;
- physics fallback;
- reference sampling policy;
- batching/device support.

**Implemented and validating.** `components/surrogate.py` defines named tensor
fields with units, basis, dtype and dynamic batch dimensions; coupled validity
domains including composition-simplex checks; training fingerprints;
uncertainty modes; invariants; explicit physics fallback; deterministic
reference sampling; devices and batch limits. No surrogate is installed or
selected merely by declaring this contract.

### Work package M4.3: guarded runtime

Record:

- surrogate/physics calls;
- OOD events;
- invariant failures;
- fallback locations/reasons;
- sampled reference errors;
- accumulated output error estimates;
- speedup and device utilization.

**Implemented and validating.** `GuardedSurrogateRuntime` validates before
prediction and records every surrogate call, OOD event, invariant or
uncertainty failure, physics fallback, reference sample, measured reference
error, accumulated error estimate, device and elapsed time. The integrated
qualification deliberately issues one in-domain and one OOD batch and accounts
for both. Utilization is presently a declared device rather than a hardware
counter; GPU telemetry belongs with an installed GPU backend.

### Work package M4.4: optimization and differentiation

Integrate existing adjoints and optimization modules with experiment evidence.
Use implicit differentiation for converged simulator solutions and dense
autodiff backends for suitable surrogate components.

**This is packaging, not new numerics.** PyMRST already carries its own AD
system under `pymrst/utils/adi/`, adjoint gradients in
`modules/optimization/adjoint.py` and `modules/ad_mechanics/adjoint/`, and one
of the eight frozen characterization cases is an adjoint uplift gradient with
a finite-difference cross-check. What is missing is a single declarative
surface over them:

- one entry point taking an objective, the variables to differentiate with
  respect to, and constraints, rather than a per-module adjoint call;
- objectives and constraints as composable, named, reusable objects;
- gradients reported with the evidence of the runs that produced them, so an
  optimization result carries its own provenance;
- a finite-difference cross-check available on demand for any objective, since
  an adjoint that is silently wrong is worse than none.

Positioning follows from this rather than from new capability: the platform
lets a user *design and calibrate* a subsurface system, not only simulate one.

**Implemented and validating.** `research_studio/optimization/` supplies the
single declarative surface: reusable named objectives and constraints,
variable bounds, normalized conservative KS aggregation, adjoint/analytic
gradient providers, on-demand central finite-difference checks and evidence-
bearing results. Existing module adjoints can be passed as providers without
moving or replacing their numerical kernels. The integrated gradient agrees
with finite differences below 1e-9.

**Constraint formulation.** Pointwise limits over cells and timesteps must be
normalized before aggregation - a pressure margin and a slip-tendency margin
are not comparable in raw units. A direct `max` over cells and steps is
nonsmooth: the identity of the critical cell changes during optimization,
which switches the active constraint and destabilizes the gradient. Aggregate
with a Kreisselmeier-Steinhauser function in its unnormalized log-sum-exp
form, which bounds the true maximum from above and so enforces conservatively,
with an approximation gap of order `log(N)/rho`. Implement it shifted by the
maximum for numerical stability, and sharpen `rho` by continuation rather than
starting large - a smooth landscape first, then a tight one.

**Do not price and constrain the same quantity.** An objective carrying a
large leakage penalty *and* a hard leakage constraint double-handles one
physical fact, and the two can fight. Choose either constrained maximization
with leakage as a limit, or an economic objective with leakage priced and only
pressure and slip constrained. For a flagship demonstration the constrained
form reads better, because each limit has a direct engineering meaning.

### Work package M4.5: performance laboratory

**Measure before designing, and in that order.** The temptation with sparse
assembly is to design an assembly-plan abstraction first. The cheaper question
comes first: how much of the cost is SciPy re-validating and canonicalizing
structures PyMRST already knows are valid? Full CSR format checking scans
arrays and may cast index types; constructing from canonical `(data, indices,
indptr)` with `copy=False` may skip most of that - but `copy=False` promises
only to avoid copying where possible, not that validation disappears. So this
is benchmarked against the installed SciPy, not assumed from documentation.

Measure separately: value generation, object construction, validation and
canonicalization, COO/CSR/CSC conversion, peak allocation, and equality of
both structure and values. If direct canonical construction removes most of
the overhead, stop there. Do not build an architecture because one would be
elegant.

Then, and only then, the duplicate-reduction kernel. `np.add.at` performs
unbuffered in-place operations and is a poor default for a hot aggregation;
`np.bincount` with `weights` handles weighted accumulation by integer bin
directly. Three live call sites use `np.add.at` today -
`utils/cell_flux_to_face_flux.py` (twice), `utils/face_flux_to_cell_velocity.py`
and the pattern documented in `utils/gridtools/grid_cell_no.py` - so this is a
concrete substitution to benchmark, not a hypothetical one. Candidates to
compare on real PyMRST structures: canonical CSR filling, `np.bincount`,
stable sort plus segmented reduction, `np.add.reduceat` after grouping, and a
compiled kernel only if NumPy remains limiting.

The progression is: avoid unnecessary construction, then unnecessary
validation, then reuse the pattern, then vectorize duplicate reduction, then
compile what remains. Not: invent an abstraction first.

**Reduction order changes floating-point sums, and that must be quantified
rather than discovered.** Any candidate is run against the F0.3 baseline and
its drift reported as a number, before it is adopted - which is what the
structural hash addendum above exists to support.

Measure sparse allocation, assembly, factorization reuse, solve time, memory
and backend accuracy. Implement in-place sparsity reuse before selecting GPU or
external solver backends.

#### M4.5 acceleration ladder under the MRST-compatible backend contract

Acceleration is an implementation choice beneath the existing `ADI`,
`SparseAutoDiffBackend`, `DiagonalAutoDiffBackend`, state-function and model
interfaces. It is not a replacement API and does not change the MRST-shaped
layout. The current NumPy/SciPy implementation remains the reference backend
and fallback. Introduce the following candidates in measured tiers:

1. **Fully vectorized NumPy/SciPy assembly.** Remove Python loops from
   transmissibility, accumulation, divergence, upwind, block concatenation and
   duplicate-reduction hot paths where profiling demonstrates material cost.
   Preserve canonical sparse ordering and compare both Jacobian structure and
   values with the reference implementation.
2. **Reusable Jacobian sparsity patterns and preallocated buffers.** Separate
   topology-dependent indices from timestep-dependent values. Cache patterns
   by grid, discretization, primary-variable layout, facility topology and
   active control configuration; invalidate explicitly when any key changes.
   Refill owned buffers without leaking mutations between Newton iterations or
   concurrent simulations.
3. **Batched property and flash calculations.** Add array-first batch APIs for
   PVT, relative permeability, phase stability and flash operations. Scalar and
   single-state calls delegate to the batch contract so there is one numerical
   definition. Record batch size, memory peak, phase-boundary cases and
   convergence/fallback counts.
4. **Numba-compiled diagonal operations.** Compile only profiled compact
   Jacobian kernels—row gathering/scattering, diagonal products, divergence,
   upwind selection, block reduction and value refill. Numba is optional;
   unsupported platforms and dtypes use the reference Python path. Compilation
   warm-up is reported separately from steady-state runtime.
5. **Optional C++/pybind11 kernels.** Introduce compiled CPU kernels only where
   vectorization and Numba leave a demonstrated field-scale bottleneck. Keep a
   narrow C-compatible array contract, no Python-owned model logic in C++, and
   wheels as an optional extra. Every kernel has the reference implementation
   as an executable oracle and fallback.
6. **JAX or PyTorch execution for differentiable/GPU components.** Do not
   pretend that traced autodiff is interchangeable with MRST's materialized
   Jacobian-block contract. Start at declared component boundaries—batched
   property models, flash/surrogate closures and selected residual kernels—and
   bridge derivatives explicitly into PyMRST AD blocks. A broader experimental
   backend may follow only after profiling and parity evidence justify it.

Every tier must satisfy the same acceptance sequence:

- reference and accelerated paths share identical public inputs and output
  schemas;
- unit tests compare values, Jacobian blocks, sparsity patterns, active-set
  decisions and raised exceptions;
- the F0.3 characterization suite passes with acceleration disabled, proving
  that installing optional acceleration does not alter the replica path;
- enabled runs pass the applicable MRST fixtures, AD-versus-finite-difference
  graph, conservation/invariant checks and cross-solver qualification;
- performance artifacts separate cold-start, warm steady-state, assembly,
  linear solve, memory and data-transfer costs on recorded hardware;
- no accelerated backend becomes a default until its numerical tolerances and
  field-scale speedup are accepted in the evidence ledger.

Suggested ownership keeps acceleration out of MRST-equivalent model files:

```text
pymrst/backends/reference/       # existing NumPy/SciPy behavior
pymrst/backends/vectorized/      # vectorized assembly and pattern reuse
pymrst/backends/numba/           # optional compiled diagonal kernels
pymrst/backends/cpp/             # optional pybind11 kernels
pymrst/backends/jax/             # experimental component adapters
pymrst/backends/torch/           # experimental component/autograd adapters
pymrst/modules/...               # unchanged MRST-shaped model owners
```

Backend selection is explicit at model/backend construction or experiment
configuration. Serialized evidence records backend name, version, compiler,
device, precision, cache state and fallback events. `auto` selection may be
offered later, but it must resolve to a recorded concrete backend and must
never silently cross a qualification boundary.

**Laboratory and the first reusable acceleration primitives are implemented;
the compiled tiers remain implementing.**
`research_studio/performance/laboratory.py` records cold and repeated warm
runtime, peak allocation, absolute/relative drift, output shape, NumPy/SciPy
versions and hardware. The first measured candidates cover the three live
duplicate-reduction strategies and separately time COO-to-CSR conversion,
canonical CSR construction with `copy=False`, and full format validation.
Numerical reduction-order drift is recorded and bounded below 1e-12 in the
qualification. `performance/patterns.py` now separates immutable canonical CSR
indices from refillable Newton values, keys caches by topology,
discretization, primary variables, facilities and controls, provides explicit
invalidation and prevents mutable index buffers leaking between simulations.
Its bounded batch evaluator preserves the leading entity dimension and is
bit-identical to the vectorized property reference in qualification. This does
not claim field-scale speedup or that Numba/C++/JAX/Torch tiers are complete;
adoption still requires profiles from real full simulations.

### Work package M4.6: solver ecosystem

Benchmark and conditionally integrate PETSc, hypre, AMGCL, AMGX or CuPy-based
solvers where field-scale profiles demonstrate value.

**Reference ecosystem implemented and validating.** The solver laboratory
qualifies SciPy direct, CG and GMRES against a direct reference with residual,
solution error and runtime. PETSc, CuPy and AMGX are registered as optional
and report their missing dependency instead of disappearing or falling back.
They are not integrated because none is installed and no field-scale profile
yet justifies choosing one. The same availability-aware contract is ready for
hypre and AMGCL adapters when their bindings and benchmark cases are selected.

### Unassigned identifiers M4.7--M4.9

The reviewed roadmap defines six Milestone 4 work packages. M4.7, M4.8 and
M4.9 have no requirements or ledger entries and therefore are not described as
implemented. New packages may use those identifiers only after their purpose,
acceptance evidence and dependency boundaries are added to this plan.

### Milestone 4 gate

- Surrogates never silently execute outside policy.
- Physics fallback and accumulated error are reported.
- Physics/surrogate variants share experiment definitions.
- Gradients pass adjoint/finite-difference qualification.
- Performance claims include accuracy, hardware and field-scale evidence.
- Component incompatibilities fail before numerical execution.

## 11. Milestone 5: Research Studio and benchmark portal

### Objective

Expose stable platform services through engineer- and researcher-centered
notebook/browser workflows.

### Work package M5.1: API service layer

Expose experiments, sessions, events, stores, evidence and comparisons through
a UI-independent service/API layer.

**Implemented and validating.** `studio/service.py` owns projects, registered
workflows, bounded concurrent execution, immutable run identity, lifecycle
events, progress, cooperative cancellation, failure isolation, result access
and parent-linked resume/branch operations. `studio/web.py` exposes this
surface through a loopback-by-default JSON API with bounded request bodies;
explicit non-loopback binding requires both an enable flag and bearer token.
`StudioMetadataStore` transactionally persists projects, runs and ordered
events in SQLite/WAL and passes close/reopen recovery. Importable production
workflows can execute behind `IsolatedWorkflowExecutor`, whose subprocess is
forcibly terminated on hard wall timeout or cancellation. The real HTTP
contract passed outside the restricted test sandbox. Arbitrary Python
owners cannot be submitted over HTTP: only workflows registered by the host
process execute.

### Work package M5.2: notebook integration

Provide rich representations, progress, convergence/conservation views,
scenario comparisons and publication exports.

**Implemented and validating.** Dependency-light `RunView`, `ExperimentView`
and `ComparisonView` rich representations display status, progress, elapsed
time, output requests and per-quantity comparison errors without duplicating
simulation logic. Untrusted status and result labels are HTML-escaped. Package
publication remains the M3.7 service rather than notebook-specific code.

### Work package M5.3: browser workspace

Implement:

- projects and experiments;
- deck/property review;
- run monitoring/cancellation/resume;
- field, slice and well visualization;
- scenario comparison;
- failure diagnosis;
- evidence graph exploration;
- report/package export.

**Implemented and validating with a real result-store integration.** The responsive local
workspace provides project and run summaries, live progress, accepted-step
events, explicit teaching guardrails, publication safety and navigation for
experiments, checkpoints, comparisons, evidence and packages. It polls only
M5.1 JSON endpoints; no numerical equation or engineering calculation exists
in JavaScript. A secured result catalog now exposes bounded lazy slices from a
completed M3 member store and only HTML viewers inside configured roots. The
launcher discovers a completed SPE1 integration result, reconstructs its real
300-cell grid through the existing black-oil setup, and embeds the existing
interactive 3-D field/slice/well/timestep browser. Workflow submission is also
functional from the page. The loopback HTTP and DOM contracts pass. Visual inspection via
the in-app browser could not start because the session lacks required
sandbox-policy metadata, and remains recorded as unverified rather than
silently replaced by a different browser driver.

### Work package M5.4: benchmark portal

Render versioned internal and eventual public benchmark evidence. Publication
status must be explicit; internal evidence is not exposed accidentally.

**Implemented and validating.** `BenchmarkPortal` versions and hashes evidence
files, distinguishes internal, candidate, public and withdrawn states, filters
the public endpoint, detects changed evidence and requires an explicit approval
flag for promotion to public. The integrated proof registers M4 evidence as
internal and confirms that the public API returns no entry.

### Work package M5.5: teaching runtime

Support bounded browser cases with declared limits. Do not imply field-scale
WebAssembly capability.

**Implemented and validating.** `TeachingRuntime` enforces cell, report-step,
ensemble-member, result-size and wall-time limits and exposes those limits to
the workspace. Its disclosure sets `field_scale_claim` to false, and the UI
states plainly that teaching mode is not a field-scale browser solver. Hard
wall enforcement for production-style importable workflows is provided by the
separate subprocess boundary rather than being misrepresented as an in-process
teaching guard.

### Milestone 5 gate

- No numerical logic is duplicated in the UI.
- Engineers can run, monitor, diagnose, resume and compare cases.
- Researchers can define and publish reproducible experiments.
- Evidence remains visible rather than being hidden behind simplified status.
- Benchmark reports are versioned and regenerable.
- Security, cancellation and concurrent-session tests pass.

## 12. Cross-cutting test strategy

### Unit tests

- schemas and identifiers;
- event ordering;
- conservation terms;
- invariant tolerances;
- failure classifiers;
- storage transactions;
- migrations;
- mappings and metrics.

### Contract tests

Every runtime entry point, model adapter, store, component and external
simulator adapter must pass a common contract suite.

### Characterization tests

Protect current numerical behavior before integration.

### Property and metamorphic tests

Target numerical identities and transformation invariance.

### Fault-injection tests

Verify that evidence and diagnostics detect intended defect classes.

### Integration tests

Exercise complete runtime → evidence → store → restart → comparison workflows.

### Field acceptance tests

Run separately from quick CI with explicit datasets, resource limits and
recorded artifacts.

## 13. Internal validation profiles

| Profile | Purpose | Typical scope |
|---|---|---|
| `unit` | Fast developer feedback | Kernels and schemas |
| `quick` | Pull-request gate | Small runtime/integration cases |
| `standard` | Numerical hardening | SPE1, compositional and mechanics matrices |
| `field` | Scale and robustness | SPE10, Egg, Norne |
| `research` | Method studies | Formulations, solvers and surrogates |

Each profile records environment, datasets, runtime, memory and pass/fail
evidence.

## 14. Documentation plan

Maintain:

```text
docs/research_studio/
  architecture.md
  runtime.md
  evidence.md
  conservation.md
  checkpoints.md
  validation_graph.md
  experiments.md
  components.md
  storage.md
  studio.md
  adr/
  schemas/
  migrations/
  internal_validation/
```

Documentation changes are part of each work package, not deferred until the
end.

## 15. Dependency strategy

### Required core

Keep the base numerical installation small. New foundational runtime/evidence
code should prefer the standard library plus existing NumPy/SciPy dependencies.

### Optional groups

Proposed extras after implementation justifies them:

```toml
storage = ["h5py", "zarr", "xarray"]
export = ["pyarrow", "meshio"]
validation = ["hypothesis"]
distributed = ["dask"]
solver-petsc = ["petsc4py"]
studio = [browser/notebook dependencies]
```

Do not add a dependency merely because it appears in the roadmap. Add it when
the owning capability and fallback behavior are implemented and tested.

## 16. Performance and non-regression policy

For runtime/evidence changes measure:

- evidence-disabled overhead;
- observe/strict profile overhead;
- output/store overhead;
- peak memory;
- event volume;
- checkpoint size and commit time.

Set explicit budgets after Foundation 0 measurement. Any regression outside
budget requires optimization, an accepted ADR or disabling an optional feature
in the default profile.

Numerical non-regression comparisons must include states, well histories,
solver decisions, conservation and achieved residuals—not only final scalar
outputs.

## 17. Risk register

| Risk | Mitigation |
|---|---|
| One session abstraction does not fit all workflows | Protocols and specialized adapters; do not force identical kernels |
| Evidence changes solver behavior | Immutable events, before/after characterization and decision comparison |
| Persistent schemas stabilize too early | Private versioning window, ADRs and migration fixtures |
| Conservation semantics are guessed | Mandatory model adapters and declared bases/signs |
| Checkpoints pickle unstable objects | Store explicit arrays/metadata only |
| Event volume becomes excessive | Configurable retention and streaming handlers |
| Long field tests slow development | Separate quick/standard/field profiles |
| UI duplicates business logic | UI-independent service layer and contract tests |
| Optional dependencies become mandatory accidentally | Extras, import guards and fallback tests |
| Surrogate silently leaves training domain | OOD policy, invariant guard and physics fallback |
| Backend speed changes accuracy | Joint accuracy/performance qualification |
| Broad roadmap creates shallow implementations | Milestone gates and incomplete/experimental labels |

## 18. Progress tracking

Create a machine-readable implementation ledger:

```text
docs/research_studio/implementation.csv
```

Recommended columns:

- milestone;
- work package;
- owner path;
- status;
- API/schema status;
- unit-test evidence;
- integration evidence;
- numerical validation;
- performance evidence;
- documentation;
- blockers;
- acceptance decision.

Statuses:

- `planned`;
- `designing`;
- `implementing`;
- `validating`;
- `accepted`;
- `blocked`;
- `experimental`.

Do not mark a work package accepted until its stated artifacts and tests exist.

## 19. Definition of done

The complete program is done when:

1. the MRST replica preservation covenant is satisfied: MRST-shaped owners,
   defaults, public entry points and certified numerical behavior remain
   available independently of Research Studio;
2. supported simulation workflows use or implement the common runtime/evidence
   protocols;
3. long simulations stream accepted work and resume safely;
4. primary and supported specialist models provide conservation and invariants;
5. failures produce retained evidence and actionable classifications;
6. formulation, analytic, MRST and OPM comparisons are represented in the
   evidence graph where applicable;
7. experiments, variants, ensembles and branches are persistent and
   reproducible;
8. learned components are guarded by validity, invariants, fallback and error
   monitoring;
9. large results operate lazily/out of core;
10. performance/backend claims are validated at field scale with accuracy;
11. Research Studio consumes platform APIs without duplicating numerics;
12. all schema migrations, corruption checks and compatibility contracts pass;
13. internal benchmark profiles regenerate from their manifests;
14. ordinary risk-ranked module coverage remains active;
15. documentation and machine-readable implementation status are current;
16. no incomplete capability is represented as accepted.

## 20. Immediate execution order

The first implementation activity should follow this order:

1. create `docs/research_studio/` and the implementation ledger;
2. inventory and classify execution paths;
3. write the runtime/evidence/storage ADRs;
4. select and freeze Foundation 0 characterization cases;
5. generate baseline numerical and performance artifacts;
6. implement identifiers and evidence schema without solver integration;
7. implement the event protocol and test handlers in isolation;
8. instrument the canonical nonlinear solver behind an opt-in path;
9. prove characterization equivalence;
10. implement the HDF5 result-store transaction model;
11. implement `SimulationSession` around the canonical schedule path;
12. expand entry-point integration one classified path at a time;
13. add checkpoint/restart and the first model-aware conservation adapters;
14. complete the Milestone 1 gate before starting Validation Laboratory
    implementation.

## Final implementation position

Research Studio should be built as a platform, not as a collection of isolated
features. The implementation order begins with architecture and
characterization, then establishes runtime, evidence and persistence before
validation, experiments, hybrid components and user interfaces.

This plan deliberately avoids rushing toward public release. It uses private
development time to make contracts durable, validate every cross-cutting change
and complete each integrated milestone end to end. When PyMRST is eventually
published as Research Studio, its benchmark reports and reproducibility
artifacts should be natural outputs of the platform rather than separate work
assembled to prove it after the fact.
