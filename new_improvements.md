# Proposed improvements: making PyMRST more robust than MRST

PyMRST now reproduces MRST closely enough that the interesting question is no
longer whether the port is faithful, but whether the port can be made harder to
get wrong than the original. This document proposes eight additions aimed at
correctness, then twelve aimed at capability, and it argues for one of each as
the place to start.

The two groups rest on different evidence, and the distinction is worth keeping
in view. Proposals 1 through 8 each trace to a defect we found while validating
the compositional module and the MATLAB file bridge, with numbers drawn from
`compositional_parity.md` and the `validate_phase114.py` docstring. The
capability gaps in the later section are argued instead from what MRST does not
offer and from what a Python implementation makes cheap, so they carry less
direct evidence and more judgement.

## What the evidence says

Four observations from recent validation work shape these proposals.

**Redundancy caught an error that reference data could not.** The
natural-variable compositional model appeared to sit 150 times further from
MATLAB than the overall-composition model, at 7.0e-02 bar against 4.6e-04 bar.
The decisive measurement was not the reference at all. PyMRST's two formulations
agreed with each other to 1.04e-10 bar, while MATLAB's two disagreed by 7.08e-02
bar. The reference had been dumped at MRST's default 1e-3 tolerances and was not
converged. Re-running it at 1e-7 collapsed MRST's internal spread to 6.3e-06 bar,
and both PyMRST models then matched to about 6e-05 bar. No PyMRST code changed.

**Unit and convention errors are the dominant silent failure.** PyMRST had no
ECLIPSE box-operator support, so a deck specifying `MULTIPLY PERMX 0.1` read
permeability ten times too high with no warning. Loading MRST grids requires
converting one-based indices to zero-based, where an off-by-one shifts every
connection by one cell. The compositional separator fed mole fractions where
mass fractions belong, and separately produced surface rates that differ by 117
times from the non-separator path because one uses a nominal surface gas density
of 100 kg/m3 and the other flashes to a real 1.1 kg/m3 at one atmosphere. None
of these announce themselves.

**Physics invariants go unchecked.** The K-value Rachford-Rice solver decided
phase state from whether every K value sat on one side of unity, which is
necessary but not sufficient. Single-phase liquid with `K_oil < 1 < K_gas`, which
is ordinary undersaturated oil, collapsed onto V = 1 and was reported as fully
vapour. On a sweep of that regime the error appeared in 176 of 4000 states, each
with the largest possible magnitude. SPE1 never reaches the branch, at 0 of
22200 evaluations, so no existing test moved.

**Untested code hides real defects.** Of 1042 modules, 574 (55%) carry no
self-test block, and 247 (24%) have neither a self-test nor any mention in
`tests/` or the validators. Writing the first tests for one such module, the
MATLAB file bridge, exposed three defects, including that PyMRST could not load
any unstructured MRST grid. Every PEBI, triangle, and tetrahedral grid failed,
because three call sites passed a `default=` while leaving `required=True`, so
the default could never be returned.

**Failures report too little to act on.** A failed step says only
`step 3: Newton failed after 20 timestep cuts`. Establishing that the
compositional separator test stalls just below 1e-10 rather than diverging took
a tolerance sweep and a timestep sweep, and the same question had already been
asked once for Norne. The solver holds everything needed to answer it directly.
Nothing is retained either: the simulators call `print()` 24 times, and the
`logging` module appears in none of the 1042 modules, so convergence history goes
to stdout and cannot be examined after a run.

## Proposal 1: cross-formulation agreement testing

**Recommended as the first addition.**

Any problem that PyMRST can solve two ways should be solved both ways, with the
results required to agree to solver tolerance. Disagreement means one path is
wrong, and the test reports which quantities diverge and by how much.

PyMRST already carries the redundancy this needs:

| Pair | What disagreement would reveal |
|---|---|
| overall-composition vs natural-variables | primary-variable handling, flash coupling |
| fully implicit vs sequential | splitting error, transport consistency |
| AD Jacobian vs finite differences | derivative errors that slow or stop Newton |
| direct vs iterative linear solve | preconditioner or scaling faults |

MRST cannot produce this signal, not because of any technical limit, but because
nobody runs a case both ways and compares. Its validation compares against
external references, so it can only check cases where a reference exists, and it
inherits whatever error the reference carries. Agreement testing needs no
reference, extends to any new case, and, as the phase 114 work shows, detects
errors inside the reference itself.

Design sketch:

```python
# pymrst/testing/agreement.py
def assert_formulations_agree(setup, formulations, *, atol, rtol=0.0, fields=None):
    """Solve `setup` under each formulation; require pairwise agreement.

    Returns a report naming the worst-disagreeing field and its magnitude so a
    failure points at a quantity rather than at a scalar norm.
    """
```

The compositional module already has the ingredients in
`compare_natural_overall`, so a first version mostly generalises existing code.

**Cost:** low. **Value:** high, and unique to PyMRST.

## Proposal 2: an opt-in dimensioned-quantity layer

`pymrst.utils.units` exports 53 plain float constants, matching MRST's design.
The consequence is that `5*BARSA + 3*DAY` evaluates to 759200.0 without
complaint. MATLAB cannot support dimension checking ergonomically, which is why
MRST does not try. Python can, at no runtime cost when the checking is
disabled.

The proposal is a `Quantity` wrapper carrying a dimension vector, enabled by an
environment variable or a context manager, with the existing float constants kept
as the default so no caller changes. Under checking, adding a pressure to a time
raises, and a function annotated as returning a permeability rejects a
transmissibility.

Two refinements matter more than the general idea:

1. **Index-base tagging.** Zero-based and one-based arrays are the same dtype and
   shape, so nothing catches a mix-up. A thin `OneBased` marker applied at the
   MATLAB boundary, cleared by the conversion helpers, turns a silent one-cell
   shift into an error. The `.mat` bridge converts `neighbors`, `cells.faces`,
   `facePos`, `nodes`, `nodePos`, `indexMap`, and well `cells`, and each is a
   separate chance to get it wrong.
2. **Basis tagging for compositions.** Mole fraction and mass fraction are both
   dimensionless arrays summing to one, so dimension checking alone does not
   separate them. The separator feed shows why the distinction needs a name.

**Cost:** medium, and it can be added incrementally at module boundaries.
**Value:** high, because it targets the class of error that produces wrong
answers rather than crashes.

## Proposal 3: runtime physics invariants

Add a set of always-true physical statements, checked when a debug switch is on
and skipped otherwise:

- saturations sum to one, and each lies in [0, 1]
- compositions are non-negative and normalised
- mass is conserved over a timestep to within solver tolerance
- the reported phase state is consistent with the flash that produced it
- vapour fraction lies in [0, 1], and equals 0 or 1 exactly when the mixture is
  single phase

The Rachford-Rice defect violated the last two and survived because no test
asserted physics rather than agreement with a stored array. An invariant layer
catches that class of error on any case a user runs, including cases with no
reference.

There is no global debug or strict switch in PyMRST today, so this proposal also
supplies one, which Proposals 2 and 4 can share.

**Cost:** low per invariant, and they can be added one at a time.
**Value:** high, because the checks work on user cases rather than only on
fixtures.

## Proposal 4: convergence provenance

MRST reports that a step converged without reporting to what. The phase 114
reference shows the cost: it was dumped at default tolerances, carried a 7.08e-02
bar internal inconsistency, and nothing in the workflow flagged it. Downstream,
that error was read as a PyMRST defect and cost real diagnostic effort.

The proposal is to stamp every solved state with the residuals actually achieved,
the tolerances requested, and the iteration counts, then to refuse, by default,
to export a result as reference data when its achieved residual exceeds a
threshold. Reference files gain a header recording the same information.

A related item belongs here. The default `tol_well = 1e-10` has now produced two
false failures from unrelated code: Norne missed it by a factor of 2.7 at a
residual of 2.69e-10, and the compositional separator test converges in 6
iterations at 1e-9 but never at 1e-10, regardless of iteration count or timestep.
In both cases the physics was sound and the tolerance was unreachable. Provenance
reporting makes such a floor visible instead of leaving it to be rediscovered.

**Cost:** low. **Value:** medium to high, and it prevents a whole class of
wasted investigation.

## Proposal 5: property-based testing of the numerical kernels

We found the Rachford-Rice defect by generating 4000 random but physically
plausible (z, K) states and comparing the solver against the textbook definition.
That is property-based testing carried out by hand. Wiring `hypothesis` into the
kernels where the correct answer is definable rather than merely referenceable
would automate it:

- flash and stability: the returned V zeroes the Rachford-Rice residual, or the
  mixture is single phase by the endpoint test
- EOS: fugacity coefficients agree between phases at equilibrium
- grid processing: face and cell counts, `facePos` monotonicity, every interior
  face carrying exactly two distinct neighbours
- unit conversion: round trips return the original value

**Cost:** low to medium. **Value:** high for the kernels, limited elsewhere.

## Proposal 6: a three-state deck coverage report

The box-operator defect is the pattern worth designing against. PyMRST parsed
`MULTIPLY PERMX 0.1` and then never applied it, so permeability was read ten
times too high with no warning. A deck reader that only separates known from
unknown keywords cannot catch that, because the keyword was known. It was known
and dropped.

The report therefore needs three states rather than two:

| State | Meaning | Risk |
|---|---|---|
| unknown | the reader does not recognise the keyword | visible, low |
| parsed, not applied | read into the deck, never reaching the model | silent, high |
| applied | read and used | none |

Print the summary when a case loads, and treat the middle row as the list of
silent-wrongness candidates. Part of the machinery exists: the reader already
records `UnhandledKeywords` per section. However, it never reaches the caller,
appearing downstream only in a skip list in `convert_input_units.py`, and it
tracks the first state alone. The middle state is the one that produced a wrong
answer.

**Cost:** low. **Value:** high, because it converts silent corruption into a
line of output at load time.

## Proposal 7: Newton failure forensics

On failure, report which residual is worst, its value, its trend across
iterations, the owning cell or well, and, above all, whether the residual
**stalled** or **diverged**. The distinction decides what to do next. A stall
means the requested tolerance is unreachable and the run needs a different
tolerance; divergence means the physics, the Jacobian, or the timestep is wrong.

Both cases we investigated were stalls read as failures. The compositional
separator converges in 6 iterations at 1e-9 and never at 1e-10, whatever the
iteration limit or timestep, because a separator flash inside the residual puts a
floor just under 1e-10. Norne misses the same 1e-10 by a factor of 2.7, at a
residual of 2.69e-10, while every other residual passes. Deducing either took
several runs that the solver could have answered in one line.

**Cost:** low, since the solver already computes every quantity involved.
**Value:** high, and it repays itself the first time a run fails.

## Proposal 8: structured logging

Convergence history is printed and then lost. Routing it through `logging` with
structured records, one per Newton iteration carrying the residuals by name,
makes failures analysable after the fact and makes Proposal 7 straightforward to
implement. Machine-readable logs also let a run be summarised or plotted without
re-running it, which matters for the long field cases where a repeat run costs
hours.

**Cost:** low. **Value:** medium alone, high as the enabler for Proposal 7.

## Capability gaps against MRST

The proposals above respond to defects. The twelve below respond to absences, and
they rest on judgement about value rather than on measured failures. Each is
something a Python implementation makes cheap and MRST either lacks, charges for,
or cannot do at all. Where a claim about the current state is checkable, such as
which solvers ship or which output formats exist, we state it; the ranking
between items is opinion.

**Standard-format result export.** The only writers are MRST-format
(`write_sim_data`, `write_grdecl`). Nothing writes VTK, HDF5, xarray, or
Parquet, so results stay inside Python objects. A VTK writer opens ParaView; an
xarray writer opens pandas and the rest of the scientific stack. MRST is confined
to `.mat` and its own plotting, so this is a place PyMRST can be better rather
than equal.

**Parallel execution for the ensemble driver.** The scaffolding already exists:
`pymrst/modules/ensemble` ports MRST's `simulateEnsembleMembers` as
`run_ensemble`, alongside a `history_matching` module. What is missing is
parallelism. `run_ensemble` evaluates its members in a list comprehension, and
neither `multiprocessing`, `joblib`, nor `concurrent.futures` appears anywhere in
the module. Ensemble work, meaning uncertainty quantification, history matching,
and sensitivity studies, is embarrassingly parallel, so the change is largely
swapping one comprehension for a pool map behind an `n_jobs` argument. MATLAB
users buy the Parallel Computing Toolbox per worker to get the same thing.
Scoped this way the item is far cheaper than it first appears, which moves it up
the list rather than down.

**A reproducibility manifest.** Stamp each result with an input hash, package
version, git commit, and solver settings, so a published figure can be
regenerated. Reservoir simulation results support papers, and MRST records none
of this.

**A static typing and lint gate.** `pyproject.toml` configures neither `mypy` nor
`ruff`. Several defects we hit are contract errors that typing catches:
`multisegment_equations.py` requires `model.raw_fluid` outright where three other
call sites fall back to `fluid`, and a test read a return value as an array after
it became a `PlotResult`.

**A differentiable backend.** PyMRST carries hand-written forward-mode AD in
`pymrst.utils.adi`. A JAX or PyTorch backend would add exact gradients and
just-in-time compilation, which unlock gradient-based history matching and
connect to the Victory surrogate work already in the tree. The payoff is the
largest available and so is the cost, so it belongs on a roadmap rather than in a
sprint. Note the scope: the case rests on gradients and on surrogate components,
not on GPU execution of the simulator core, for the reasons measured in the
section below.

**An independent external reference through OPM Flow.** Proposal 1 argues that
redundancy catches errors that reference data cannot, and the argument does not
stop at PyMRST's own formulations. OPM Flow is open source, industry grade, and
reads the same ECLIPSE decks, so agreement between PyMRST and OPM Flow is
evidence from two independent implementations rather than from one implementation
and a stored array. The phase 114 episode shows why the distinction matters,
since the stored array was itself wrong.

Much of the machinery already exists but sits in an example rather than in the
library: `ad_blackoil/examples/utils/load_aquifer_eclipse_result.py` reads
ECLIPSE `UNSMRY` and `UNRST` output and already provides `compare_states` and
`compare_aquifer_history`. Promoting it to a first-class module, extending it to
`EGRID` and `INIT`, and pointing it at OPM Flow would convert a one-off example
into a second reference for every deck-based case. Ranked against the other
capability gaps, we consider this the most valuable, because it strengthens the
correctness argument rather than only adding reach.

**Notebook-native interaction.** Nothing in the package defines `_repr_html_`,
and neither `tqdm` nor any progress reporting appears in the simulators, so a
long field run prints nothing about how far it has come. Rich representations for
grids, states, and wells, along with live convergence plots and progress bars,
would make PyMRST considerably better than MRST for teaching and for exploratory
work, where MATLAB Live Scripts remain awkward.

**Browser execution.** A WebAssembly build through Pyodide would let students run
real reservoir simulations with no install and no license. MRST cannot follow,
because MATLAB requires both. For teaching, and for reviewers who want to
reproduce a figure without setting up an environment, the gap is not incremental.
The existing Streamlit application under `compositional/ML_models/Web_app`
suggests the workflow already appeals.

**A modern linear-solver ecosystem.** The package ships two solvers, `CPRSolver`
and `DirectSolver`, with `pyamg` as an optional extra for the elliptic stage of
CPR. Bindings to PETSc or hypre, to AMGCL, or to GPU solvers would matter most at
field scale, which is exactly where MATLAB's direct solve is weakest. Of all the
proposals here, this is the clearest route to PyMRST being **faster** than MRST
rather than merely as accurate.

**A curated benchmark suite with published references.** PyMRST already runs
SPE1, SPE3, SPE9, SPE10, Egg, and Norne, but as scattered examples rather than as
a suite with documented tolerances and expected results. Packaging them, and
adding the 11th SPE Comparative Solution Project for CO2 storage, which appears
nowhere in the tree, would give the port a citable standard to report against and
would connect it to current CO2 sequestration work.

**Out-of-core and streaming state handling.** States accumulate in memory, and no
chunked or memory-mapped path exists. SPE10 alone carries 1.1 million cells.
Streaming results to disk as they are produced, and loading them lazily, would
remove a ceiling that MATLAB also struggles with.

**A documented surrogate-component contract.** The Victory work in
`compositional/ML_models` already replaces the flash with a trained model behind
`victory_backend.py`. Generalising that into a stated contract, so any component
such as flash, relative permeability, or PVT can be swapped for a learned
surrogate with a physics fallback and an accuracy guard, would make PyMRST a
platform for hybrid physics and machine-learning research. MRST offers no
equivalent.

## On moving the arrays to the GPU

Replacing NumPy arrays with PyTorch tensors is an obvious-looking route to GPU
execution, so it is worth recording why the measurements do not support it. The
short answer is that the simulator is not limited by array arithmetic, and a GPU
accelerates array arithmetic.

A torch path already exists. `pymrst/backends/array_backend.py` provides an
`ArrayBackend` with torch support, and `run_backend_blackoil_tutorial_spe1` runs
SPE1 through it. Timed against the NumPy path over eight SPE1 steps:

| Path | Time |
|---|---|
| NumPy, `scipy.sparse`, forward-mode AD | 3.946 s |
| torch, autograd, dense solve | 7.798 s |

Profiling explains the gap. In the torch path, `run_backward` accounts for 7.406
of 7.798 seconds, or 94%, while the linear solve, `linalg_solve`, takes 0.153 s.
In the NumPy path at 2592 cells, the sparse solve `_superlu.gssv` takes 0.133 s,
close to 3% of the step, and the largest single cost is constructing
`scipy.sparse` matrices: 402337 of them across eight SPE1 steps, with
`_compressed.__init__`, `get_index_dtype`, `prune`, and `check_format` together
taking between 33% and 56% of wall time depending on problem size.

Both paths are therefore limited by derivative bookkeeping rather than by
floating-point work: sparse Jacobian assembly in one, autograd graph traversal in
the other. Neither improves on a GPU, and moving small arrays across a bus makes
per-operation dispatch worse.

A structural mismatch reinforces the measurement. PyMRST's AD is forward mode
with sparse Jacobian blocks, whereas torch autograd is reverse mode with weak
sparse support. `torch.sparse` offers no sparse direct solver and no algebraic
multigrid or CPR preconditioner, which is what reservoir simulation needs at
scale. The existing torch backend avoids the problem by solving dense, which
works at 300 cells and does not at 100000.

Precision settles the question independently of any profiling. The package pins
`float64` in 204 places, because material balance and ill-conditioned Jacobians
need double precision, and MRST parity is stated in double throughout. Apple's
MPS backend does not support `float64` at all, so on Apple hardware no GPU path
for the solver exists. Consumer NVIDIA cards run `float64` at roughly one
sixty-fourth of their `float32` rate, which puts meaningful double-precision GPU
work on A100 or H100 class hardware. Dropping to `float32` to obtain GPU speed
does not degrade the parity work, it ends it.

Gradients deserve a similar caution. Differentiating a Newton solve by recording
its iterations is not the same as differentiating the solution, and the tape
version is both wrong and expensive. What optimisation needs is the adjoint of
the converged system, which is implicit differentiation, and PyMRST already
carries that machinery in `modules/ad_mechanics/adjoint` and
`modules/optimization`. Autograd over the simulator core therefore buys less than
it first appears.

If GPU work does go ahead, CuPy is the better vehicle than torch. It mirrors the
NumPy API, so it is a far closer drop-in for 784 files that import NumPy, and it
wraps cuSPARSE and cuSOLVER, so it covers the sparse solve that `torch.sparse`
does not. Three `import cupy` sites already exist in the tree. Torch wins only
where autograd itself is the requirement, which is the surrogate work rather than
the core.

Three conclusions follow, and they redirect rather than abandon the effort:

1. **Assemble the Jacobian in place.** Preallocating the sparsity pattern once
   and filling values, which is standard practice in production simulators,
   attacks the measured bottleneck directly and needs no GPU.
2. **Put the GPU behind the linear solve**, through cupy, AmgX, or PETSc rather
   than torch. That is the modern linear-solver item above, and it is the right
   home for GPU work.
3. **Keep torch for the surrogates.** Learned components such as the Victory
   flash are dense and batched, which suits a GPU. The simulator core is neither.

One limit on this evidence: we measured up to 2592 cells. The solve fraction
grows with problem size, so at 100000 cells and beyond the balance shifts toward
conclusion 2 and the case for GPU acceleration of the linear solve becomes
strong. Treat the 3% figure as a small-case result. Profiling SPE10 at 1.1
million cells would settle the field-scale question and is worth doing before any
solver investment.

The distinction is worth stating plainly, because the intuitive answer runs the
other way. Reservoir simulation is usually described as linear-solve bound, and
counting call sites appears to confirm it: the package contains 117 `spsolve`,
68 `gmres`, and 11 `splu` references. Call sites measure surface area, not time.
At the sizes PyMRST's own test suite runs, the profiler puts the solve at about
3% and derivative bookkeeping above 30%, so optimising the solver first would
buy a few percent while the larger cost went untouched. Whether the usual
description holds here is a question for a profile of SPE10, not for intuition.

Finally, sequence matters more than any individual choice. A backend or dtype
change perturbs results through different reduction orders and accumulation, so
parity tests begin failing for reasons unrelated to correctness and there is no
way to separate real regressions from numerical noise. The agreement testing in
Proposal 1 and the provenance in Proposal 4 are what make later optimisation
safe. Validate first, then optimise.

## Suggested sequence

1. **Proposal 1**, cross-formulation agreement testing. It is cheapest relative
   to value, needs no MATLAB, reuses `compare_natural_overall`, and it is the one
   capability that lets PyMRST find defects MRST cannot.
2. **Proposals 8 then 7**, structured logging followed by Newton forensics, in
   that order because the second is far easier once the first exists. Together
   they are the smallest change with an immediate daily return, since every
   failed run now costs a manual investigation that the solver could
   answer directly.
3. **Proposal 6**, the deck coverage report, which is nearly self-contained and
   closes a known route to silent corruption.
4. **Proposal 3**, invariants, together with the shared debug switch, since
   Proposals 2 and 4 both depend on that switch.
5. **Proposal 4**, provenance, which also settles the `tol_well` question and
   pairs naturally with the logging from step 2.
6. **Proposal 5**, property-based tests for the flash and grid kernels.
7. **Proposal 2**, dimensioned quantities, last among the correctness work
   because it is the largest change, though the index-base tagging in point 1 of
   that section is worth pulling forward and doing early.

Among the capability gaps, the OPM Flow reference comes first. It is the one that
strengthens the correctness argument rather than only extending reach, it
completes Proposal 1 by supplying an independent second opinion, and it starts
from code that already exists in an example. Result export and the ensemble
driver follow, since both are self-contained and neither disturbs existing
numerics. The linear-solver ecosystem is the item to reach for when performance
rather than correctness becomes the constraint. The differentiable backend and
browser execution are roadmap entries rather than tasks.

The capability gaps divide by cost in a way worth keeping in mind:

| Effort | Items |
|---|---|
| days | reproducibility manifest, typing and lint gate, progress reporting |
| weeks | OPM Flow reference, result export, ensemble driver, benchmark suite |
| months | in-place Jacobian assembly, linear-solver ecosystem, out-of-core state, surrogate contract |
| roadmap | differentiable backend, browser execution |

In-place Jacobian assembly appears in that table without a numbered proposal of
its own because it is a performance change rather than a capability. It earns a
place because the profiling under "On moving the arrays to the GPU" puts it at
the top of the measured cost.

## Relationship to the existing test-coverage gap

These proposals do not replace the need for ordinary tests on the 51 untested
modules longer than 60 lines. They complement it. Ordinary tests fix known
modules one at a time, whereas agreement testing, invariants, and property tests
apply to every case a user runs, including cases nobody thought to write a test
for. The MATLAB file bridge argues for doing both: writing its first tests found
three defects in a single module, which suggests the untested set holds more.
