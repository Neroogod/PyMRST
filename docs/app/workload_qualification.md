# PyMRST Studio workload qualification

Workload qualification answers a narrower question than numerical validation:
can this installed PyMRST build execute a known workload on this machine within
declared time, CPU, memory, and storage budgets without losing lifecycle control?
It does not replace MRST parity, analytical validation, or field-case acceptance.

## Safety boundary

Qualifications execute in dedicated child processes. The application owns their
process groups, consumes a versioned JSON-lines protocol, supports cancellation,
and marks abandoned queued/running records `interrupted` after restart. Only
workloads compiled into the registry can run; API payloads cannot select modules,
functions, executables, shell fragments, or arbitrary filesystem paths.

The worker receives a deterministic seed and writes only within
`.pymrst-app/workload-qualifications/<qualification-id>`. Requests are immutable.
Large arrays are disposable; the durable artifact is an atomic `evidence.json`
record whose SHA-256 is stored in SQLite and can be reverified later.

## Acceptance model

Every execution checks:

- maximum wall time;
- maximum process CPU time;
- maximum resident memory;
- maximum workload I/O volume;
- wall-time regression against the newest passing same-workload/same-tier run; and
- memory regression against that baseline.

By default the worker performs one untimed warmup followed by three measured
runs in the same isolated process. Imports, sparse-kernel initialization and
other cold-start effects are absorbed by the warmup. The stored `wall_seconds`
and `cpu_seconds` are medians of the measured runs, accompanied by raw samples,
median absolute deviation (MAD), minimum, maximum, P10 and P90. A run also fails
when wall-time MAD divided by the median exceeds the configured dispersion
limit. Only a previous passing qualification with at least three measured runs
may become a regression baseline, preventing legacy single-shot evidence from
silently becoming the reference.

All repetitions receive the same seed and reconstruct their workload state.
Their final result hashes must agree; disagreement is retained as the
`deterministic_across_measured_runs` correctness failure. The evidence also
records total qualification wall time and warmup duration, while absolute and
regression performance decisions use the measured median.

## Resource telemetry

The isolated worker samples itself every 250 ms from startup through aggregation.
When the application profile includes `psutil`, each bounded sample records:

- relative monotonic time;
- current resident memory;
- cumulative process CPU and interval CPU utilization;
- cumulative process read/write bytes;
- process thread count; and
- whether the sample belongs to startup, warmup, a numbered measured run, or
  aggregation.

The evidence retains at most 20,000 samples and explicitly marks truncation.
Its summary identifies the sampled RSS peak and its time/phase/run, peak CPU
utilization and time, I/O deltas, and maximum thread count. The authoritative
memory budget still uses the operating system's process high-water mark, because
a periodic sampler can miss a brief allocation spike; sampled RSS supplies the
location and shape of memory growth. On systems without `psutil`, a dependency-
free `resource` fallback retains CPU and RSS samples while marking I/O and thread
fields unavailable. Monotonic timestamps, sample presence and sampled/high-water
consistency are correctness invariants.

Samples are stored in an atomic `work/resource-telemetry.json` artifact rather
than in SQLite or routine API responses. Qualification metrics retain its
relative path, byte size and SHA-256. Evidence-bundle verification checks both
the primary evidence record and this subordinate artifact, detects missing or
escaped paths, and reports each checksum independently.

Reservoir workloads additionally publish named boolean correctness invariants.
Any failed invariant fails the qualification independently of performance, with
`CorrectnessInvariantFailed` retained separately from a budget regression.

A successfully executed workload can therefore have status `failed` when a
performance budget is exceeded. Its metrics and evidence are retained so this is
distinguishable from worker or orchestration failure.

The supported tiers are `smoke`, `engineering`, `field_scale`, and `endurance`.
Default limits are deliberately conservative and may be overridden explicitly for
a qualification. A single qualification runs at a time by default to avoid
benchmark contamination and competition with engineering runs.

## Initial workloads

`numerical-core` exercises vectorized face flux/divergence assembly and a SciPy
sparse solve. `state-io` exercises compressed, non-pickle numeric state round
trips. They qualify the infrastructure and common performance paths.

`spe1-blackoil` is the first production simulator owner. Its smoke tier runs one
report step and its engineering tier runs 30 through the normal
`GenericBlackOilModel` and `NonLinearSolver`. Evidence includes immutable deck
and final-state hashes, solver/assembly timings, Newton/cut/line-search counts,
CNV/MB/well/control residual maxima, pressure and saturation envelopes, final
BHP, and ten named correctness checks. The service resolves `BENCH_SPE1.DATA`;
the API cannot supply a path. When the dataset is absent the catalog reports the
owner unavailable with remediation instead of allowing a doomed submission.

`mandel-poromechanics` is the first physically distinct owner. It executes the
fully coupled single-phase flow/VEM mechanics kernel under a constant plate
load. The smoke tier uses 48 cells and 20 accepted coupled steps; engineering
uses 160 cells and 100 steps. Evidence records the analytical pressure error,
Mandel-Cryer centre-pressure overshoot, through-thickness pressure uniformity,
plate-force equilibrium, mechanical degrees of freedom and final coupled-state
checksum. Nine correctness invariants distinguish a physically wrong but
successfully executed solve from an accepted installation.

## Adding reservoir workloads

Norne, compositional, fractured/shale, and poromechanics workloads must be
added as explicit registry owners, never as arbitrary entrypoint strings. Each
owner must declare:

1. immutable dataset identity and checksum;
2. exact model, solver and schedule configuration;
3. deterministic seed policy where relevant;
4. required optional capabilities;
5. accepted correctness invariants in addition to resource budgets;
6. checkpoint/cancellation points;
7. expected evidence fields and result checksums; and
8. tier-specific sizing and release thresholds.

Field-scale release evidence must be collected on an otherwise idle machine and
retain its warm/cold-cache policy. The engine now supplies the minimum three
measured repetitions and robust statistics; release profiles may increase this
to five, seven, or nine measured runs when host variance warrants it.

## Release gates still required

- additional reservoir-family workload owners and dataset qualification;
- low-disk-space and forced-crash fault injection;
- checkpoint/resume performance evidence;
- signed native-runner evidence for every supported OS/architecture; and
- a release summary report combining environment, workload, recovery, and
  numerical-validation evidence.
