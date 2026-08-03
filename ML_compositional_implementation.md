# Victory ML compositional comparison implementation

## Research Studio qualification conclusion

The completed 10,000-state PyMRST/MRST/Victory run is now represented by
`docs/research_studio/qualification/victory_surrogate_evidence_graph.json`.
The graph preserves the important negative result: PyMRST/MRST physical parity
passes and Victory inference is approximately 20.16 times faster, but Victory
fails six of seven declared output-accuracy gates and its physical-safety gate.
It is therefore not accepted as a production flash or stability classifier.
This does not alter `eos.flash()` or the default compositional backend.

## Purpose

This work will add an **experimental, opt-in comparison workflow** for the
Victory three-component neural-network flash models. Victory will be compared
against both PyMRST and MATLAB MRST 2024a. It will not replace, modify, or
silently intercept either simulator's existing Peng--Robinson stability test,
two-phase flash, automatic differentiation, or compositional simulator.

The first objective is to answer, quantitatively:

1. How closely does the Victory ensemble reproduce its bundled reference
   flash results?
2. How closely do PyMRST and MATLAB MRST reproduce those same reference results
   when all methods use the same C1/C3/C14 fluid and thermodynamic conditions?
3. How closely do Victory, PyMRST, and MATLAB MRST agree pairwise?
4. Does a PyMRST/Victory discrepancy come from the neural model, from PyMRST's
   port, or from a difference already visible against canonical MRST?
5. Where do the largest discrepancies occur: phase boundaries, liquid
   fraction, liquid composition, vapor composition, or physical residuals?
6. What is the inference/flash cost of each method for identical batches?

The result will be a reproducible benchmark and report, not a production flash
backend.

## Implementation status

The non-invasive comparison workflow is now implemented:

- `victory_flash.py` reconstructs the exact seven-fold ensemble, fixed scaling,
  C1/C3/C14 mixture, domain checks, and reference-data loader without requiring
  the Streamlit UI;
- `compare_with_pymrst.py` runs Victory and PyMRST, exports/imports digested MAT
  fixtures, imports optional MRST results, and writes aggregate,
  phase-stratified, per-fold, uncertainty, physical-residual, and worst-case
  reports; its default six-stage console display includes a row-level PyMRST
  progress bar with elapsed time and ETA;
- `scripts/matlab/run_mrst_victory_flash_reference.m` runs canonical MRST 2024a
  `standaloneFlash` on the exact exported rows;
- `tests/compositional/ml` covers the checkpoint contract, fixed-scaler batch
  invariance, mixture constants, phase-aware masks, fixture digest, importer,
  and physical metrics;
- no production EOS dispatch or simulator code has been modified.

A live 1,000-row acceptance run has been completed with the installed MATLAB
and MRST 2024a. PyMRST and MRST selected the same phase for most rows and agreed
at approximately `10^-6` MAE on representative phase compositions wherever
the compared phase was present in both. Their aggregate `L` difference was
larger because a small subset received different single/two-phase or gas/liquid
classifications; the generated confusion matrices and worst-case rows preserve
that distinction for diagnosis.

The active PyMRST PR flash has subsequently been upgraded to use MRST's
vectorized closed-form positive cubic roots, a vectorized active-row
Rachford--Rice solver, the fugacity-based MRST stability test, and active-row
SSI convergence. On the completed 10,000-state comparison:

- PyMRST and MRST phase confusion matrices are identical;
- PyMRST-to-MRST `L` MAE is approximately `3.9e-6`;
- component-composition MAEs are approximately `2.4e-7` to `2.9e-6`;
- PyMRST material-balance maximum error is below `1.0e-10`;
- one-batch PyMRST flash time is approximately `0.33 s`, compared with about
  `0.91 s` for the recorded MRST batch on this machine;
- the default 1,000-row progress batching trades some throughput for ten useful
  progress updates during a 10,000-row run.

## Recovered model contract

The files in `pymrst/modules/compositional/ML_models/Web_app` provide the
inference contract that is absent from the raw `.pt` checkpoints.

### Training system

- Components and order: methane / propane / tetradecane (`C1/C3/C14`).
- Temperature: fixed at 353 K.
- Pressure in the bundled reference data: approximately 0.101 MPa to
  34.474 MPa.
- EOS used by the web-app physics checks: Peng--Robinson.
- Critical pressures, Pa: `[4.604e6, 4.249e6, 1.620e6]`.
- Critical temperatures, K: `[190.58, 369.82, 723.00]`.
- Acentric factors: `[0.0108, 0.1520, 0.8000]`.
- Binary interaction coefficients:
  - C1--C3: `0.00683`
  - C1--C14: `0.01230`
  - C3--C14: `0.00082`

### Network

- Input width: 6.
- Hidden network: four fully connected layers of width 256.
- Hidden activation: ReLU.
- Output width: 7.
- Output activation: sigmoid.
- Ensemble: seven cross-validation fold models.
- Selected initial scheme: Modified learning-rate annealing, beta 2.2.
- The seven beta-2.2 checkpoints in `ML_models/` are byte-identical to the
  corresponding checkpoints under `Web_app/models/beta2.2/`.

### Inputs and outputs

The raw input ordering is:

```text
[z1, z3, z14, pressure, Gas, 2phase]
```

Only pressure is MinMax-scaled. The scaling range is fitted once from
`Web_app/reference/test_data.csv`; uploaded/evaluation batches must never be
used to refit the scaler.

The unscaled output ordering is:

```text
[x1, x3, x14, y1, y3, y14, L]
```

where `L` is liquid molar fraction. Outputs are inverse-transformed using the
fixed output MinMax scaler fitted from the same reference file.

### Important experimental limitation

`Gas` and `2phase` are required model inputs. In the bundled workflow they are
derived from the known reference `L`:

```text
L = 0       -> Gas=1, 2phase=0
L = 1       -> Gas=0, 2phase=0
0 < L < 1   -> Gas=0, 2phase=1
```

Consequently, this model is **not a stability classifier**. The initial
comparison will supply the same reference phase flags to Victory while PyMRST
performs its normal flash independently. Results must clearly distinguish
phase classification from conditional flash regression. No claim that Victory
replaces stability analysis will be made from this experiment.

## Proposed layout

The comparison implementation will follow the compositional module layout and
keep reusable inference code separate from the web UI:

```text
pymrst/modules/compositional/ML_models/
├── README.md
├── model_runtest_beta2.2_fold[1-7].pt
├── victory_flash.py
├── compare_with_pymrst.py
├── Web_app/
│   ├── inference.py
│   ├── reference/test_data.csv
│   └── ...
└── results/                         # generated and git-ignored where appropriate

tests/compositional/ml/
├── test_victory_model_contract.py
├── test_victory_inference.py
└── test_victory_pymrst_comparison.py

scripts/matlab/
├── run_mrst_victory_flash_reference.m
└── ref_victory_mrst_flash.mat          # generated fixture, when retained
```

`victory_flash.py` will expose a small programmatic API without importing
Streamlit. `compare_with_pymrst.py` will be the command-line experiment.

## Implementation steps

### 1. Isolate a reusable Victory inference adapter

Implement a class such as:

```python
VictoryFlashEnsemble(
    model_scheme="beta2.2",
    device="cpu",
    reference_csv=None,
)
```

Responsibilities:

- load the seven checkpoints with `map_location` and safe state-dict loading;
- reconstruct the exact ReLU/sigmoid `FlashNet` architecture;
- load the fixed pressure and output scalers;
- validate columns, finite values, normalized feed composition, pressure
  bounds, temperature, and phase flags;
- run vectorized inference;
- inverse-transform each fold before calculating ensemble mean and standard
  deviation;
- return a typed result containing `x`, `y`, `L`, per-output uncertainty, and
  timing information;
- avoid importing Streamlit, Plotly, or other UI-only dependencies.

The existing web app remains usable. Shared inference logic can later be
factored into this adapter if doing so does not change web-app predictions.

### 2. Define the matching PyMRST fluid

Create an explicitly named C1/C3/C14 `Mixture` using the recovered critical
properties, acentric factors, and binary interactions. Add documented critical
volumes and molar masses from an authoritative component table already used by
PyMRST/MRST, or isolate those values because the value-level PR flash itself
does not consume them.

The comparison must assert:

- component count and order match the trained model;
- temperature equals 353 K within a strict configurable tolerance;
- pressures use Pa;
- feed compositions are nonnegative and row-normalized;
- the PyMRST EOS type matches the reference PR alpha correlation.

This experiment must not use PyMRST's existing C1/n-C4/n-C10
`simple_mixture()`, since that is a different fluid.

### 3. Build a common evaluation dataset

Initially use a deterministic sample from
`Web_app/reference/test_data.csv`. The CLI will accept:

- `--rows` for a bounded sample or all 150,000 rows;
- `--seed` for deterministic random sampling;
- `--phase all|gas|liquid|two-phase`;
- `--pressure-min` and `--pressure-max` filters;
- an optional external CSV following the same schema.

For every selected row retain:

```text
p, z, reference L/x/y, reference ZL/ZV, reference phase label
```

The train/test provenance of the bundled file must be reported. Because its
name is `test_data.csv`, the first experiment treats it as the supplied held-out
reference, but this should be confirmed against Victory's training workflow.

### 4. Export the common input batch for MATLAB MRST

The Python comparison CLI will write a MATLAB-readable input fixture containing
only the selected thermodynamic states and immutable experiment metadata:

```text
p                 pressure in Pa, N-by-1
T                 temperature in K, N-by-1
z                 overall mole fractions, N-by-3 in C1/C3/C14 order
reference_L        bundled reference liquid fraction
reference_x/y      bundled reference phase compositions
sample_indices     original reference CSV row numbers
mixture constants  Tc, Pc, Vc, acentric factor, molar mass and BIC
```

Use MAT v7/v7.3 or CSV plus a JSON manifest according to dataset size. The
exporter will record a SHA-256 digest so Python and MATLAB results can prove
that they used exactly the same rows and ordering.

### 5. Run MATLAB MRST independently

Add `scripts/matlab/run_mrst_victory_flash_reference.m`. The script will:

1. add the local MRST root and activate at least `ad-core` and
   `compositional`;
2. load the Python-generated input batch;
3. construct a `CompositionalMixture` explicitly in C1/C3/C14 order using the
   exact recovered constants rather than relying on a possibly different
   component database revision;
4. set the same symmetric binary-interaction matrix;
5. create `EquationOfStateModel([], mixture, 'Peng-Robinson')`;
6. call canonical MRST `standaloneFlash(p, T, z, eos)`;
7. save `L`, `x`, `y`, `Z_L`, `Z_V`, densities when available, convergence
   reports, elapsed time, MRST version/path information, and input digest;
8. avoid supplying Victory/reference phase flags to MRST;
9. run noninteractively through MATLAB `-batch`.

The script will accept explicit input/output paths so it does not depend on the
current MATLAB working directory. It must fail clearly if the expected MRST
2024a tree or required modules are unavailable.

MRST fixture command (exact arguments will be finalized during implementation):

```bash
/Applications/MATLAB_R2021b.app/bin/matlab -batch \
  "run_mrst_victory_flash_reference('inputs.mat', 'mrst_results.mat')"
```

MATLAB version and MRST release are separate concerns: the experiment will
record both, because the installed executable may be MATLAB R2021b while the
source tree is MRST 2024a.

### 6. Run Victory and PyMRST independently

For each identical `(p, T=353 K, z)` batch:

1. Run Victory with the reference-derived `Gas` and `2phase` inputs.
2. Run the standard PyMRST `eos.flash()` without injecting reference phase
   flags or Victory predictions.
3. Import the previously exported MATLAB MRST result and verify its input
   digest, row count, component order, units, and finite outputs.
4. Convert PyMRST vapor fraction to the common liquid-fraction convention:
   `L = 1 - V`.
5. Store PyMRST `x`, `y`, `L`, `Z_L`, `Z_V`, phase flag, iteration count, and
   elapsed time.
6. Store MRST's corresponding outputs without postprocessing other than shape
   normalization and the documented phase convention.

Each method will be timed after a warm-up run. CPU thread count, PyTorch device,
batch size, row count, Python/MATLAB/MRST versions, and software versions will
be recorded. MATLAB startup and fixture I/O time will be reported separately
from `standaloneFlash` execution. Timing results will be labeled carefully
because batched neural inference and iterative NumPy/MATLAB flash do not have
identical execution characteristics.

### 7. Use phase-aware comparisons

Absent-phase compositions are undefined and must not dominate aggregate error.
Metrics will therefore be evaluated as follows:

- liquid rows: compare `x` and `L`; ignore `y`;
- gas rows: compare `y` and `L`; ignore `x`;
- two-phase rows: compare all `x`, `y`, and `L` outputs;
- compare phase classification for PyMRST and MRST versus the supplied
  reference, because Victory receives the reference phase class as an input.

The report will include all six pairings:

```text
Victory vs reference
PyMRST  vs reference
MRST    vs reference
Victory vs PyMRST
Victory vs MRST
PyMRST  vs MRST
```

MRST is the canonical implementation baseline for judging port parity, while
the bundled reference remains the likely training-data generator baseline.
Where MRST and the bundled reference disagree, the report will not assume one
is correct: it will expose EOS options, constants, stability decisions, and
convergence status for diagnosis.

### 8. Quantify numerical errors

For every applicable scalar/vector output report:

- mean error (bias);
- mean absolute error;
- root-mean-square error;
- median absolute error;
- 95th and 99th percentile absolute errors;
- maximum absolute error and the state at which it occurs;
- relative error where the reference magnitude is safely nonzero;
- coefficient of determination for phase-aware aggregate outputs.

Specific quantities include:

- liquid fraction `L`;
- each liquid composition `x1`, `x3`, `x14`;
- each vapor composition `y1`, `y3`, `y14`;
- `Z_L` and `Z_V` comparison for PyMRST and MRST versus the reference and each
  other;
- phase-label confusion matrices for PyMRST and MRST versus reference;
- Victory ensemble standard deviation and its correlation with actual error.

Metrics will also be stratified by:

- reference phase;
- pressure bins;
- liquid-fraction bins;
- distance from the phase boundary;
- ensemble-uncertainty bins.

In addition, report a dedicated PyMRST-to-MRST parity table with engineering
tolerances and counts outside tolerance. This separates Python-port parity
from Victory model accuracy.

### 9. Quantify physical consistency

For Victory, PyMRST, and MRST independently calculate:

#### Composition closure

```text
abs(sum(x) - 1)
abs(sum(y) - 1)
```

#### Interphase material balance

```text
z_hat = L*x + (1-L)*y
residual = z_hat - z
```

For single-phase rows use the present-phase composition against `z`.

#### Fugacity equilibrium

For two-phase rows calculate the C1/C3/C14 Peng--Robinson residual using the
same component constants and binary interactions. Report both log-fugacity
difference and the web app's existing weighted `FugRMSE` for continuity.

#### Bounds and finite checks

Report counts of:

- nonfinite outputs;
- negative or greater-than-one compositions;
- `L` outside `[0, 1]`;
- composition-closure failures;
- material-balance failures;
- large fugacity residuals.

Raw Victory outputs should be measured before any clipping or renormalization.
A second, explicitly labeled projected result may be reported, but projection
must never hide the raw model error.

### 10. Produce reproducible artifacts

The CLI will write a chosen output directory containing:

```text
summary.json              machine-readable configuration and metrics
per_output_metrics.csv    detailed scalar metrics
phase_metrics.csv         phase-stratified metrics
worst_cases.csv           states with the largest discrepancies
predictions.csv           optional row-level reference/ML/PyMRST/MRST values
comparison.md             concise engineering report
inputs.mat                exact common input passed to MATLAB
mrst_results.mat          raw canonical MRST result
```

Optional plots, generated only when plotting dependencies are installed:

- predicted-versus-reference parity plots;
- absolute error versus pressure;
- absolute error versus reference `L`;
- ternary maps of error;
- ensemble uncertainty versus actual error;
- phase confusion matrices for PyMRST and MRST;
- PyMRST-versus-MRST parity plots.

The default test suite will not require a graphical renderer.

### 11. Add tests

Tests will cover:

1. Checkpoint architecture and all seven-fold files.
2. Equality of root beta-2.2 and web-app beta-2.2 checkpoints.
3. Exact input/output column order.
4. Fixed-scaler behavior: inference on one row and a larger batch must give the
   same prediction for that row.
5. Deterministic ensemble predictions on CPU.
6. Known reference rows with stored prediction tolerances.
7. C1/C3/C14 mixture properties and BIC ordering.
8. Metric behavior for gas, liquid, and two-phase rows.
9. Absent-phase compositions excluded from phase-aware errors.
10. Material-balance and closure calculations.
11. Clear rejection of unsupported temperature/component configurations.
12. MRST MAT-fixture schema, digest, shapes, component ordering, and units.
13. Phase-aware MRST/reference and PyMRST/MRST metric behavior.
14. A small end-to-end Python comparison smoke test.

MATLAB-dependent validation will be separated from the normal fast pytest
suite. A committed small MRST fixture can test the importer everywhere; live
fixture regeneration will be an explicit local acceptance test when MATLAB and
MRST are available.

Torch and scikit-learn are optional dependencies. ML-specific tests will skip
with an actionable message when they are unavailable; importing the main
compositional package must continue to work without them.

### 12. Document and run the experiment

Planned command:

```bash
python3 -m pymrst.modules.compositional.ML_models.compare_with_pymrst \
  --model beta2.2 \
  --rows 10000 \
  --seed 42 \
  --mrst-results results/victory_beta2.2_10000/mrst_results.mat \
  --output results/victory_beta2.2_10000
```

The CLI will also support an orchestration option that writes the input fixture
and prints the exact MATLAB command. Live MATLAB execution will remain explicit
instead of occurring unexpectedly during a normal Python test.

An all-row acceptance run can then use:

```bash
python3 -m pymrst.modules.compositional.ML_models.compare_with_pymrst \
  --model beta2.2 \
  --rows all \
  --output results/victory_beta2.2_full
```

The report must state that Victory was supplied with reference phase flags and
therefore its phase-classification accuracy was not measured. It must also
distinguish canonical MRST results from the bundled reference dataset rather
than referring to both generically as ground truth.

## Initial exploratory baseline

A preliminary read-only run of the unmodified web-app inference code produced
the following beta-2.2 ensemble results:

- 10,000 randomly selected bundled reference rows:
  - phase-aware R2: approximately `0.97423`;
  - composition-closure RMSE: approximately `0.00215`;
  - interphase/material-balance RMSE: approximately `0.01201`;
  - CPU ensemble inference time after loading: approximately `0.9 s` on the
    development machine.
- On a separate 20,000-row sample, 4,800 rows were two-phase. Their approximate
  mean absolute errors were:
  - `L`: `0.08131`;
  - `x1`: `0.06251`;
  - `x3`: `0.07146`;
  - `x14`: `0.09893`;
  - `y1`: `0.02843`;
  - `y3`: `0.02836`;
  - `y14`: `0.000282`.

These numbers are exploratory only. They used the web app's current metrics
and reference-derived phase flags. The implemented comparison will regenerate
them reproducibly, add PyMRST and MATLAB MRST results, stratify errors
correctly, and record configuration and version information.

## Non-goals for this first stage

- Do not replace `eos.flash()`.
- Do not change `backend_flash()` dispatch.
- Do not use Victory inside reservoir timesteps.
- Do not modify MATLAB MRST source files.
- Do not alter compositional AD derivatives.
- Do not claim stability prediction from a model that receives phase flags.
- Do not extrapolate to temperatures other than 353 K.
- Do not apply the model to C1/C4/C10 or arbitrary mixtures.
- Do not retrain or fine-tune the checkpoints.
- Do not silently clip or normalize outputs before reporting raw error.

## Completion criteria

This comparison stage is complete when:

- the Victory adapter reproduces the existing web-app beta-2.2 predictions;
- PyMRST runs the same C1/C3/C14 states with matched PR parameters;
- MATLAB MRST `standaloneFlash` runs the same digested input rows with matched
  PR parameters;
- all six comparison pairings are reported with phase-aware metrics;
- PyMRST-to-MRST parity is reported separately from Victory accuracy;
- physical residuals and worst-case states are saved;
- timing methodology and experimental limitations are explicit;
- the small deterministic test suite passes;
- a 10,000-row report and an optional full-reference report can be reproduced
  from documented Python and MATLAB commands;
- no default PyMRST runtime path has changed.

## Possible later experiments

Only after this baseline is understood should later work consider:

1. comparing all available Victory training schemes;
2. using Victory predictions as optional PR-flash initial guesses;
3. generating an independent PyMRST/MATLAB validation grid;
4. training a separate stability classifier without phase flags;
5. adding temperature as an input;
6. training a component-aware model for mixtures beyond C1/C3/C14;
7. testing the alternative inside a reservoir simulation behind an explicit
   experimental option.
