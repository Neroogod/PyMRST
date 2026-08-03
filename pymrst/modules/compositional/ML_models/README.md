# Optional Victory compositional ML experiment

This directory contains Victory's seven-fold neural flash ensembles and a
standalone comparison workflow. The ML code is experimental and opt-in: it is
not used by PyMRST's compositional simulator, does not replace the standard
Peng--Robinson flash, and is not imported when the compositional package starts.

The supplied models were trained for the isothermal C1/C3/C14 system at 353 K.
They take `[z1, z3, z14, p, Gas, 2phase]` and predict
`[x1, x3, x14, y1, y3, y14, L]`. Since `Gas` and `2phase` are inputs, these
models are conditional flash regressors rather than independent stability
classifiers.

## Side-by-side comparison

Run the seven beta-2.2 folds as an ensemble and compare their mean/spread with
PyMRST and the bundled reference:

```bash
python3 -m pymrst.modules.compositional.ML_models.compare_with_pymrst \
  --model beta2.2 \
  --rows 10000 \
  --seed 42 \
  --output results/victory_beta2.2_10000
```

Progress is displayed by default. The PyMRST stage advances once per 1,000-state
batch and shows completed rows, elapsed time, and ETA. Adjust the update
frequency with `--batch-size 500`, use `--batch-size 10000` for a faster
single-batch 10,000-row benchmark, or use `--no-progress` for CI/log files.

The command also writes `inputs.mat`. Run canonical MRST on exactly those
states from the PyMRST repository root:

```bash
/Applications/MATLAB_R2021b.app/bin/matlab -batch \
  "addpath('$PWD/scripts/matlab'); run_mrst_victory_flash_reference( ...
  '$PWD/results/victory_beta2.2_10000/inputs.mat', ...
  '$PWD/results/victory_beta2.2_10000/mrst_results.mat')"
```

Then regenerate the report with MRST included:

```bash
python3 -m pymrst.modules.compositional.ML_models.compare_with_pymrst \
  --model beta2.2 \
  --rows 10000 \
  --seed 42 \
  --mrst-results results/victory_beta2.2_10000/mrst_results.mat \
  --output results/victory_beta2.2_10000
```

Generated artifacts include `summary.json`, `per_output_metrics.csv`,
`phase_metrics.csv`, `fold_metrics.csv`, `worst_cases.csv`, `predictions.csv`, and
`comparison.md`. The input digest prevents results from a different sample
being compared accidentally.

See `ML_compositional_implementation.md` at the repository root for the model
contract, limitations, metrics, and acceptance plan.
