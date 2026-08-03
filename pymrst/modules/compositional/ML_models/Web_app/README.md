# Flash Calculation Predictor — Web App

A self-contained Streamlit app that serves the trained flash-calculation
neural-network ensembles. Upload feed data, pick a model, and get equilibrium
predictions (`x`, `y`, `L`) plus physics-consistency / accuracy metrics with
interactive ternary and parity plots.

> This folder is **self-contained** — it bundles its own copies of every model
> checkpoint, the `thermo` module, and a reference dataset. It does not depend
> on, or modify, any of the sibling inference folders.

## What it does

- **Model picker** — choose among the trained schemes: DOE optimum (Run 276),
  ReLoBRaLo, and Modified LR annealing for beta = 1, 1.2, 1.5, 2, 2.2, 2.5, 3.
- **7-fold ensemble** — predictions are the mean of each scheme's 7
  cross-validation fold models; the per-fold spread is reported as uncertainty
  (`*_std` columns).
- **Conditional metrics** (the key behavior you asked for):
  - **No ground truth uploaded** → reports the three *physics residuals*:
    `FugRMSE`, `CompRMSE`, `InterRMSE`. The fugacity residual is gated/weighted
    on the model's **predicted** `L`/`x`/`y`.
  - **Ground truth uploaded** (true `x1,x3,x14,y1,y3,y14`) → additionally
    reports phase-aware **R²**, and the fugacity residual is gated/weighted on
    the **true** labels (the original benchmark behavior).
- **Plots** — ternary diagrams of predicted liquid/vapor compositions; a 7-panel
  predicted-vs-true parity plot when ground truth is present.
- **Download** — predictions table as CSV.

## Input format

| Column group | Columns | Required? |
|---|---|---|
| Feed | `z1, z3, z14, p` | **Yes** |
| Phase info | `L` (0=gas, 1=liquid, fraction=two-phase) **or** `Gas` & `2phase` flags | **Yes** (one of) |
| Ground truth | `x1, x3, x14, y1, y3, y14` | Optional (enables R²) |

The app derives the `Gas`/`2phase` model inputs from `L` if the flags are not
given. Don't have a file? Use the in-app **"Generate a sample"** panel to build
one (with or without ground-truth columns) from the bundled reference set.

## Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Layout

```
Web_app/
├── app.py                 # Streamlit UI
├── inference.py           # Model loading, scalers, prediction, metrics
├── requirements.txt
├── README.md
├── thermo/zFact_n_logphi3.py
├── reference/test_data.csv   # used ONLY to fix input/output scaling
└── models/
    ├── doe276/    model_run276_fold[1-7].pt
    ├── relobralo/ model_runtest_fold[1-7].pt
    └── beta{1,1.2,1.5,2,2.2,2.5,3}/ model_runtest_beta{B}_fold[1-7].pt
```

## Notes on correctness

- **Fixed scaling.** Scalers are fit **once** on `reference/test_data.csv` and
  held constant — they are *not* refit on uploads. This is required for correct
  predictions on small/arbitrary inputs. MinMax scaling transfers across schemes
  because every model shares the same physical ranges (compositions in [0, 1],
  the same DOE pressure range, `L` in [0, 1]).
- All schemes use the identical `FlashNet` architecture (6→256×4→7, ReLU +
  sigmoid output), so a single input/output schema serves every model.
