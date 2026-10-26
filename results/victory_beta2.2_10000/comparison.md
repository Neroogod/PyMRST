# Victory / PyMRST / MRST flash comparison

Rows: 10000

> Victory receives reference-derived Gas/2phase flags; this experiment does not measure Victory stability classification.

## Timings

- victory: 0.0557463 s
- pymrst: 1.12368 s
- mrst: 0.90942 s

## Physical consistency

- reference: material RMSE=5.48178e-07, log-fugacity RMSE=2.7906614013726587
- victory: material RMSE=0.00638565, log-fugacity RMSE=3.5220460715926767
- pymrst: material RMSE=2.15733e-11, log-fugacity RMSE=3.0151381546093607e-10
- mrst: material RMSE=2.15734e-11, log-fugacity RMSE=3.21945709648745e-05

See `per_output_metrics.csv`, `predictions.csv`, and `summary.json` for complete results.
