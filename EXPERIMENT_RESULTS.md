# As-of Fixed-Holdout Forecasting Results


This run evaluates LSTM, Transformer, and PVG-iTransformer under one leakage-controlled protocol: causal minute imputation, lag-1 as-of monthly weather, chronological train/validation/test splitting, validation-selected checkpoints, and one direct forecast at a fixed test boundary.

## Data and leakage controls

- The daily series has 1,440 consecutive rows from 2006-12-17 through 2010-11-25.
- Minute gaps use only earlier original observations at `t-7`, `t-14`, `t-21`, and `t-28` days within the same month. If none exists, the most recent earlier raw minute is used; imputed values are never donors.
- Each original power variable has 25,979 missing minutes: 25,912 are repaired by past weekly donors and 67 by causal forward fill. Across all seven variables, the 181,853-row audit has zero cases with `latest_donor_time >= value_time`.
- The prediction target is the daily sum of 1,440 minute-level `Global_active_power` readings in kW. It is a summed-reading target, not kWh; dividing a daily target or prediction by 60 gives approximate kWh under one-minute sampling.
- SURESNES monthly weather uses a strict one-calendar-month lag: target month `m` receives weather from completed month `m-1`. The full mapping is saved in `weather_monthly_source_mapping.csv`.
- `RR` is monthly cumulative precipitation in millimetres and is not divided by 10. `NBJRR1`, `NBJRR5`, and `NBJRR10` are day counts.
- Feature and target scalers are fitted only through 2008-07-09, the end of the training segment.

## Split and scored intervals

| Segment | Dates | Rows |
|---|---|---:|
| Train | 2006-12-17 to 2008-07-09 | 571 |
| Validation | 2008-07-10 to 2009-07-09 | 365 |
| Test | 2009-07-10 to 2010-11-25 | 504 |

The 65% boundary is the end of the combined train-plus-validation development period. Training itself uses 571 rows, producing 392 windows for the 90-day task and 117 for the 365-day task.

| Horizon | Validation forecast scored | Fixed test input | Fixed test forecast scored |
|---:|---|---|---|
| 90 | 2008-07-10 to 2008-10-07 | 2009-04-11 to 2009-07-09 | 2009-07-10 to 2009-10-07 |
| 365 | 2008-07-10 to 2009-07-09 | 2009-04-11 to 2009-07-09 | 2009-07-10 to 2010-07-09 |

For each horizon, checkpoint selection uses exactly one fixed validation window and testing uses exactly one fixed test window. Each model emits the full horizon in one forward pass. No test observation is reused as a later input, and test rows after the stated forecast interval are not scored for that task.

## Five-seed neural-model results

Values are mean ± sample standard deviation over seeds 42, 43, 44, 45, and 46. All seeds evaluate the same single temporal origin, so the standard deviation describes optimization variability rather than variation across forecast periods.

MSE and MAE below are computed on the unconverted daily summed-reading target described above.

| Horizon | Model | MSE | MAE |
|---:|---|---:|---:|
| 90 | LSTM | 184773.74 ± 43647.35 | 348.12 ± 41.61 |
| 90 | Transformer | 175509.94 ± 31397.14 | 336.56 ± 28.69 |
| 90 | PVG-iTransformer | **115742.75 ± 5026.54** | **273.23 ± 5.86** |
| 365 | LSTM | 161242.08 ± 5814.77 | 319.02 ± 10.22 |
| 365 | Transformer | 154158.50 ± 9028.59 | 300.25 ± 8.19 |
| 365 | PVG-iTransformer | **149700.87 ± 3660.12** | **297.61 ± 2.93** |

PVG-iTransformer has the lowest mean MSE and MAE among the three required neural models at both horizons. This ranking is limited to the fixed intervals above.

## Validation and interpretation limits

- There is only one validation origin per horizon. Selecting among 30 epochs on one validation trajectory can make checkpoint choice sensitive to that episode.
- There is only one test origin per horizon. Five seeds do not provide evidence of performance across seasons or forecast origins.
- The lag-1 weather merge removes same-month future information but remains a monthly proxy. It assumes the previous month's aggregate is available at the next month's start and contains no daily weather forecast.
- The experiment compares the three course-required neural models under the same data split, training configuration, and evaluation protocol.

## Integrity checks and provenance

- 30 unique neural training runs: 3 models × 2 horizons × 5 seeds.
- 30 metric rows and 6 summary groups; all use `evaluation_protocol=fixed_holdout`.
- Every metric row has one validation window and one test window.
- Thirty validation-selected checkpoints; selected epochs range from 4 through 30.
- Six representative seed-42 prediction arrays have shapes `(1, 90)` or `(1, 365)` and begin at 2009-07-10.
- The saved three-model summary recomputes from the run-level metrics; the experiment-signature digest and its six bound data hashes also verify.
- Weather source mapping covers every target month from 2006-12 through 2010-11 with the immediately preceding source month.
- Minute-imputation audit rows with a non-causal donor: 0.

Hashes:

- forecasting script SHA-256: `4e8f8fdafd8ac66111b5754e2e6a07892e9480f1b9d1102179b465574275c21e`
- experiment signature: `3e1a56d478a376d7640a377241706690a166b7f3ca3cd8b82e03fe6af516ae5b`
- split manifest SHA-256: `60b0ed1b283b8f9582bcff1331eb4eee7adec0d6a2eb7182cc39df41e416765d`
- `metrics_runs.csv` SHA-256: `1be956d97f7ef7ad78e18eea69b9147aa0ed80642ccf2998451fe4133e374a3c`
- `metrics_summary.csv` SHA-256: `f08e959b72d867ddaf42a9ba1491bb265ecfd3ebddfb257ed44485db6dcb8fd3`

The experiment signature binds the forecasting script, preprocessing version, weather lag, evaluation protocol, source and processed data hashes, dependency environment, device, and model arguments. Detailed hashes and paths are stored in `data/processed_causal_asof_lag1_v2/split_manifest.json` and `results/run_metadata.json`.
