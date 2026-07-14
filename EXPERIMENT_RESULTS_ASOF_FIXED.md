# As-of Fixed-Holdout Forecasting Results

Verified run:

```text
/mnt/sdc/zoujunjie/mlearn_power_coursework/runs/fixed_holdout_asof_lag1_v2
```

Processed data:

```text
/mnt/sdc/zoujunjie/mlearn_power_coursework/data/processed_causal_asof_lag1_v2
```

This run evaluates LSTM, Transformer, and PVG-iTransformer under one leakage-controlled protocol: causal minute imputation, lag-1 as-of monthly weather, chronological train/validation/test splitting, validation-selected checkpoints, and one direct forecast at a fixed test boundary. No rolling-origin result is included.

The revised report corresponding to these exact artifacts is [`reports/mlearn_power_report_polished_updated.docx`](reports/mlearn_power_report_polished_updated.docx).

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

## Train-only seasonal baseline

As a deterministic diagnostic, a month-of-year mean baseline predicts each forecast date with the corresponding calendar-month mean computed only from the 571 training rows. It uses no validation or test target values and has no seed standard deviation.

| Horizon | Method | MSE | MAE |
|---:|---|---:|---:|
| 90 | Train-only month-of-year mean | **102190.07** | **261.05** |
| 90 | Best neural model (PVG-iTransformer) | 115742.75 | 273.23 |
| 365 | Train-only month-of-year mean | 153091.25 | 301.97 |
| 365 | Best neural model (PVG-iTransformer) | **149700.87** | **297.61** |

On the 90-day interval, the seasonal baseline is better than PVG-iTransformer by 13.26% in MSE and 4.67% in MAE; it therefore also beats the other two neural models. On the 365-day interval, PVG-iTransformer is better than the seasonal baseline by 2.21% in MSE and 1.44% in MAE.

The defensible conclusion is therefore narrower than “PVG is always superior”: PVG is the strongest of the three neural models in this run, does not beat the simple seasonal baseline on the single 90-day interval, and only modestly beats it on the single 365-day interval.

## Validation and interpretation limits

- There is only one validation origin per horizon. Selecting among 30 epochs on one validation trajectory can make checkpoint choice sensitive to that episode.
- There is only one test origin per horizon. Five seeds do not provide evidence of performance across seasons or forecast origins.
- The lag-1 weather merge removes same-month future information but remains a monthly proxy. It assumes the previous month's aggregate is available at the next month's start and contains no daily weather forecast.
- The seasonal baseline is a diagnostic computed under the same fixed test dates, not one of the three course-required trained models. It is kept separate from `metrics_runs.csv` and `metrics_summary.csv`.
- No ablation result from an earlier preprocessing or rolling-origin run is used here.

## Integrity checks and provenance

- 30 unique neural training runs: 3 models × 2 horizons × 5 seeds.
- 30 metric rows and 6 summary groups; all use `evaluation_protocol=fixed_holdout`.
- Every metric row has one validation window and one test window.
- Thirty validation-selected checkpoints; selected epochs range from 4 through 30.
- Representative seed-42 prediction arrays have shapes `(1, 90)` and `(1, 365)` and begin at 2009-07-10.
- Weather source mapping covers every target month from 2006-12 through 2010-11 with the immediately preceding source month.
- Minute-imputation audit rows with a non-causal donor: 0.

Hashes:

- forecasting script SHA-256: `262f0405845635d3468e66c279066944019588900e257acdc356d6c7680c4fd1`
- experiment signature: `8e78be2f38d9acfe19d9b02677fb4fe355bb1f6bc1175c82a10ec379655e7971`
- split manifest SHA-256: `4ac1216e646e064d0c44c16810bb0616149c4b48208fcd8368cf14a33ebcd188`
- `metrics_runs.csv` SHA-256: `1be956d97f7ef7ad78e18eea69b9147aa0ed80642ccf2998451fe4133e374a3c`
- `metrics_summary.csv` SHA-256: `f08e959b72d867ddaf42a9ba1491bb265ecfd3ebddfb257ed44485db6dcb8fd3`

The experiment signature binds the forecasting script, preprocessing version, weather lag, evaluation protocol, source and processed data hashes, dependency environment, device, and model arguments. Detailed hashes and paths are stored in `split_manifest.json` and `run_metadata.json`.

## Current artifact scope and history

In the current repository checkout, the top-level `results/` and `figures/` directories contain this formal `fixed_holdout_asof_lag1_v2` run. Older same-month-weather, non-causal, or rolling-origin artifacts occur only in earlier repository commits and must not be mixed with the current artifacts.
