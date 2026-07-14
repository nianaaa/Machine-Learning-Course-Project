# Machine Learning Course Project: Household Power Forecasting

This repository implements the three models required by the 2026 professional-master machine-learning course assessment:

1. LSTM
2. Transformer
3. PVG-iTransformer

Each model is trained independently for direct 90-day and 365-day forecasting. The verified experiment uses five random seeds and a single fixed test origin. It does not use rolling-origin evaluation.

The complete verified results and their limitations are reported in [`EXPERIMENT_RESULTS_ASOF_FIXED.md`](EXPERIMENT_RESULTS_ASOF_FIXED.md). The matching revised course report is available as [`reports/mlearn_power_report_polished_updated.docx`](reports/mlearn_power_report_polished_updated.docx).

## Leakage-controlled preprocessing

### Minute-level power data

The UCI Individual Household Electric Power Consumption data are first placed on a complete one-minute index. Each missing value is filled strictly from earlier original observations:

- primary donors are the same minute at `t-7`, `t-14`, `t-21`, and `t-28` days, restricted to the same calendar month;
- if no weekly donor exists, the most recent earlier raw minute is used;
- an imputed value is never reused as a donor;
- every fill records its value time, latest donor time, method, and donor count, and the pipeline asserts `latest_donor_time < value_time`.

Each of the seven original power variables has 25,979 missing minutes. Past weekly donors repair 25,912 values per variable (99.742%); the remaining 67 use causal forward fill. The fallback lag is at most 38 minutes, and the saved audit has zero non-causal donor records.

After imputation, power and sub-metering variables are aggregated to daily sums, voltage and current to daily means, and `Sub_metering_remainder` is derived before daily aggregation. In particular, the forecast target is the daily sum of the 1,440 minute-level `Global_active_power` readings, whose source unit is kW. The stored target is therefore a sum of minute-level kW readings, not kWh; dividing it by 60 gives the approximate daily energy in kWh under one-minute sampling.

### As-of monthly weather

The weather variables come from the SURESNES Meteo-France station (`NUM_POSTE=92073001`):

- `RR`: monthly cumulative precipitation in millimetres;
- `NBJRR1`, `NBJRR5`, and `NBJRR10`: numbers of days in the month meeting the corresponding precipitation thresholds.

The source already stores `RR` in millimetres to one decimal place; no division by 10 is applied.

Weather is merged with a one-calendar-month lag. Every day in target month `m` uses only the completed monthly statistics from month `m-1`. The target/source-month mapping is saved explicitly, so a feature never uses the still-incomplete weather aggregate of its own month. This is an as-of proxy; it assumes the previous month's aggregate is available at the start of the current month and does not model publication delay or daily weather.

## Chronological split and fixed evaluation

The 1,440 consecutive daily rows are split chronologically. The 65% boundary separates the development period (train plus validation) from the test period; it is not a 65% pure-training split.

| Segment | Dates | Rows |
|---|---|---:|
| Train | 2006-12-17 to 2008-07-09 | 571 |
| Validation | 2008-07-10 to 2009-07-09 | 365 |
| Test | 2009-07-10 to 2010-11-25 | 504 |

Feature and target scalers are fitted only on the 571 training rows. Training uses overlapping historical windows whose outputs remain entirely inside the training segment: 392 windows for the 90-day task and 117 for the 365-day task.

Checkpoint selection and final testing each use one fixed-origin direct forecast:

| Task | Validation scoring interval | Test input interval | Test scoring interval | Test windows |
|---:|---|---|---|---:|
| 90 days | 2008-07-10 to 2008-10-07 | 2009-04-11 to 2009-07-09 | 2009-07-10 to 2009-10-07 | 1 |
| 365 days | 2008-07-10 to 2009-07-09 | 2009-04-11 to 2009-07-09 | 2009-07-10 to 2010-07-09 | 1 |

The model outputs the complete horizon in one forward pass. No later test observation is fed back as input, and test rows after the stated scoring interval are not included in the corresponding metric. The lowest fixed-validation MSE checkpoint is evaluated once at the test boundary.

This design prevents test-window reuse but has an important limitation: both checkpoint selection and testing represent only one temporal origin per horizon. The standard deviation over seeds measures optimization variability at that origin, not variability across different time periods.

## Verified run

The verified run is:

```text
/mnt/sdc/zoujunjie/mlearn_power_coursework/runs/fixed_holdout_asof_lag1_v2
```

Processed data are stored in:

```text
/mnt/sdc/zoujunjie/mlearn_power_coursework/data/processed_causal_asof_lag1_v2
```

The three neural models are compared in the result document. A deterministic train-only month-of-year mean baseline is also reported there as a diagnostic. It beats all three neural models on the single 90-day test interval, while PVG-iTransformer modestly beats it on the 365-day interval. Accordingly, the experiment does not support an unconditional claim that a neural model is superior for every horizon.

## Reproduce

The verified configuration used Python 3.10.20, PyTorch 2.4.1+cu118, and an NVIDIA GeForce RTX 3080. Run the following with an empty run directory; change `RUN` when retaining the verified artifacts above.

```bash
BASE=/mnt/sdc/zoujunjie
WORK=$BASE/mlearn_power_coursework
PROCESSED=$WORK/data/processed_causal_asof_lag1_v2
RUN=$WORK/runs/fixed_holdout_asof_lag1_v2
PYTHON=$BASE/miniconda3/envs/mlearn/bin/python

export HOME=$BASE
export TMPDIR=$BASE/tmp
export CONDA_PKGS_DIRS=$BASE/conda_pkgs
export PIP_CACHE_DIR=$BASE/.cache/pip
export XDG_CACHE_HOME=$BASE/.cache
export PYTHONNOUSERSITE=1

cd "$WORK"
"$PYTHON" scripts/run_forecasting.py \
  --base "$BASE" \
  --work-dir "$WORK" \
  --processed-dir "$PROCESSED" \
  --run-dir "$RUN" \
  --rebuild-data \
  --input-len 90 \
  --split-ratio 0.65 \
  --validation-days 365 \
  --pvg-time-pooling last \
  --models lstm transformer pvg_itransformer \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --seeds 42 43 44 45 46 \
  --horizons 90 365
```

Use `--resume` only to continue an interrupted run with the same experiment signature. Completed model/horizon/seed combinations are skipped.

## Outputs

Key processed artifacts:

- `data/processed_causal_asof_lag1_v2/daily_power.csv`
- `data/processed_causal_asof_lag1_v2/train.csv`
- `data/processed_causal_asof_lag1_v2/validation.csv`
- `data/processed_causal_asof_lag1_v2/test.csv`
- `data/processed_causal_asof_lag1_v2/split_manifest.json`
- `data/processed_causal_asof_lag1_v2/minute_imputation_summary.csv`
- `data/processed_causal_asof_lag1_v2/minute_imputation_audit.csv.gz`
- `data/processed_causal_asof_lag1_v2/weather_monthly_source_mapping.csv`

The repository's top-level formal-run artifacts are:

- `results/metrics_runs.csv`
- `results/metrics_summary.csv`
- `results/run_metadata.json`
- `results/baseline_metrics.csv`
- `results/integrity_report.json`
- `results/split_manifest.json`
- `results/weather_monthly_source_mapping.csv`
- `figures/*_fixed_holdout_prediction.png`
- `figures/metrics_summary_table.png`
- `logs/full_run.log`

The full server run path shown above additionally retains checkpoints and representative prediction arrays used for integrity verification.

## Provenance

- forecasting script SHA-256: `262f0405845635d3468e66c279066944019588900e257acdc356d6c7680c4fd1`
- experiment signature: `8e78be2f38d9acfe19d9b02677fb4fe355bb1f6bc1175c82a10ec379655e7971`
- preprocessing version: `causal_minute_asof_weather_lag1_v2`

The signature binds the script, source and processed data hashes, environment, model configuration, weather lag, and evaluation protocol. Full artifact hashes are recorded in `split_manifest.json` and `run_metadata.json`.

## Current artifact scope and history

The current top-level `results/`, `figures/`, and `logs/` directories are the formal `fixed_holdout_asof_lag1_v2` artifacts described here. The current processed data are in `data/processed_causal_asof_lag1_v2/`.

Earlier repository commits contain artifacts produced with same-month weather, non-causal preprocessing, or rolling-origin evaluation. Those historical artifacts must not be combined with the files in the current checkout.
