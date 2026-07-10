# Machine Learning Course Project: Household Power Forecasting

This repository implements the three forecasting models required by the 2026
professional master machine-learning course assessment:

1. LSTM
2. Transformer
3. PVG-iTransformer

The main experiment predicts 90-day and 365-day household electricity demand
with five random seeds.

The corrected five-seed results and integrity checks are summarized in
[`EXPERIMENT_RESULTS_CAUSAL_TVT.md`](EXPERIMENT_RESULTS_CAUSAL_TVT.md).

## Data preprocessing

The electricity data comes from the UCI Individual Household Electric Power
Consumption dataset. Minute-level gaps are repaired before daily aggregation.

The previous implementation averaged both past and future weekly neighbours.
The current implementation is strictly causal:

- primary donors: original observations at `t-7`, `t-14`, `t-21`, and `t-28`
  days, at the same minute and within the same month;
- fallback: the most recent earlier raw minute when no weekly donor exists;
- filled values are never reused as donors;
- every fill records its value time, latest donor time, method, and donor count;
- the pipeline asserts `latest_donor_time < value_time`.

Dropping every day containing a missing minute was evaluated and rejected. It
would remove 82 days and split the series into 68 fragments; the longest
continuous fragment would be only 54 days, so neither the 90-day nor 365-day
task would retain a valid continuous window.

Monthly Meteo-France weather variables from the SURESNES station
(`NUM_POSTE=92073001`) are retained because weather features are required by
the course. `RR`, `NBJRR1`, `NBJRR5`, and `NBJRR10` are copied to each day in
their corresponding month. These variables are monthly background statistics,
not an as-of daily weather forecast feed, and this limitation is recorded in
the experiment metadata.

## Chronological split and evaluation

The 1,440 complete daily rows are divided chronologically:

- training: 571 days, 2006-12-17 to 2008-07-09;
- validation: 365 days, 2008-07-10 to 2009-07-09;
- test: 504 days, 2009-07-10 to 2010-11-25.

Feature and target scalers are fitted only on the training segment. A model
checkpoint is selected using validation MSE after each epoch. The final test is
never used for checkpoint selection.

Each trained checkpoint is evaluated under two separately labelled protocols:

- `fixed_holdout`: one forecast at the fixed test boundary, without using any
  later test observation as input;
- `rolling_origin`: the model parameters remain fixed, while later forecast
  origins may use observations that have become available earlier in the test
  period.

## Environment

The server environment used for the experiment is:

```bash
/mnt/sdc/zoujunjie/miniconda3/envs/mlearn/bin/python
```

Deterministic PyTorch algorithms and deterministic CUDA attention kernels are
enabled. The run metadata records the device, PyTorch version, split,
preprocessing policy, model configuration, checkpoint epoch, and both
evaluation protocols.

`requirements.txt` records the tested Python package versions. The server run
used Python 3.10.20, PyTorch 2.4.1 with CUDA 11.8, and cuDNN 9.1.

## Run

```bash
BASE=/mnt/sdc/zoujunjie
WORK=$BASE/mlearn_power_coursework
RUN=$WORK/runs/causal_tvt_full_20260711

export HOME=$BASE
export TMPDIR=$BASE/tmp
export CONDA_PKGS_DIRS=$BASE/conda_pkgs
export PIP_CACHE_DIR=$BASE/.cache/pip
export XDG_CACHE_HOME=$BASE/.cache
export PYTHONNOUSERSITE=1

cd "$WORK"
"$BASE/miniconda3/envs/mlearn/bin/python" scripts/run_forecasting.py \
  --base "$BASE" \
  --work-dir "$WORK" \
  --processed-dir "$WORK/data/processed_causal" \
  --run-dir "$RUN" \
  --rebuild-data \
  --models lstm transformer pvg_itransformer \
  --epochs 30 \
  --batch-size 32 \
  --seeds 42 43 44 45 46 \
  --horizons 90 365
```

An interrupted experiment can be continued with the same arguments plus
`--resume`. Completed model/horizon/seed combinations are skipped.

## Outputs

Processed data and audit files:

- `data/processed_causal/daily_power.csv`
- `data/processed_causal/train.csv`
- `data/processed_causal/validation.csv`
- `data/processed_causal/test.csv`
- `data/processed_causal/tes.csv`
- `data/processed_causal/split_manifest.json`
- `data/processed_causal/minute_imputation_summary.csv`
- `data/processed_causal/minute_imputation_audit.csv.gz`

Clean experiment outputs:

- `runs/causal_tvt_full_20260711/results/metrics_runs.csv`
- `runs/causal_tvt_full_20260711/results/metrics_summary.csv`
- `runs/causal_tvt_full_20260711/results/run_metadata.json`
- `runs/causal_tvt_full_20260711/results/*_predictions.npz`
- `runs/causal_tvt_full_20260711/figures/*_prediction.png`
- `runs/causal_tvt_full_20260711/figures/metrics_summary_table.png`
- `runs/causal_tvt_full_20260711/checkpoints/*_best_validation.pt`

Prediction arrays and checkpoints are generated on the server but intentionally
ignored by Git to keep the repository compact; metrics, metadata, processed
data, audit records, and figures are versioned.

Legacy files under the old top-level `results/` and `figures/` directories use
the earlier preprocessing and evaluation protocol and must not be mixed with
the clean run above.
