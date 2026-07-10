# Machine Learning Course Project: Household Power Forecasting

This repository contains the complete submission for the 2026 professional
master machine learning course project.

The project implements and compares:

1. LSTM forecasting
2. Transformer forecasting
3. PVG-iTransformer forecasting

It also includes PVG-iTransformer ablation runs that remove the Time Patch
branch, the Variable branch, and the learnable Gate module.

## Data And Preprocessing

The experiment uses the UCI household power data and merges monthly weather
background variables from the Meteo-France data.gouv.fr resource for department
92 Hauts-de-Seine.

The final experiment uses the SURESNES station (`NUM_POSTE=92073001`) and four
complete monthly precipitation variables: `RR`, `NBJRR1`, `NBJRR5`, and
`NBJRR10`. `RR` is divided by 10 according to the course PDF before being
merged. `NBJBROU` is not used because it is missing for 47 of the 48 months at
the selected station.

Minute-level electricity gaps are filled before daily aggregation. For each
missing minute, the script averages available same-month weekly-neighbor values
at `t+/-7`, `t+/-14`, `t+/-21`, and `t+/-28` days with the same weekday and
minute of day. The resulting daily dataset is evaluated with rolling-origin
test windows.

## Environment

The experiments were run in the isolated conda environment on the course server:

```bash
/mnt/sdc/zoujunjie/miniconda3/envs/mlearn/bin/python
```

For a fresh environment, install the Python dependencies listed in
`requirements.txt`.

## Run

```bash
BASE=/mnt/sdc/zoujunjie
WORK=$BASE/mlearn_power_coursework
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
  --rebuild-data \
  --models lstm transformer pvg_itransformer pvg_no_time pvg_no_variable pvg_no_gate \
  --epochs 30 \
  --batch-size 32 \
  --seeds 42 43 44 45 46 \
  --horizons 90 365
```

Generate the report:

```bash
"$BASE/miniconda3/envs/mlearn/bin/python" scripts/build_report_detailed.py
```

## Repository Contents

- `scripts/run_forecasting.py`: data preparation, model definitions, training, evaluation, and plotting.
- `scripts/build_report_detailed.py`: report generation script.
- `data/processed/`: processed daily power data and selected weather station summaries.
- `data/weather/`: Meteo-France monthly weather source file.
- `results/`: run-level metrics, summary metrics, and metadata.
- `figures/`: prediction curves and result table image.
- `reports/`: final DOCX and PDF report.
- `logs/`: latest full training and ablation logs.

## Main Deliverables

- `reports/mlearn_power_detailed_report.pdf`
- `reports/mlearn_power_detailed_report.docx`
- `results/metrics_summary.csv`
- `results/metrics_runs.csv`

## GitHub Submission

Repository URL:

```text
https://github.com/nianaaa/Machine-Learning-Course-Project
```
