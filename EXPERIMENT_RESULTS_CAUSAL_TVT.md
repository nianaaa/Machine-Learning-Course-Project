# Causal Preprocessing and Chronological Validation Results

Run directory: `runs/causal_tvt_full_20260711`

This experiment replaces the earlier bidirectional minute imputation and
training-loss checkpoint selection. It uses strictly causal minute imputation,
a chronological train/validation/test split, validation-selected checkpoints,
and separately reported fixed-holdout and rolling-origin evaluation.

## Missing-data decision

The raw full-day interval contains 1,440 days and 25,979 missing minutes.
Deleting every affected day would remove 82 days and split the remaining data
into 68 fragments. The longest fragment would contain only 54 consecutive
days, leaving zero valid continuous windows for either `input=90, horizon=90`
or `input=90, horizon=365`.

The selected causal repair uses original observations from the same minute at
`t-7`, `t-14`, `t-21`, and `t-28` days. It covers 25,912/25,979 missing minutes
(99.742%). The remaining 67 minutes use the most recent earlier raw
observation; the maximum fallback lag is 38 minutes. The saved audit contains
181,853 variable-level fill records and zero cases with
`latest_donor_time >= value_time`.

## Split and checkpoint policy

| Segment | Dates | Days |
|---|---|---:|
| Train | 2006-12-17 to 2008-07-09 | 571 |
| Validation | 2008-07-10 to 2009-07-09 | 365 |
| Test | 2009-07-10 to 2010-11-25 | 504 |

All scalers are fitted on the training segment only. Each epoch is evaluated
on validation data, and the lowest validation-MSE checkpoint is used for both
test protocols. Across the 30 training runs, selected checkpoints occurred
between epochs 3 and 21.

Monthly SURESNES weather statistics are still copied to each day of the same
month because the course requires weather features. They are monthly
background statistics rather than a real-time/as-of weather feed; this
limitation is explicitly retained in the run metadata.

## Five-seed test results

Values are mean ± sample standard deviation over seeds 42-46.

| Horizon | Protocol | Model | MSE | MAE |
|---:|---|---|---:|---:|
| 90 | Fixed holdout | LSTM | **114026.73 ± 18041.30** | **269.58 ± 25.90** |
| 90 | Fixed holdout | Transformer | 162343.10 ± 25156.72 | 334.03 ± 25.83 |
| 90 | Fixed holdout | PVG-iTransformer | 205982.81 ± 55601.24 | 371.29 ± 51.95 |
| 90 | Rolling origin | LSTM | 163790.00 ± 2866.45 | 316.07 ± 3.08 |
| 90 | Rolling origin | Transformer | 168910.70 ± 9199.44 | 323.07 ± 10.49 |
| 90 | Rolling origin | PVG-iTransformer | **159866.64 ± 10748.08** | **314.26 ± 13.20** |
| 365 | Fixed holdout | LSTM | 157782.37 ± 5982.65 | 314.69 ± 8.74 |
| 365 | Fixed holdout | Transformer | 152953.03 ± 10790.84 | 300.85 ± 10.47 |
| 365 | Fixed holdout | PVG-iTransformer | **144538.92 ± 3880.30** | **293.40 ± 2.88** |
| 365 | Rolling origin | LSTM | 297809.74 ± 21072.39 | 429.38 ± 17.19 |
| 365 | Rolling origin | Transformer | 376921.80 ± 25017.89 | 492.07 ± 20.63 |
| 365 | Rolling origin | PVG-iTransformer | **285783.83 ± 15876.76** | **423.41 ± 13.63** |

## Interpretation

The model ranking depends on the evaluation protocol:

- PVG-iTransformer has the best mean result for both rolling-origin horizons
  and for the 365-day fixed holdout.
- LSTM has the best 90-day fixed-holdout result.
- Therefore the corrected experiment does not support an unconditional claim
  that one model is superior under every deployment protocol.

No seasonal or linear baselines are included in this run. These results compare
only the three course-required neural models.

## Integrity checks

- 30 unique model/horizon/seed training runs;
- 60 unique protocol-level metric rows;
- five seeds in every model/horizon/protocol group;
- 30 validation-selected checkpoints;
- zero duplicate metric keys and zero missing metrics;
- summary CSV recomputation maximum absolute difference: `1.82e-12`;
- 1,440 consecutive daily rows with zero calendar gaps;
- repeated deterministic smoke-test prediction SHA256 values were identical.
