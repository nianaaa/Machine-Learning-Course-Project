from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_MODELS = ["lstm", "transformer", "pvg_itransformer"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the report-matching formal run")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/mnt/sdc/zoujunjie/mlearn_power_coursework"),
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    processed_dir = (
        args.processed_dir
        or work_dir / "data" / "processed_causal_asof_lag1_v2"
    ).resolve()
    run_dir = (
        args.run_dir or work_dir / "runs" / "fixed_holdout_asof_lag1_v2"
    ).resolve()
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    checkpoints_dir = run_dir / "checkpoints"
    script_path = work_dir / "scripts" / "run_forecasting.py"

    metadata = json.loads((results_dir / "run_metadata.json").read_text("utf-8"))
    script_hash = sha256_file(script_path)
    require(metadata["script"]["sha256"] == script_hash, "forecasting script hash mismatch")
    require(not (processed_dir / "tes.csv").exists(), "legacy tes.csv still exists")

    signature = metadata["experiment_signature"]
    signature_payload = signature["payload"]
    signature_digest = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    require(signature["sha256"] == signature_digest, "experiment signature mismatch")
    require(
        signature_payload["script_sha256"] == script_hash,
        "signature does not bind the current forecasting script",
    )
    require(
        signature_payload["preprocessing_version"] == metadata["preprocessing_version"],
        "signature preprocessing version differs from metadata",
    )

    manifest = json.loads((processed_dir / "split_manifest.json").read_text("utf-8"))
    for name, expected_hash in manifest["processed_output_sha256"].items():
        path = processed_dir / name
        require(path.is_file(), f"processed artifact is missing: {name}")
        require(sha256_file(path) == expected_hash, f"processed hash mismatch: {name}")

    data_paths = {
        "raw_power": work_dir / "data" / "raw" / "household_power_consumption.txt",
        "weather": work_dir
        / "data"
        / "weather"
        / "MENSQ_92_previous-1950-2024.csv.gz",
        "daily_power": processed_dir / "daily_power.csv",
        "test_split": processed_dir / "test.csv",
        "split_manifest": processed_dir / "split_manifest.json",
        "weather_source_mapping": processed_dir / "weather_monthly_source_mapping.csv",
    }
    payload_data_hashes = signature_payload["data_sha256"]
    require(
        set(payload_data_hashes) == set(data_paths),
        "signature data-hash keys differ from the formal artifact set",
    )
    require(
        metadata["data"]["data_sha256"] == payload_data_hashes,
        "metadata and signature data hashes differ",
    )
    for name, path in data_paths.items():
        require(path.is_file(), f"signature-bound input is missing: {name}")
        require(
            sha256_file(path) == payload_data_hashes[name],
            f"signature-bound data hash mismatch: {name}",
        )

    model_names = list(metadata["arguments"]["models"])
    horizons = [int(value) for value in metadata["arguments"]["horizons"]]
    seeds = [int(value) for value in metadata["arguments"]["seeds"]]
    require(model_names == REQUIRED_MODELS, f"unexpected model list: {model_names}")

    runs = pd.read_csv(results_dir / "metrics_runs.csv")
    key_cols = ["model_name", "horizon", "seed"]
    expected_keys = set(product(model_names, horizons, seeds))
    actual_keys = set(map(tuple, runs[key_cols].itertuples(index=False, name=None)))
    require(actual_keys == expected_keys, "metric keys do not match the formal configuration")
    require(not runs.duplicated(key_cols).any(), "duplicate metric keys found")
    require(set(runs["evaluation_protocol"]) == {"fixed_holdout"}, "wrong protocol")
    require((runs["test_windows"] == 1).all(), "test window count must be one")
    require((runs["validation_windows"] == 1).all(), "validation window count must be one")

    summary = pd.read_csv(results_dir / "metrics_summary.csv")
    recomputed = (
        runs.groupby(["model_name", "horizon", "evaluation_protocol"], as_index=False)
        .agg(
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", "std"),
            runs=("seed", "count"),
            train_windows=("train_windows", "first"),
            test_windows=("test_windows", "first"),
        )
        .rename(columns={"model_name": "model"})
    )
    summary_key_cols = ["model", "horizon", "evaluation_protocol"]
    expected_summary_keys = set(product(model_names, horizons, ["fixed_holdout"]))
    saved_summary_keys = set(
        map(tuple, summary[summary_key_cols].itertuples(index=False, name=None))
    )
    recomputed_summary_keys = set(
        map(tuple, recomputed[summary_key_cols].itertuples(index=False, name=None))
    )
    require(not summary.duplicated(summary_key_cols).any(), "duplicate summary keys found")
    require(
        saved_summary_keys == expected_summary_keys,
        "saved summary keys do not match the formal configuration",
    )
    require(
        recomputed_summary_keys == expected_summary_keys,
        "recomputed summary keys do not match the formal configuration",
    )
    merged = summary.merge(
        recomputed,
        on=summary_key_cols,
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    numeric_fields = [
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "runs",
        "train_windows",
        "test_windows",
    ]
    max_summary_difference = max(
        float(
            np.max(
                np.abs(
                    merged[f"{field}_saved"].to_numpy(dtype=float)
                    - merged[f"{field}_recomputed"].to_numpy(dtype=float)
                )
            )
        )
        for field in numeric_fields
    )
    require(max_summary_difference < 1.0e-9, "saved summary does not recompute")

    baseline = pd.read_csv(results_dir / "baseline_metrics.csv")
    baseline_key_cols = ["method", "horizon", "evaluation_protocol"]
    require(not baseline.duplicated(baseline_key_cols).any(), "duplicate baseline keys found")
    require(
        set(map(tuple, baseline[baseline_key_cols].itertuples(index=False, name=None)))
        == {("train_month_climatology", horizon, "fixed_holdout") for horizon in horizons},
        "baseline keys differ from the formal configuration",
    )

    daily = pd.read_csv(processed_dir / "daily_power.csv", parse_dates=["date"])
    split_ratio = float(metadata["arguments"]["split_ratio"])
    validation_days = int(metadata["arguments"]["validation_days"])
    test_start_idx = int(len(daily) * split_ratio)
    train_end_idx = test_start_idx - validation_days
    require(train_end_idx == int(manifest["train_rows"]), "baseline train range differs")
    require(test_start_idx == int(manifest["train_rows"] + manifest["validation_rows"]), "baseline test origin differs")
    monthly_means = (
        pd.DataFrame(
            {
                "month": daily["date"].iloc[:train_end_idx].dt.month,
                "target": daily["Global_active_power"].iloc[:train_end_idx],
            }
        )
        .groupby("month")["target"]
        .mean()
    )
    expected_baseline_rows = []
    for horizon in horizons:
        scored = daily.iloc[test_start_idx : test_start_idx + horizon]
        require(len(scored) == horizon, f"baseline horizon exceeds test data: {horizon}")
        prediction = scored["date"].dt.month.map(monthly_means).to_numpy(dtype=float)
        truth = scored["Global_active_power"].to_numpy(dtype=float)
        require(not np.isnan(prediction).any(), "baseline encountered an unseen month")
        expected_baseline_rows.append(
            {
                "method": "train_month_climatology",
                "horizon": horizon,
                "evaluation_protocol": "fixed_holdout",
                "mse": float(np.mean(np.square(truth - prediction))),
                "mae": float(np.mean(np.abs(truth - prediction))),
                "fit_scope": f"train_only_{train_end_idx}_days",
                "test_start": scored["date"].iloc[0].strftime("%Y-%m-%d"),
                "test_end": scored["date"].iloc[-1].strftime("%Y-%m-%d"),
            }
        )
    expected_baseline = pd.DataFrame(expected_baseline_rows)
    baseline_check = baseline.merge(
        expected_baseline,
        on=baseline_key_cols,
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    for field in ["fit_scope", "test_start", "test_end"]:
        require(
            (baseline_check[f"{field}_saved"] == baseline_check[f"{field}_recomputed"]).all(),
            f"baseline {field} differs",
        )
    baseline_max_difference = max(
        float(
            np.max(
                np.abs(
                    baseline_check[f"{field}_saved"].to_numpy(dtype=float)
                    - baseline_check[f"{field}_recomputed"].to_numpy(dtype=float)
                )
            )
        )
        for field in ["mse", "mae"]
    )
    require(baseline_max_difference < 1.0e-9, "saved baseline does not recompute")

    first_seed = seeds[0]
    expected_prediction_files = set()
    for model, horizon in product(model_names, horizons):
        path = (
            results_dir
            / f"{model}_h{horizon}_seed{first_seed}_fixed_holdout_predictions.npz"
        )
        require(path.is_file(), f"prediction archive missing: {path.name}")
        with np.load(path, allow_pickle=True) as arrays:
            require(arrays["prediction"].shape == (1, horizon), f"bad shape: {path.name}")
            require(arrays["ground_truth"].shape == (1, horizon), f"bad truth: {path.name}")
            require(str(arrays["origin_date"][0]) == "2009-07-10", f"bad origin: {path.name}")
        expected_prediction_files.add(path.name)
    actual_prediction_files = {
        path.name for path in results_dir.glob("*_fixed_holdout_predictions.npz")
    }
    require(
        actual_prediction_files == expected_prediction_files,
        "prediction archive set contains missing or extra files",
    )

    expected_checkpoints = {
        f"{model}_h{horizon}_seed{seed}_best_validation.pt"
        for model, horizon, seed in expected_keys
    }
    actual_checkpoints = {path.name for path in checkpoints_dir.glob("*.pt")}
    require(
        actual_checkpoints == expected_checkpoints,
        "checkpoint set contains missing or extra files",
    )
    expected_figures = {
        f"{model}_h{horizon}_fixed_holdout_prediction.png"
        for model, horizon in product(model_names, horizons)
    }
    actual_figures = {path.name for path in figures_dir.glob("*.png")}
    require(actual_figures == expected_figures, "figure set contains missing or extra files")

    mapping = pd.read_csv(processed_dir / "weather_monthly_source_mapping.csv")
    target_period = pd.PeriodIndex(mapping["target_month"].astype(str), freq="M")
    expected_source = (target_period - 1).strftime("%Y%m").astype(int)
    require(
        np.array_equal(expected_source, mapping["source_month"].to_numpy(dtype=int)),
        "weather mapping is not a strict one-month lag",
    )
    example = mapping.loc[mapping["target_month"] == 200907].iloc[0]
    monthly_weather = pd.read_csv(processed_dir / "weather_monthly_suresnes.csv")
    weather_example = monthly_weather.loc[
        monthly_weather["target_month"] == 200907
    ].iloc[0]

    audit = pd.read_csv(
        processed_dir / "minute_imputation_audit.csv.gz",
        parse_dates=["value_time", "latest_donor_time"],
    )
    noncausal = int((audit["latest_donor_time"] >= audit["value_time"]).sum())
    require(noncausal == 0, "non-causal minute donor found")

    report = {
        "status": "pass",
        "script_sha256": script_hash,
        "experiment_signature_sha256": signature_digest,
        "signature_bound_data_files": len(data_paths),
        "preprocessing_version": metadata["preprocessing_version"],
        "device": metadata["device"]["name"],
        "metric_rows": int(len(runs)),
        "unique_metric_keys": int(len(actual_keys)),
        "duplicate_metric_keys": int(runs.duplicated(key_cols).sum()),
        "completed_training_runs": int(metadata["completed_training_runs"]),
        "summary_max_abs_recompute_difference": max_summary_difference,
        "baseline_max_abs_recompute_difference": baseline_max_difference,
        "prediction_archives": len(actual_prediction_files),
        "checkpoints": len(actual_checkpoints),
        "figures": len(actual_figures),
        "test_windows_per_run": int(runs["test_windows"].unique()[0]),
        "validation_windows_per_run": int(runs["validation_windows"].unique()[0]),
        "test_origin": "2009-07-10",
        "weather_lag_months": int(metadata["experiment_signature"]["payload"]["weather_lag_months"]),
        "weather_mapping_example": {
            "target_month": int(example["target_month"]),
            "source_month": int(example["source_month"]),
            "RR_mm": float(weather_example["RR"]),
        },
        "minute_imputation_audit_rows": int(len(audit)),
        "noncausal_imputation_donors": noncausal,
        "ablation_models_present": False,
        "legacy_tes_csv_present": False,
    }
    output = args.output or results_dir / "integrity_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
