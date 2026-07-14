from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import site
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
WEATHER_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/MENS/MENSQ_92_previous-1950-2024.csv.gz"
)
WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10"]
WEATHER_STATION_ID = 92073001
WEATHER_STATION_NAME = "SURESNES"
WEATHER_STATION_DISTANCE_KM = 11.1
PREPROCESSING_VERSION = "causal_minute_asof_weather_lag1_v2"
WEATHER_LAG_MONTHS = 1
EVALUATION_PROTOCOL = "fixed_holdout"
DEFAULT_MODELS = ["lstm", "transformer", "pvg_itransformer"]
MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "pvg_itransformer": "PVG-iTransformer",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_environment_metadata() -> dict:
    modules = {
        "numpy": np,
        "pandas": pd,
        "scikit_learn": sklearn,
        "matplotlib": matplotlib,
        "torch": torch,
    }
    dependencies = {}
    for name, module in modules.items():
        dependencies[name] = {
            "version": str(getattr(module, "__version__", "unknown")),
            "path": str(Path(module.__file__).resolve()),
        }

    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_sites = [user_site]
    else:
        user_sites = list(user_site)
    resolved_user_sites = []
    for path in user_sites:
        try:
            resolved_user_sites.append(Path(path).expanduser().resolve())
        except OSError:
            continue
    loaded_from_user_site = []
    for name, entry in dependencies.items():
        module_path = Path(entry["path"])
        if any(module_path == root or root in module_path.parents for root in resolved_user_sites):
            loaded_from_user_site.append(name)

    return {
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "dependencies": dependencies,
        "python_no_user_site_flag": bool(sys.flags.no_user_site),
        "python_user_site_paths": [str(path) for path in resolved_user_sites],
        "packages_loaded_from_user_site": loaded_from_user_site,
    }


def warn_about_user_site(environment: dict) -> None:
    loaded = environment["packages_loaded_from_user_site"]
    if loaded:
        print(
            "WARNING: packages were imported from the Python user-site before "
            f"runtime environment configuration: {loaded}. Their exact paths are recorded "
            "in run_metadata.json."
        )
    elif not environment["python_no_user_site_flag"]:
        print(
            "WARNING: Python was not launched with user-site disabled "
            "(sys.flags.no_user_site=False). Exact dependency paths are recorded in metadata."
        )


def device_metadata(device: torch.device) -> dict:
    if device.type == "cuda":
        capability = list(torch.cuda.get_device_capability(0))
        name = torch.cuda.get_device_name(0)
    else:
        capability = None
        name = "CPU"
    return {
        "type": str(device),
        "name": name,
        "cuda_capability": capability,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def build_experiment_signature(
    args: argparse.Namespace,
    script_hash: str,
    data_hashes: dict[str, str],
    device_info: dict,
    environment_info: dict,
) -> dict:
    payload = {
        "script_sha256": script_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "weather_lag_months": WEATHER_LAG_MONTHS,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "data_sha256": data_hashes,
        "environment": environment_info,
        "device": {
            "type": device_info["type"],
            "name": device_info["name"],
            "cuda_capability": device_info["cuda_capability"],
            "torch_cuda_version": device_info["torch_cuda_version"],
        },
        "arguments": {
            "input_len": args.input_len,
            "split_ratio": args.split_ratio,
            "validation_days": args.validation_days,
            "pvg_time_pooling": args.pvg_time_pooling,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": list(args.seeds),
            "horizons": list(args.horizons),
            "models": list(args.models),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "payload": payload}


def signature_differences(expected, actual, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{child}: unexpected prior value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing from prior metadata")
            else:
                differences.extend(signature_differences(expected[key], actual[key], child))
        return differences
    if expected != actual:
        return [f"{prefix}: current={expected!r}, prior={actual!r}"]
    return []


def validate_resume_signature(metadata_path: Path, current_signature: dict) -> dict:
    if not metadata_path.exists():
        raise RuntimeError(
            f"Cannot --resume because {metadata_path} is missing. Choose a new --run-dir."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        prior_metadata = json.load(f)
    prior_signature = prior_metadata.get("experiment_signature")
    if not isinstance(prior_signature, dict) or "payload" not in prior_signature:
        raise RuntimeError(
            "Cannot --resume: prior metadata has no compatible experiment signature. "
            "Choose a new --run-dir."
        )
    differences = signature_differences(
        current_signature["payload"], prior_signature["payload"]
    )
    if differences or current_signature["sha256"] != prior_signature.get("sha256"):
        detail = "\n  - ".join(differences[:20]) or "signature digest differs"
        raise RuntimeError(
            "Refusing to mix incompatible runs under --resume. Differences:\n  - "
            f"{detail}\nChoose a new --run-dir or restore the original experiment environment."
        )
    return prior_metadata


def configure_environment(base: Path) -> None:
    os.environ.setdefault("HOME", str(base))
    os.environ.setdefault("TMPDIR", str(base / "tmp"))
    os.environ.setdefault("CONDA_PKGS_DIRS", str(base / "conda_pkgs"))
    os.environ.setdefault("PIP_CACHE_DIR", str(base / ".cache" / "pip"))
    os.environ.setdefault("XDG_CACHE_HOME", str(base / ".cache"))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for p in [
        base / "tmp",
        base / "conda_pkgs",
        base / ".cache" / "pip",
        base / ".cache" / "matplotlib",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def model_display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def download_uci(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "individual_household_power_consumption.zip"
    txt_path = raw_dir / "household_power_consumption.txt"
    if txt_path.exists():
        return txt_path
    if not zip_path.exists():
        print(f"Downloading UCI household power dataset to {zip_path}")
        urllib.request.urlretrieve(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [name for name in zf.namelist() if name.endswith(".txt")]
        if not candidates:
            raise RuntimeError("UCI archive does not contain a .txt file")
        zf.extract(candidates[0], raw_dir)
        extracted = raw_dir / candidates[0]
        if extracted != txt_path:
            extracted.replace(txt_path)
    return txt_path


def download_weather(weather_dir: Path) -> Path:
    weather_dir.mkdir(parents=True, exist_ok=True)
    weather_path = weather_dir / "MENSQ_92_previous-1950-2024.csv.gz"
    if not weather_path.exists():
        print(f"Downloading Meteo-France monthly weather data to {weather_path}")
        urllib.request.urlretrieve(WEATHER_URL, weather_path)
    return weather_path


def previous_month_id(month_id: int) -> int:
    year, month = divmod(int(month_id), 100)
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid YYYYMM month id: {month_id}")
    period = pd.Period(year=year, month=month, freq="M") - WEATHER_LAG_MONTHS
    return int(period.strftime("%Y%m"))


def build_weather_monthly(
    weather_path: Path,
    out_dir: Path,
    month_ids: list[int],
) -> pd.DataFrame:
    """Build as-of monthly weather features without using the target month.

    Each target calendar month receives measurements from the immediately preceding
    completed calendar month. The returned merge key remains ``AAAAMM`` and the four
    feature names remain unchanged for compatibility with the model input schema.
    """
    target_months = [int(month_id) for month_id in month_ids]
    if len(target_months) != len(set(target_months)):
        raise ValueError("month_ids must not contain duplicates")
    source_months = [previous_month_id(month_id) for month_id in target_months]

    weather = pd.read_csv(weather_path, sep=";", compression="gzip", low_memory=False)
    keep_cols = ["AAAAMM", "NUM_POSTE", "NOM_USUEL", "LAT", "LON", *WEATHER_COLS]
    weather = weather[keep_cols].copy()
    for col in ["AAAAMM", "NUM_POSTE", "LAT", "LON", *WEATHER_COLS]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather[
        (weather["NUM_POSTE"] == WEATHER_STATION_ID)
        & weather["AAAAMM"].isin(source_months)
    ].copy()

    source_values = weather.sort_values("AAAAMM").drop_duplicates("AAAAMM")
    source_values = source_values.set_index("AAAAMM").reindex(source_months)
    missing = source_values[WEATHER_COLS].isna()
    if missing.any().any():
        missing_months = {
            col: source_values.index[missing[col]].astype(int).tolist()
            for col in WEATHER_COLS
            if missing[col].any()
        }
        raise RuntimeError(
            "Missing SURESNES values for preceding source months required by the "
            f"as-of lag-{WEATHER_LAG_MONTHS} merge: {missing_months}"
        )

    station_summary = (
        weather.groupby(["NUM_POSTE", "NOM_USUEL", "LAT", "LON"], dropna=False)
        .agg(
            months_present=("AAAAMM", "nunique"),
            rr_missing=("RR", lambda s: int(s.isna().sum())),
            nbjrr1_missing=("NBJRR1", lambda s: int(s.isna().sum())),
            nbjrr5_missing=("NBJRR5", lambda s: int(s.isna().sum())),
            nbjrr10_missing=("NBJRR10", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    station_summary.to_csv(out_dir / "weather_station_suresnes_summary.csv", index=False)

    source_values = source_values.reset_index().rename(columns={"AAAAMM": "source_month"})
    source_values["source_month"] = source_values["source_month"].astype(int)
    mapping = pd.DataFrame(
        {"target_month": target_months, "source_month": source_months}
    )
    lagged = mapping.merge(
        source_values[["source_month", *WEATHER_COLS]],
        on="source_month",
        how="left",
        validate="one_to_one",
    )
    if lagged[WEATHER_COLS].isna().any().any():
        raise RuntimeError("As-of monthly weather mapping unexpectedly produced missing values")
    mapping.to_csv(out_dir / "weather_monthly_source_mapping.csv", index=False)
    lagged.to_csv(out_dir / "weather_monthly_suresnes.csv", index=False)

    merge_ready = lagged.rename(columns={"target_month": "AAAAMM"})
    return merge_ready[["AAAAMM", *WEATHER_COLS]]


def impute_minute_power(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Fill minute gaps using original observations strictly from the past.

    The primary donor set is the same minute of day at t-7/-14/-21/-28 days,
    restricted to the same calendar month to retain the original coursework
    preprocessing intent. If none of those raw observations exists, the most
    recent raw observation is used as a short causal forward-fill fallback.
    Filled values are never reused as donors.
    """
    offsets = [-28, -21, -14, -7]
    index = df.index
    original = df[numeric_cols].copy()
    summary: dict[str, dict] = {}
    audit_frames: list[pd.DataFrame] = []

    for col in numeric_cols:
        original_col = original[col]
        missing = original_col.isna().to_numpy()
        candidates = []
        candidate_times = []
        for days in offsets:
            candidate_index = index + pd.Timedelta(days=days)
            candidate = original_col.reindex(candidate_index).to_numpy(dtype=np.float64)
            same_month = (
                (candidate_index.year == index.year)
                & (candidate_index.month == index.month)
            )
            candidate[~same_month] = np.nan
            candidates.append(candidate)
            times = candidate_index.to_numpy(dtype="datetime64[ns]")
            times[~same_month] = np.datetime64("NaT")
            candidate_times.append(times)

        stacked = np.vstack(candidates)
        available_counts = np.sum(~np.isnan(stacked), axis=0)
        fill_values = np.full(len(index), np.nan, dtype=np.float64)
        np.divide(
            np.nansum(stacked, axis=0),
            available_counts,
            out=fill_values,
            where=available_counts > 0,
        )

        latest_donor_time = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
        for candidate, times in zip(candidates, candidate_times):
            valid = ~np.isnan(candidate)
            latest_donor_time[valid] = times[valid]

        weekly_fill = missing & (available_counts > 0)
        fallback_fill = missing & (available_counts == 0)
        if fallback_fill.any():
            fallback_values = original_col.ffill().to_numpy(dtype=np.float64)
            observed_times = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
            observed = original_col.notna().to_numpy()
            observed_times.iloc[np.flatnonzero(observed)] = index[observed].to_numpy()
            fallback_times = observed_times.ffill().to_numpy(dtype="datetime64[ns]")
            if np.isnan(fallback_values[fallback_fill]).any() or pd.isna(
                fallback_times[fallback_fill]
            ).any():
                bad_times = index[fallback_fill & np.isnan(fallback_values)]
                raise RuntimeError(
                    f"{col} has missing minutes without any earlier raw observation: "
                    f"{bad_times[:10].astype(str).tolist()}"
                )
            fill_values[fallback_fill] = fallback_values[fallback_fill]
            latest_donor_time[fallback_fill] = fallback_times[fallback_fill]

        if np.isnan(fill_values[missing]).any():
            raise RuntimeError(f"Causal imputation left unresolved values in {col}")

        value_times = index[missing].to_numpy(dtype="datetime64[ns]")
        donor_times = latest_donor_time[missing]
        if pd.isna(donor_times).any() or np.any(donor_times >= value_times):
            raise AssertionError(f"Non-causal donor detected while imputing {col}")

        df.loc[df.index[missing], col] = fill_values[missing]
        methods = np.where(weekly_fill[missing], "past_weekly_mean", "past_forward_fill")
        donor_counts = np.where(weekly_fill[missing], available_counts[missing], 1)
        lag_minutes = (value_times - donor_times) / np.timedelta64(1, "m")
        audit_frames.append(
            pd.DataFrame(
                {
                    "variable": col,
                    "value_time": pd.to_datetime(value_times),
                    "latest_donor_time": pd.to_datetime(donor_times),
                    "method": methods,
                    "donor_count": donor_counts.astype(int),
                    "latest_donor_lag_minutes": lag_minutes.astype(float),
                }
            )
        )
        summary[col] = {
            "missing_minutes": int(missing.sum()),
            "weekly_mean_minutes": int(weekly_fill.sum()),
            "forward_fill_minutes": int(fallback_fill.sum()),
            "min_candidates": int(available_counts[missing].min()) if missing.any() else 0,
            "max_candidates": int(available_counts[missing].max()) if missing.any() else 0,
            "mean_candidates": float(available_counts[missing].mean()) if missing.any() else 0.0,
            "max_donor_lag_minutes": float(lag_minutes.max()) if missing.any() else 0.0,
        }

    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    return summary, audit


def build_daily_frame(
    raw_txt: Path,
    weather_path: Path,
    out_dir: Path,
    split_ratio: float,
    validation_days: int,
    rebuild: bool = False,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "daily_power.csv"
    train_path = out_dir / "train.csv"
    validation_path = out_dir / "validation.csv"
    test_path = out_dir / "test.csv"
    audit_path = out_dir / "minute_imputation_audit.csv.gz"
    split_manifest_path = out_dir / "split_manifest.json"
    required_cache_paths = {
        "daily_power.csv": daily_path,
        "train.csv": train_path,
        "validation.csv": validation_path,
        "test.csv": test_path,
        "minute_imputation_audit.csv.gz": audit_path,
        "minute_imputation_summary.csv": out_dir / "minute_imputation_summary.csv",
        "weather_monthly_source_mapping.csv": out_dir / "weather_monthly_source_mapping.csv",
        "weather_monthly_suresnes.csv": out_dir / "weather_monthly_suresnes.csv",
        "weather_station_suresnes_summary.csv": (
            out_dir / "weather_station_suresnes_summary.csv"
        ),
        "split_manifest.json": split_manifest_path,
    }
    existing_cache_paths = {
        name: path for name, path in required_cache_paths.items() if path.exists()
    }
    if not rebuild and existing_cache_paths:
        problems = []
        missing_names = sorted(set(required_cache_paths) - set(existing_cache_paths))
        if missing_names:
            problems.append(f"missing cache files: {missing_names}")

        manifest = None
        if split_manifest_path.exists():
            try:
                with split_manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"cannot read split_manifest.json: {exc}")
        if manifest is not None:
            expected_manifest_values = {
                "preprocessing_version": PREPROCESSING_VERSION,
                "validation_days": int(validation_days),
                "weather_lag_months": WEATHER_LAG_MONTHS,
                "weather_rr_unit": "mm",
            }
            for key, expected in expected_manifest_values.items():
                if manifest.get(key) != expected:
                    problems.append(
                        f"split_manifest.{key}: cached={manifest.get(key)!r}, "
                        f"requested={expected!r}"
                    )
            cached_ratio = manifest.get("split_ratio_train_plus_validation")
            if not isinstance(cached_ratio, (int, float)) or not math.isclose(
                float(cached_ratio), float(split_ratio), rel_tol=0.0, abs_tol=1.0e-12
            ):
                problems.append(
                    "split_manifest.split_ratio_train_plus_validation: "
                    f"cached={cached_ratio!r}, requested={split_ratio!r}"
                )

            source_hashes = manifest.get("source_data_sha256", {})
            current_source_hashes = {
                "raw_power": sha256_file(raw_txt),
                "weather": sha256_file(weather_path),
            }
            for key, expected in current_source_hashes.items():
                if source_hashes.get(key) != expected:
                    problems.append(
                        f"source hash {key}: cached={source_hashes.get(key)!r}, "
                        f"current={expected!r}"
                    )

            output_hashes = manifest.get("processed_output_sha256", {})
            for name, path in required_cache_paths.items():
                if name == "split_manifest.json" or not path.exists():
                    continue
                expected_hash = output_hashes.get(name)
                current_hash = sha256_file(path)
                if expected_hash != current_hash:
                    problems.append(
                        f"processed hash {name}: cached={expected_hash!r}, "
                        f"current={current_hash!r}"
                    )

        cached = None
        if daily_path.exists():
            try:
                cached = pd.read_csv(daily_path, parse_dates=["date"])
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                problems.append(f"cannot read daily_power.csv: {exc}")
        if cached is not None:
            missing_columns = [col for col in WEATHER_COLS if col not in cached.columns]
            if missing_columns:
                problems.append(f"daily_power.csv missing weather columns: {missing_columns}")
            if "NBJBROU" in cached.columns:
                problems.append("daily_power.csv contains excluded NBJBROU")
            expected_test_start = int(len(cached) * split_ratio)
            expected_train_end = expected_test_start - validation_days
            if manifest is not None:
                expected_rows = {
                    "train_rows": expected_train_end,
                    "validation_rows": validation_days,
                    "test_rows": len(cached) - expected_test_start,
                }
                for key, expected in expected_rows.items():
                    if manifest.get(key) != expected:
                        problems.append(
                            f"split_manifest.{key}: cached={manifest.get(key)!r}, "
                            f"expected={expected!r}"
                        )

        if problems:
            detail = "\n  - ".join(problems)
            raise RuntimeError(
                f"Processed cache is incompatible or incomplete:\n  - {detail}\n"
                "Rerun with --rebuild-data (or choose a new --processed-dir)."
            )
        if cached is None:
            raise RuntimeError(
                "Processed cache could not be validated. Rerun with --rebuild-data."
            )
        return cached

    df = pd.read_csv(
        raw_txt,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "datetime", dt)

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("datetime").sort_index().asfreq("1min")
    imputation_summary, imputation_audit = impute_minute_power(df, numeric_cols)
    imputation_df = pd.DataFrame.from_dict(imputation_summary, orient="index")
    imputation_df.index.name = "variable"
    imputation_df.reset_index().to_csv(out_dir / "minute_imputation_summary.csv", index=False)
    imputation_audit.to_csv(
        audit_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    df["Sub_metering_remainder"] = (
        df["Global_active_power"] * 1000.0 / 60.0
        - (
            df["Sub_metering_1"]
            + df["Sub_metering_2"]
            + df["Sub_metering_3"]
        )
    )
    agg = {
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
        "Sub_metering_remainder": "sum",
    }
    daily = df.resample("D").agg(agg)
    full_days = (daily.index > df.index.min().normalize()) & (
        daily.index < df.index.max().normalize()
    )
    daily = daily.loc[full_days].copy()
    daily["dayofweek_sin"] = np.sin(2 * np.pi * daily.index.dayofweek / 7)
    daily["dayofweek_cos"] = np.cos(2 * np.pi * daily.index.dayofweek / 7)
    daily["month_sin"] = np.sin(2 * np.pi * daily.index.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily.index.month / 12)
    daily["AAAAMM"] = daily.index.strftime("%Y%m").astype(int)
    month_ids = [
        int(period.strftime("%Y%m"))
        for period in pd.period_range(daily.index.min(), daily.index.max(), freq="M")
    ]
    weather_monthly = build_weather_monthly(weather_path, out_dir, month_ids)
    daily = daily.reset_index().rename(columns={"datetime": "date"})
    daily = daily.merge(weather_monthly, on="AAAAMM", how="left")
    if daily[WEATHER_COLS].isna().any().any():
        raise RuntimeError("Daily frame has missing weather values after SURESNES merge")
    daily = daily.drop(columns=["AAAAMM"])

    test_start_idx = int(len(daily) * split_ratio)
    train_end_idx = test_start_idx - validation_days
    if train_end_idx <= 0:
        raise ValueError(
            f"validation_days={validation_days} leaves no training rows before "
            f"test_start_idx={test_start_idx}"
        )
    daily.to_csv(daily_path, index=False)
    daily.iloc[:train_end_idx].to_csv(train_path, index=False)
    daily.iloc[train_end_idx:test_start_idx].to_csv(validation_path, index=False)
    daily.iloc[test_start_idx:].to_csv(test_path, index=False)
    split_manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "strategy": "chronological_self_split",
        "split_ratio_train_plus_validation": split_ratio,
        "validation_days": validation_days,
        "weather_lag_months": WEATHER_LAG_MONTHS,
        "weather_as_of_rule": "target calendar month uses previous completed calendar month",
        "weather_rr_unit": "mm",
        "train_rows": int(train_end_idx),
        "validation_rows": int(test_start_idx - train_end_idx),
        "test_rows": int(len(daily) - test_start_idx),
        "train_start": str(daily["date"].iloc[0].date()),
        "train_end": str(daily["date"].iloc[train_end_idx - 1].date()),
        "validation_start": str(daily["date"].iloc[train_end_idx].date()),
        "validation_end": str(daily["date"].iloc[test_start_idx - 1].date()),
        "test_start": str(daily["date"].iloc[test_start_idx].date()),
        "test_end": str(daily["date"].iloc[-1].date()),
        "source_data_sha256": {
            "raw_power": sha256_file(raw_txt),
            "weather": sha256_file(weather_path),
        },
    }
    split_manifest["processed_output_sha256"] = {
        name: sha256_file(path)
        for name, path in required_cache_paths.items()
        if name != "split_manifest.json"
    }
    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)
    return daily


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        starts: np.ndarray,
        input_len: int,
        output_len: int,
    ) -> None:
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.starts = starts.astype(np.int64)
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        x = self.features[start : start + self.input_len]
        y_start = start + self.input_len
        y = self.target[y_start : y_start + self.output_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_windows(
    daily: pd.DataFrame,
    input_len: int,
    output_len: int,
    split_ratio: float,
    validation_days: int,
) -> tuple[WindowDataset, WindowDataset, dict[str, WindowDataset], dict]:
    feature_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "Sub_metering_remainder",
        *WEATHER_COLS,
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
    ]
    target_col = "Global_active_power"
    values = daily[feature_cols].to_numpy(dtype=np.float32)
    target = daily[target_col].to_numpy(dtype=np.float32)
    test_start_idx = int(len(daily) * split_ratio)
    train_end_idx = test_start_idx - validation_days
    if validation_days < output_len:
        raise ValueError(
            f"validation_days={validation_days} must be at least horizon={output_len}"
        )
    if train_end_idx < input_len + output_len:
        raise ValueError(
            f"Training segment is too short for input_len={input_len}, "
            f"horizon={output_len}: train_rows={train_end_idx}"
        )

    feat_mean = values[:train_end_idx].mean(axis=0, keepdims=True)
    feat_std = values[:train_end_idx].std(axis=0, keepdims=True) + 1.0e-6
    target_mean = float(target[:train_end_idx].mean())
    target_std = float(target[:train_end_idx].std() + 1.0e-6)

    values_norm = (values - feat_mean) / feat_std
    target_norm = (target - target_mean) / target_std

    max_start = len(daily) - input_len - output_len + 1
    all_starts = np.arange(max_start, dtype=np.int64)
    out_start = all_starts + input_len
    out_end = out_start + output_len
    train_starts = all_starts[out_end <= train_end_idx]
    validation_starts = all_starts[out_start == train_end_idx]
    fixed_holdout_starts = all_starts[out_start == test_start_idx]
    if (
        len(train_starts) == 0
        or len(validation_starts) != 1
        or len(fixed_holdout_starts) != 1
    ):
        raise RuntimeError(
            f"Not enough windows for output_len={output_len}: "
            f"train={len(train_starts)}, validation={len(validation_starts)}, "
            f"fixed={len(fixed_holdout_starts)}"
        )

    train = WindowDataset(values_norm, target_norm, train_starts, input_len, output_len)
    validation = WindowDataset(
        values_norm,
        target_norm,
        validation_starts,
        input_len,
        output_len,
    )
    test_sets = {
        "fixed_holdout": WindowDataset(
            values_norm,
            target_norm,
            fixed_holdout_starts,
            input_len,
            output_len,
        ),
    }
    dates = pd.to_datetime(daily["date"])
    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "train_windows": int(len(train_starts)),
        "validation_windows": int(len(validation_starts)),
        "validation_protocol": "fixed_holdout",
        "validation_origin_dates": [
            str(dates.iloc[int(start) + input_len].date())
            for start in validation_starts
        ],
        "fixed_holdout_windows": int(len(fixed_holdout_starts)),
        "train_date_start": str(dates.iloc[0].date()),
        "train_date_end": str(dates.iloc[train_end_idx - 1].date()),
        "validation_date_start": str(dates.iloc[train_end_idx].date()),
        "validation_date_end": str(dates.iloc[test_start_idx - 1].date()),
        "test_date_start": str(dates.iloc[test_start_idx].date()),
        "test_date_end": str(dates.iloc[-1].date()),
        "scaler_fit_end": str(dates.iloc[train_end_idx - 1].date()),
    }
    return train, validation, test_sets, meta


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_len: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


def make_patch_starts(input_len: int, patch_len: int, stride: int) -> list[int]:
    if patch_len > input_len:
        raise ValueError("patch_len must not exceed input_len")
    starts = list(range(0, input_len - patch_len + 1, stride))
    last_start = input_len - patch_len
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def infer_variable_group_ids(feature_cols: list[str]) -> list[int]:
    group_ids = []
    for col in feature_cols:
        if col == "Global_active_power":
            group_ids.append(0)
        elif col in {"Global_reactive_power", "Voltage", "Global_intensity"}:
            group_ids.append(1)
        elif col.startswith("Sub_metering"):
            group_ids.append(2)
        elif col in WEATHER_COLS:
            group_ids.append(3)
        elif col in {"dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"}:
            group_ids.append(4)
        else:
            group_ids.append(1)
    return group_ids


class PVGiTransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_len: int,
        input_len: int,
        feature_cols: list[str],
        d_model: int = 64,
        nhead: int = 4,
        patch_len: int = 7,
        stride: int = 3,
        patch_layers: int = 2,
        var_layers: int = 2,
        dropout: float = 0.1,
        time_pooling: str = "last",
    ) -> None:
        super().__init__()
        if time_pooling not in {"mean", "last"}:
            raise ValueError(f"Unknown time_pooling={time_pooling}")
        patch_starts = make_patch_starts(input_len, patch_len, stride)
        self.input_dim = input_dim
        self.input_len = input_len
        self.patch_len = patch_len
        self.time_pooling = time_pooling
        self.register_buffer(
            "patch_starts",
            torch.tensor(patch_starts, dtype=torch.long),
            persistent=False,
        )

        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        self.patch_pos = nn.Parameter(torch.zeros(1, len(patch_starts), d_model))
        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.patch_encoder = nn.TransformerEncoder(patch_layer, num_layers=patch_layers)

        self.var_embed = nn.Linear(input_len, d_model)
        self.var_type_embed = nn.Embedding(input_dim, d_model)
        self.var_group_embed = nn.Embedding(5, d_model)
        self.register_buffer(
            "var_ids",
            torch.arange(input_dim, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "var_group_ids",
            torch.tensor(infer_variable_group_ids(feature_cols), dtype=torch.long),
            persistent=False,
        )
        var_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.var_encoder = nn.TransformerEncoder(var_layer, num_layers=var_layers)

        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )
        nn.init.normal_(self.patch_pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_starts = self.patch_starts.detach().cpu().tolist()
        patches = [
            x[:, start : start + self.patch_len, :].reshape(x.size(0), -1)
            for start in patch_starts
        ]
        patch_tokens = self.patch_embed(torch.stack(patches, dim=1)) + self.patch_pos
        patch_states = self.patch_encoder(patch_tokens)
        if self.time_pooling == "last":
            f_time = patch_states[:, -1]
        else:
            f_time = patch_states.mean(dim=1)

        var_tokens = self.var_embed(x.transpose(1, 2))
        var_tokens = (
            var_tokens
            + self.var_type_embed(self.var_ids).unsqueeze(0)
            + self.var_group_embed(self.var_group_ids).unsqueeze(0)
        )
        f_var = self.var_encoder(var_tokens)[:, 0]

        gate = self.gate(torch.cat([f_time, f_var], dim=-1))
        fused = gate * f_time + (1.0 - gate) * f_var
        return self.head(fused)


@dataclass
class TrainConfig:
    model_name: str
    horizon: int
    seed: int
    input_len: int
    validation_protocol: str
    pvg_time_pooling: str
    epochs: int
    batch_size: int
    lr: float


def build_model(
    name: str,
    input_dim: int,
    output_len: int,
    input_len: int,
    feature_cols: list[str],
    pvg_time_pooling: str,
) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(input_dim=input_dim, output_len=output_len)
    if name == "transformer":
        return TransformerForecaster(input_dim=input_dim, output_len=output_len)
    if name == "pvg_itransformer":
        return PVGiTransformerForecaster(
            input_dim=input_dim,
            output_len=output_len,
            input_len=input_len,
            feature_cols=feature_cols,
            time_pooling=pvg_time_pooling,
        )
    raise ValueError(name)


def train_one(
    cfg: TrainConfig,
    train_ds: WindowDataset,
    validation_ds: WindowDataset,
    test_sets: dict[str, WindowDataset],
    input_dim: int,
    feature_cols: list[str],
    target_mean: float,
    target_std: float,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = build_model(
        cfg.model_name,
        input_dim,
        cfg.horizon,
        cfg.input_len,
        feature_cols,
        cfg.pvg_time_pooling,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    final_train_loss = float("nan")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item()) * len(x)
            seen += len(x)
        train_loss = total / max(seen, 1)

        model.eval()
        val_total = 0.0
        val_seen = 0
        with torch.inference_mode():
            for x, y in validation_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_loss = loss_fn(pred, y)
                val_total += float(val_loss.item()) * len(x)
                val_seen += len(x)
        validation_loss = val_total / max(val_seen, 1)
        final_train_loss = train_loss
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"{cfg.model_name} horizon={cfg.horizon} seed={cfg.seed} "
                f"epoch={epoch}/{cfg.epochs} train_mse_norm={train_loss:.5f} "
                f"validation_mse_norm={validation_loss:.5f}"
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(cfg),
            "best_epoch": best_epoch,
            "best_validation_mse_norm": best_val_loss,
            "feature_cols": feature_cols,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        checkpoint_path,
    )
    model.eval()
    results: list[dict] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for protocol, test_ds in test_sets.items():
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )
        preds = []
        trues = []
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x).cpu().numpy()
                preds.append(pred)
                trues.append(y.numpy())
        pred_norm = np.concatenate(preds, axis=0)
        true_norm = np.concatenate(trues, axis=0)
        pred = pred_norm * target_std + target_mean
        true = true_norm * target_std + target_mean
        mse = mean_squared_error(true.reshape(-1), pred.reshape(-1))
        mae = mean_absolute_error(true.reshape(-1), pred.reshape(-1))
        results.append(
            {
                **asdict(cfg),
                "evaluation_protocol": protocol,
                "test_mse": float(mse),
                "test_mae": float(mae),
                "best_epoch": int(best_epoch),
                "best_validation_mse_norm": float(best_val_loss),
                "final_train_mse_norm": float(final_train_loss),
                "test_windows": int(len(test_ds)),
                "validation_windows": int(len(validation_ds)),
                "train_windows": int(len(train_ds)),
                "checkpoint_path": str(checkpoint_path),
            }
        )
        predictions[protocol] = (pred, true)
    return results, predictions


def plot_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    title: str,
    ylabel: str = "Daily global active power",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8), dpi=180)
    x = np.arange(true.shape[1])
    plt.plot(x, true[0], label="Ground Truth", linewidth=2.0)
    plt.plot(x, pred[0], label="Prediction", linewidth=2.0)
    plt.xlabel("Forecast day")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["model_name", "horizon", "evaluation_protocol"], as_index=False)
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
    model_order = {name: idx for idx, name in enumerate(DEFAULT_MODELS)}
    summary["_model_order"] = summary["model"].map(model_order).fillna(99)
    summary = summary.sort_values(
        ["horizon", "_model_order", "model"]
    ).drop(
        columns=["_model_order"]
    )
    return summary


def train_only_month_climatology(
    daily: pd.DataFrame,
    split_ratio: float,
    validation_days: int,
    horizons: list[int],
) -> pd.DataFrame:
    test_start_idx = int(len(daily) * split_ratio)
    train_end_idx = test_start_idx - validation_days
    dates = pd.to_datetime(daily["date"])
    target = daily["Global_active_power"].to_numpy(dtype=float)
    train_months = dates.iloc[:train_end_idx].dt.month
    monthly_means = (
        pd.DataFrame({"month": train_months, "target": target[:train_end_idx]})
        .groupby("month")["target"]
        .mean()
    )
    rows = []
    for horizon in horizons:
        end_idx = test_start_idx + int(horizon)
        if end_idx > len(daily):
            raise ValueError(
                f"horizon={horizon} exceeds the fixed test segment of "
                f"{len(daily) - test_start_idx} days"
            )
        scored_dates = dates.iloc[test_start_idx:end_idx]
        truth = target[test_start_idx:end_idx]
        prediction = scored_dates.dt.month.map(monthly_means).to_numpy(dtype=float)
        if np.isnan(prediction).any():
            raise RuntimeError("Train-only month climatology has an unseen test month")
        rows.append(
            {
                "method": "train_month_climatology",
                "horizon": int(horizon),
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "mse": float(mean_squared_error(truth, prediction)),
                "mae": float(mean_absolute_error(truth, prediction)),
                "fit_scope": f"train_only_{train_end_idx}_days",
                "test_start": scored_dates.iloc[0].strftime("%Y-%m-%d"),
                "test_end": scored_dates.iloc[-1].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/mnt/sdc/zoujunjie"))
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/sdc/zoujunjie/mlearn_power_coursework"))
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--split-ratio", type=float, default=0.65)
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument(
        "--pvg-time-pooling",
        choices=["mean", "last"],
        default="last",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    unknown_models = sorted(set(args.models) - set(DEFAULT_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    configure_environment(args.base)
    environment_info = dependency_environment_metadata()
    warn_about_user_site(environment_info)
    processed_dir = (
        args.processed_dir
        or args.work_dir / "data" / "processed_causal_asof_lag1_v2"
    )
    run_dir = (
        args.run_dir
        or args.work_dir / "runs" / "fixed_holdout_asof_lag1_v2"
    )
    preexisting_run_entries = list(run_dir.iterdir()) if run_dir.exists() else []
    for path in [
        args.work_dir / "data" / "raw",
        args.work_dir / "data" / "weather",
        processed_dir,
        run_dir / "results",
        run_dir / "figures",
        run_dir / "logs",
        run_dir / "checkpoints",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    runs_path = run_dir / "results" / "metrics_runs.csv"
    metadata_path = run_dir / "results" / "run_metadata.json"
    if args.resume:
        if not (runs_path.exists() and metadata_path.exists()):
            raise RuntimeError(
                "Cannot --resume: metrics_runs.csv and signed run_metadata.json must both "
                "already exist. Choose a completed/partial signed run directory, or start "
                "without --resume in a new empty --run-dir."
            )
    elif preexisting_run_entries:
        entry_names = sorted(path.name for path in preexisting_run_entries)
        raise FileExistsError(
            f"Refusing to start a new experiment in non-empty {run_dir}; existing entries: "
            f"{entry_names}. Choose a new empty --run-dir, or use --resume only for a "
            "compatible signed run."
        )

    raw_txt = download_uci(args.work_dir / "data" / "raw")
    weather_path = download_weather(args.work_dir / "data" / "weather")
    daily = build_daily_frame(
        raw_txt,
        weather_path,
        processed_dir,
        args.split_ratio,
        args.validation_days,
        rebuild=args.rebuild_data,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    device_info = device_metadata(device)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    data_paths = {
        "raw_power": raw_txt,
        "weather": weather_path,
        "daily_power": processed_dir / "daily_power.csv",
        "test_split": processed_dir / "test.csv",
        "split_manifest": processed_dir / "split_manifest.json",
        "weather_source_mapping": processed_dir / "weather_monthly_source_mapping.csv",
    }
    data_hashes = {name: sha256_file(path) for name, path in data_paths.items()}
    experiment_signature = build_experiment_signature(
        args,
        script_hash=script_hash,
        data_hashes=data_hashes,
        device_info=device_info,
        environment_info=environment_info,
    )
    if args.resume and runs_path.exists():
        validate_resume_signature(metadata_path, experiment_signature)

    if runs_path.exists():
        previous_results = pd.read_csv(runs_path)
        rows: list[dict] = previous_results.to_dict("records")
    else:
        rows = []
    required_protocols = {EVALUATION_PROTOCOL}
    completed: set[tuple[str, int, int]] = set()
    if rows:
        existing = pd.DataFrame(rows)
        for key, group in existing.groupby(["model_name", "horizon", "seed"]):
            if required_protocols.issubset(set(group["evaluation_protocol"])):
                completed.add((str(key[0]), int(key[1]), int(key[2])))

    metadata: dict[str, dict] = {
        "script": {
            "path": str(script_path),
            "sha256": script_hash,
        },
        "preprocessing_version": PREPROCESSING_VERSION,
        "experiment_signature": experiment_signature,
        "environment": environment_info,
        "data": {
            "raw_txt": str(raw_txt),
            "processed_dir": str(processed_dir),
            "data_paths": {name: str(path) for name, path in data_paths.items()},
            "data_sha256": data_hashes,
            "weather_url": WEATHER_URL,
            "weather_path": str(weather_path),
            "weather_department": "92 Hauts-de-Seine",
            "weather_station_id": WEATHER_STATION_ID,
            "weather_station_name": WEATHER_STATION_NAME,
            "weather_station_distance_km": WEATHER_STATION_DISTANCE_KM,
            "weather_cols": WEATHER_COLS,
            "weather_column_units": {
                "RR": "mm",
                "NBJRR1": "days",
                "NBJRR5": "days",
                "NBJRR10": "days",
            },
            "weather_lag_months": WEATHER_LAG_MONTHS,
            "weather_source_mapping": str(
                processed_dir / "weather_monthly_source_mapping.csv"
            ),
            "weather_excluded_cols": {
                "NBJBROU": "not used because the selected SURESNES station has 47 missing values out of 48 months"
            },
            "weather_merge": (
                "Strict as-of lag-1 merge: each target calendar month uses SURESNES "
                "precipitation values from the immediately preceding completed calendar "
                "month. target_month/source_month are saved explicitly; RR is stored in mm."
            ),
            "missing_imputation": (
                "Strictly causal minute-level mean from original same-month observations at "
                "t-7/t-14/t-21/t-28 days; fallback uses the most recent earlier raw minute. "
                "Filled values are never reused as donors."
            ),
            "daily_rows": int(len(daily)),
            "split_strategy": "chronological self-split",
            "train_plus_validation_ratio": args.split_ratio,
            "validation_days": args.validation_days,
            "validation_protocol": "fixed_holdout",
            "pvg_time_pooling": args.pvg_time_pooling,
            "input_len": args.input_len,
            "models": args.models,
        },
        "evaluation_protocols": {
            "fixed_holdout": (
                "One forecast at the fixed test boundary; no later test observations are "
                "used as inputs."
            ),
        },
        "device": {
            **device_info,
            "torch": torch.__version__,
        },
        "run_dir": str(run_dir),
        "arguments": {
            "input_len": args.input_len,
            "split_ratio": args.split_ratio,
            "validation_days": args.validation_days,
            "pvg_time_pooling": args.pvg_time_pooling,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": args.seeds,
            "horizons": args.horizons,
            "models": args.models,
        },
        "runs": rows,
    }
    for horizon in args.horizons:
        train_ds, validation_ds, test_sets, meta = make_windows(
            daily,
            input_len=args.input_len,
            output_len=horizon,
            split_ratio=args.split_ratio,
            validation_days=args.validation_days,
        )
        metadata[f"horizon_{horizon}"] = meta
        input_dim = len(meta["feature_cols"])
        for model_name in args.models:
            for seed in args.seeds:
                run_key = (model_name, int(horizon), int(seed))
                if run_key in completed:
                    print(
                        f"Skipping completed run model={model_name} "
                        f"horizon={horizon} seed={seed}"
                    )
                    continue
                cfg = TrainConfig(
                    model_name=model_name,
                    horizon=horizon,
                    seed=seed,
                    input_len=args.input_len,
                    validation_protocol=EVALUATION_PROTOCOL,
                    pvg_time_pooling=args.pvg_time_pooling,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )
                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"{model_name}_h{horizon}_seed{seed}_best_validation.pt"
                )
                run_results, predictions = train_one(
                    cfg,
                    train_ds,
                    validation_ds,
                    test_sets,
                    input_dim=input_dim,
                    feature_cols=meta["feature_cols"],
                    target_mean=meta["target_mean"],
                    target_std=meta["target_std"],
                    device=device,
                    checkpoint_path=checkpoint_path,
                )
                rows.extend(run_results)
                metadata["runs"] = rows
                pd.DataFrame(rows).to_csv(runs_path, index=False)
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                if seed == args.seeds[0]:
                    for protocol, (pred, true) in predictions.items():
                        fig_name = f"{model_name}_h{horizon}_{protocol}_prediction.png"
                        protocol_title = protocol.replace("_", " ").title()
                        plot_prediction(
                            pred,
                            true,
                            run_dir / "figures" / fig_name,
                            title=(
                                f"{model_display_name(model_name)} {horizon}-day "
                                f"forecast ({protocol_title})"
                            ),
                        )
                        test_ds = test_sets[protocol]
                        origin_indices = test_ds.starts + args.input_len
                        origin_dates = (
                            pd.to_datetime(daily["date"])
                            .iloc[origin_indices]
                            .dt.strftime("%Y-%m-%d")
                            .to_numpy()
                        )
                        np.savez(
                            run_dir
                            / "results"
                            / f"{model_name}_h{horizon}_seed{seed}_{protocol}_predictions.npz",
                            prediction=pred,
                            ground_truth=true,
                            origin_date=origin_dates,
                        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.drop_duplicates(
        subset=["model_name", "horizon", "seed", "evaluation_protocol"],
        keep="last",
    )
    summary_df = summarize(results_df.to_dict("records"))
    results_df.to_csv(runs_path, index=False)
    summary_df.to_csv(run_dir / "results" / "metrics_summary.csv", index=False)
    baseline_df = train_only_month_climatology(
        daily,
        split_ratio=args.split_ratio,
        validation_days=args.validation_days,
        horizons=list(args.horizons),
    )
    baseline_df.to_csv(run_dir / "results" / "baseline_metrics.csv", index=False)
    metadata["runs"] = results_df.to_dict("records")
    metadata["completed_training_runs"] = int(
        results_df[["model_name", "horizon", "seed"]].drop_duplicates().shape[0]
    )
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
