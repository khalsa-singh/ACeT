#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate current fixed-head robustness sources for Supplementary Figures S1-S4.

This orchestrator does not implement a second modeling pipeline.  It:

1. regenerates the composite-stratified train/test partitions for split seeds
   0-5 from the canonical full-order DataS1/DataS2 tables;
2. verifies that seed 0 reproduces the already locked manuscript split;
3. uses the locked seed-0 predictions supplied in the clean package
   under the endpoint primary-result folders;
4. runs the existing existing pooling scripts with only
   GlobalAveragePooling1D and the already selected endpoint head for seeds 1-5:
      * viscosity: KAN
      * mouse clearance: spline
5. consolidates the held-out predictions and metrics into simple CSV sources
   consumed by ``generate_all_supplementary_figures.m``.

No manuscript or Supplement file is edited.  Existing outputs are not deleted.
The run is resumable at the seed level.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "2026-08-13.1"
SEEDS = tuple(range(6))


class ControlledStop(RuntimeError):
    """A deliberate stop with a user-readable reason."""


@dataclass(frozen=True)
class Endpoint:
    name: str
    prefix: str
    target: str
    test_size: float
    selected_head: str
    synthetic_rows: int
    full_order_rel: str
    seed0_train_rel: str
    seed0_test_rel: str
    pooling_script_rel: str
    output_prefix: str
    anchor_kind: str
    anchor_rel: str
    expected_seed0_r2: float
    expected_seed0_n: int


ENDPOINTS = (
    Endpoint(
        name="viscosity",
        prefix="VIS",
        target="Viscosity",
        test_size=0.30,
        selected_head="KAN",
        synthetic_rows=26,
        full_order_rel="Data/robustness_splits/viscosity/DataS1_viscosity_antibodies.csv",
        seed0_train_rel="Data/fixed_splits/viscosity/DataS1_viscosity_seed0_train.csv",
        seed0_test_rel="Data/fixed_splits/viscosity/DataS1_viscosity_seed0_test.csv",
        pooling_script_rel="MainPack/shared/pooling_ablation/viscosity_pooling_ablation.py",
        output_prefix="viscosity_full",
        anchor_kind="csv",
        anchor_rel="MainPack/viscosity/results/primary/viscosity_predictions.csv",
        expected_seed0_r2=0.740886,
        expected_seed0_n=23,
    ),
    Endpoint(
        name="clearance",
        prefix="CLR",
        target="AUCt",
        test_size=0.20,
        selected_head="spline",
        synthetic_rows=84,
        full_order_rel="Data/robustness_splits/mouse_exposure/DataS2_clearance_antibodies.csv",
        seed0_train_rel="Data/fixed_splits/mouse_exposure/DataS2_clearance_seed0_train.csv",
        seed0_test_rel="Data/fixed_splits/mouse_exposure/DataS2_clearance_seed0_test.csv",
        pooling_script_rel="MainPack/shared/pooling_ablation/clearance_pooling_ablation.py",
        output_prefix="clearance_full",
        anchor_kind="mat",
        anchor_rel="MainPack/mouse_exposure/results/primary/clearance_panel3A_data.mat",
        expected_seed0_r2=0.810019,
        expected_seed0_n=11,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_package_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    env = os.environ.get("ACET_PACKAGE_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend((start, *start.parents))
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (root / "MANIFEST.csv").is_file() and (root / "Data").is_dir():
            return root
    raise ControlledStop(
        "Could not locate the ACeT package root. Pass --package-root or set "
        "ACET_PACKAGE_ROOT. The root must contain MANIFEST.csv and Data/."
    )


def load_split_module(package_root: Path):
    script = (
        package_root
        / "GenBench"
        / "robustness"
        / "split_sensitivity"
        / "code"
        / "source"
        / "acet_composite_stratified_split.py"
    )
    spec = importlib.util.spec_from_file_location("acet_split", script)
    if spec is None or spec.loader is None:
        raise ControlledStop(f"Could not import split generator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_model_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ControlledStop(f"Missing target column {target!r}")
    feature_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    result = df[feature_cols + [target]].copy()
    if result.isna().any().any():
        raise ControlledStop("Required split table contains missing numeric values.")
    return result


def rounded_row_keys(df: pd.DataFrame, decimals: int = 10) -> list[tuple[float, ...]]:
    values = np.round(df.to_numpy(float), decimals=decimals)
    return [tuple(row.tolist()) for row in values]


def map_rows_to_original(full: pd.DataFrame, subset: pd.DataFrame) -> list[int]:
    """Map a numeric subset to the original full-order row indices."""
    full_keys = rounded_row_keys(full)
    buckets: dict[tuple[float, ...], list[int]] = {}
    for idx, key in enumerate(full_keys):
        buckets.setdefault(key, []).append(idx)
    out: list[int] = []
    used: set[int] = set()
    for key in rounded_row_keys(subset):
        candidates = [idx for idx in buckets.get(key, []) if idx not in used]
        if not candidates:
            raise ControlledStop("Could not map a split row back to the canonical full-order table.")
        idx = candidates[0]
        used.add(idx)
        out.append(idx)
    return out


def same_row_multiset(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return sorted(rounded_row_keys(left)) == sorted(rounded_row_keys(right))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_true = float(np.mean(y_true))
    spearman = spearmanr(y_true, y_pred)
    rho = spearman.statistic if hasattr(spearman, "statistic") else spearman[0]
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / mean_true if mean_true != 0 else float("nan"),
        "nmae": mae / mean_true if mean_true != 0 else float("nan"),
        "spearman": float(rho),
    }


def jackknife_max_delta_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    base = float(r2_score(y_true, y_pred))
    values = []
    for idx in range(len(y_true)):
        keep = np.ones(len(y_true), dtype=bool)
        keep[idx] = False
        values.append(abs(float(r2_score(y_true[keep], y_pred[keep])) - base))
    return float(max(values))


def read_seed0_anchor(package_root: Path, endpoint: Endpoint) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    current_test = pd.read_csv(package_root / endpoint.seed0_test_rel)
    current_test_numeric = numeric_model_frame(current_test, endpoint.target)
    anchor_path = package_root / endpoint.anchor_rel
    if not anchor_path.exists():
        raise ControlledStop(f"Locked seed-0 anchor is missing: {anchor_path}")
    if endpoint.anchor_kind == "csv":
        anchor = pd.read_csv(anchor_path)
        expected_cols = {"True_Viscosity", "Predicted_Viscosity"}
        if not expected_cols.issubset(anchor.columns):
            raise ControlledStop(f"Unexpected viscosity anchor schema: {anchor.columns.tolist()}")
        y_true = anchor["True_Viscosity"].to_numpy(float)
        y_pred = anchor["Predicted_Viscosity"].to_numpy(float)
    else:
        mat = loadmat(anchor_path)
        if "trues" not in mat or "preds" not in mat:
            raise ControlledStop(f"Unexpected clearance anchor MAT fields: {sorted(mat.keys())}")
        y_true = np.asarray(mat["trues"], dtype=float).reshape(-1)
        y_pred = np.asarray(mat["preds"], dtype=float).reshape(-1)
    if len(y_true) != endpoint.expected_seed0_n or len(y_pred) != endpoint.expected_seed0_n:
        raise ControlledStop(
            f"{endpoint.name}: expected {endpoint.expected_seed0_n} anchor predictions; "
            f"found {len(y_true)}/{len(y_pred)}"
        )
    if not np.allclose(y_true, current_test_numeric[endpoint.target].to_numpy(float), atol=1e-10, rtol=0):
        raise ControlledStop(f"{endpoint.name}: anchor true values do not match the current seed-0 test table order.")
    metrics = metric_dict(y_true, y_pred)
    if abs(metrics["r2"] - endpoint.expected_seed0_r2) > 2e-4:
        raise ControlledStop(
            f"{endpoint.name}: locked seed-0 R2 mismatch. Expected about {endpoint.expected_seed0_r2}; "
            f"found {metrics['r2']:.9f}."
        )
    return current_test_numeric, y_true, y_pred


def split_for_seed(
    split_module: Any,
    full: pd.DataFrame,
    endpoint: Endpoint,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    X, y, _ = split_module._ensure_numeric_matrix(full, target_col=endpoint.target)
    best = split_module.find_best_params(
        X,
        y,
        seed=seed,
        cluster_list=[3, 4, 5, 6, 7, 8],
        bin_list=[1, 2, 3, 4, 5],
        test_size=endpoint.test_size,
    )
    tr_idx, te_idx, _, _, _ = split_module.composite_stratified_split(
        X,
        y,
        test_size=endpoint.test_size,
        n_clusters=best["n_clusters"],
        n_bins=best["n_bins"],
        seed=seed,
    )
    return (
        full.iloc[tr_idx].reset_index(drop=True),
        full.iloc[te_idx].reset_index(drop=True),
        best,
        np.asarray(tr_idx, dtype=int),
        np.asarray(te_idx, dtype=int),
    )


def run_pooling_script(
    package_root: Path,
    endpoint: Endpoint,
    seed: int,
    train_csv: Path,
    test_csv: Path,
    seed_dir: Path,
    force: bool,
) -> Path:
    script = package_root / endpoint.pooling_script_rel
    result_csv = seed_dir / f"{endpoint.output_prefix}_predictions.csv"
    signature_path = seed_dir / "ORCHESTRATOR_COMPLETE.json"
    signature = {
        "script_version": SCRIPT_VERSION,
        "endpoint": endpoint.name,
        "seed": seed,
        "head": endpoint.selected_head,
        "pooling": "avg",
        "train_sha256": sha256_file(train_csv),
        "test_sha256": sha256_file(test_csv),
        "producer_sha256": sha256_file(script),
    }
    if not force and result_csv.exists() and signature_path.exists():
        old = json.loads(signature_path.read_text(encoding="utf-8"))
        if old == signature:
            print(f"[reuse] {endpoint.name} seed {seed}: {result_csv}")
            return result_csv
    if seed_dir.exists():
        for child in seed_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--full",
        "--pooling",
        "avg",
        "--train_file",
        str(train_csv),
        "--test_file",
        str(test_csv),
        "--output_dir",
        str(seed_dir),
        "--seed",
        str(seed),
        "--verbose",
        "0",
    ]
    print("[run] " + " ".join(command))
    completed = subprocess.run(command, cwd=script.parent, check=False)
    if completed.returncode != 0:
        raise ControlledStop(
            f"{endpoint.name} seed {seed} producer returned exit code {completed.returncode}. "
            f"Inspect {seed_dir / 'RUN_LOG.txt'}."
        )
    if not result_csv.exists():
        raise ControlledStop(f"Expected result was not created: {result_csv}")
    write_json(signature_path, signature)
    return result_csv


def parse_pooling_predictions(path: Path, endpoint: Endpoint, expected_n: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"pooling", "split", "true", "predicted", "row_index"}
    if not required.issubset(frame.columns):
        raise ControlledStop(f"Unexpected pooling output schema in {path}: {frame.columns.tolist()}")
    subset = frame[(frame["pooling"] == "avg") & (frame["split"] == "heldout_test")].copy()
    subset = subset.sort_values("row_index")
    if len(subset) != expected_n:
        raise ControlledStop(f"{endpoint.name}: expected {expected_n} held-out rows; found {len(subset)}")
    return subset["true"].to_numpy(float), subset["predicted"].to_numpy(float)


def write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_outputs(output_dir: Path, prediction_frames: dict[str, pd.DataFrame], metric_frames: dict[str, pd.DataFrame]) -> None:
    rows: list[dict[str, Any]] = []
    for endpoint_name, predictions in prediction_frames.items():
        metrics = metric_frames[endpoint_name]
        row: dict[str, Any] = {
            "endpoint": endpoint_name,
            "split_seeds": "0-5",
            "n_splits": int(metrics["split_seed"].nunique()),
            "n_pooled_test_predictions": int(len(predictions)),
            "median_test_r2": float(metrics["r2"].median()),
            "p25_test_r2": float(metrics["r2"].quantile(0.25)),
            "p75_test_r2": float(metrics["r2"].quantile(0.75)),
            "median_test_spearman": float(metrics["spearman"].median()),
            "median_test_nrmse": float(metrics["nrmse"].median()),
        }
        if endpoint_name == "viscosity":
            logerr = np.abs(np.log10(predictions["predicted_value"]) - np.log10(predictions["true_value"]))
            row.update(
                {
                    "pooled_median_abs_log10_error": float(np.median(logerr)),
                    "pooled_p75_abs_log10_error": float(np.percentile(logerr, 75)),
                    "pooled_p90_abs_log10_error": float(np.percentile(logerr, 90)),
                    "pooled_within_2x_pct": float(np.mean(logerr <= math.log10(2)) * 100),
                    "pooled_within_3x_pct": float(np.mean(logerr <= math.log10(3)) * 100),
                }
            )
        else:
            ape = np.abs((predictions["predicted_value"] - predictions["true_value"]) / predictions["true_value"]) * 100
            row.update(
                {
                    "pooled_median_absolute_percent_error": float(np.median(ape)),
                    "pooled_p75_absolute_percent_error": float(np.percentile(ape, 75)),
                    "pooled_p90_absolute_percent_error": float(np.percentile(ape, 90)),
                    "pooled_within_15pct": float(np.mean(ape <= 15) * 100),
                    "pooled_within_30pct": float(np.mean(ape <= 30) * 100),
                    "median_max_jackknife_delta_r2": float(metrics["max_jackknife_delta_r2"].median()),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "alternate_partition_summary.csv", index=False)

    table13_rows = []
    for row in rows:
        if row["endpoint"] == "viscosity":
            text = (
                f"Median test R² = {row['median_test_r2']:.3f}; median Spearman ρ = "
                f"{row['median_test_spearman']:.3f}; pooled median absolute log10 error = "
                f"{row['pooled_median_abs_log10_error']:.3f}; P75 = "
                f"{row['pooled_p75_abs_log10_error']:.3f}; "
                f"{row['pooled_within_2x_pct']:.1f}% within 2-fold and "
                f"{row['pooled_within_3x_pct']:.1f}% within 3-fold error."
            )
        else:
            text = (
                f"Median test R² = {row['median_test_r2']:.3f} (P25-P75: "
                f"{row['p25_test_r2']:.3f}-{row['p75_test_r2']:.3f}); median Spearman ρ = "
                f"{row['median_test_spearman']:.3f}; median nRMSE = {row['median_test_nrmse']:.3f}; "
                f"pooled median absolute percentage error = "
                f"{row['pooled_median_absolute_percent_error']:.1f}%."
            )
        table13_rows.append(
            {
                "Endpoint": "Viscosity" if row["endpoint"] == "viscosity" else "Mouse clearance",
                "Repeat basis": "Six composite train/test splits (seeds 0-5); fixed submitted head and current pipeline",
                "Summary": text,
            }
        )
    pd.DataFrame(table13_rows).to_csv(output_dir / "ALTERNATE_PARTITION_DESCRIPTIVE_ROWS.csv", index=False)

    md = [
        "# Current fixed-head robustness summary for Supplementary Figures S1-S4",
        "",
        "Seed 0 uses the locked manuscript predictions. Seeds 1-5 use the existing existing pooling producer with average pooling and the endpoint-specific CV-selected head fixed.",
        "",
    ]
    try:
        md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        md.extend(["```csv", summary.to_csv(index=False).rstrip(), "```"])
    (output_dir / "ALTERNATE_PARTITION_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def preflight(package_root: Path, output_dir: Path) -> dict[str, Any]:
    split_script = package_root / "GenBench" / "robustness" / "split_sensitivity" / "code" / "source" / "acet_composite_stratified_split.py"
    required = [split_script]
    endpoint_report = []
    split_module = load_split_module(package_root)
    for endpoint in ENDPOINTS:
        full_path = package_root / endpoint.full_order_rel
        train_path = package_root / endpoint.seed0_train_rel
        test_path = package_root / endpoint.seed0_test_rel
        pool_path = package_root / endpoint.pooling_script_rel
        anchor_path = package_root / endpoint.anchor_rel
        required.extend([full_path, train_path, test_path, pool_path, anchor_path])
        if any(not path.exists() for path in [full_path, train_path, test_path, pool_path, anchor_path]):
            continue
        full = pd.read_csv(full_path)
        generated_train, generated_test, best, _, _ = split_for_seed(split_module, full, endpoint, 0)
        current_train = numeric_model_frame(pd.read_csv(train_path), endpoint.target)
        current_test = numeric_model_frame(pd.read_csv(test_path), endpoint.target)
        if not same_row_multiset(generated_train, current_train) or not same_row_multiset(generated_test, current_test):
            raise ControlledStop(f"{endpoint.name}: seed-0 split generator did not reproduce the locked current split.")
        _, y_true, y_pred = read_seed0_anchor(package_root, endpoint)
        metrics = metric_dict(y_true, y_pred)
        endpoint_report.append(
            {
                "endpoint": endpoint.name,
                "seed0_best_clusters": best["n_clusters"],
                "seed0_best_bins": best["n_bins"],
                "seed0_train_n": len(generated_train),
                "seed0_test_n": len(generated_test),
                "seed0_r2": metrics["r2"],
                "selected_head": endpoint.selected_head,
                "synthetic_rows": endpoint.synthetic_rows,
            }
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ControlledStop("Required files are missing:\n" + "\n".join(missing))
    report = {
        "script_version": SCRIPT_VERSION,
        "package_root": str(package_root),
        "python": sys.version,
        "endpoints": endpoint_report,
        "status": "passed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "PREFLIGHT_REPORT.json", report)
    return report


def run_all(package_root: Path, output_dir: Path, force: bool) -> None:
    split_module = load_split_module(package_root)
    all_prediction_frames: dict[str, pd.DataFrame] = {}
    all_metric_frames: dict[str, pd.DataFrame] = {}
    split_manifest_rows: list[dict[str, Any]] = []

    for endpoint in ENDPOINTS:
        print(f"\n=== {endpoint.name.upper()} robustness ===")
        full_path = package_root / endpoint.full_order_rel
        full = pd.read_csv(full_path)
        full_numeric = numeric_model_frame(full, endpoint.target)
        endpoint_prediction_rows: list[dict[str, Any]] = []
        endpoint_metric_rows: list[dict[str, Any]] = []

        for seed in SEEDS:
            train, test, best, tr_idx, te_idx = split_for_seed(split_module, full, endpoint, seed)
            split_dir = output_dir / "splits" / endpoint.name / f"seed_{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            train_csv = split_dir / "train.csv"
            test_csv = split_dir / "test.csv"
            train.to_csv(train_csv, index=False)
            test.to_csv(test_csv, index=False)
            write_json(
                split_dir / "split_meta.json",
                {
                    "endpoint": endpoint.name,
                    "seed": seed,
                    "n_clusters": best["n_clusters"],
                    "n_bins": best["n_bins"],
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_indices": tr_idx.tolist(),
                    "test_indices": te_idx.tolist(),
                },
            )
            split_manifest_rows.append(
                {
                    "endpoint": endpoint.name,
                    "split_seed": seed,
                    "n_clusters": best["n_clusters"],
                    "n_bins": best["n_bins"],
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_csv": str(train_csv.relative_to(package_root)),
                    "test_csv": str(test_csv.relative_to(package_root)),
                    "selected_head": endpoint.selected_head,
                    "synthetic_rows": endpoint.synthetic_rows,
                }
            )

            test_orig_indices = list(map(int, te_idx))
            if seed == 0:
                current_test, y_true, y_pred = read_seed0_anchor(package_root, endpoint)
                current_orig_indices = map_rows_to_original(full_numeric, current_test)
                if sorted(current_orig_indices) != sorted(test_orig_indices):
                    raise ControlledStop(f"{endpoint.name}: seed-0 anchor rows do not match generated seed-0 test membership.")
                # Anchor order follows current seed-0 test table, not necessarily generated split order.
                orig_indices = current_orig_indices
                source = "locked_seed0_anchor"
            else:
                seed_run_dir = output_dir / "runs" / endpoint.name / f"seed_{seed}"
                result_csv = run_pooling_script(package_root, endpoint, seed, train_csv, test_csv, seed_run_dir, force)
                y_true, y_pred = parse_pooling_predictions(result_csv, endpoint, len(test))
                if not np.allclose(y_true, test[endpoint.target].to_numpy(float), atol=1e-10, rtol=0):
                    raise ControlledStop(f"{endpoint.name} seed {seed}: producer true values do not match generated test table.")
                orig_indices = test_orig_indices
                source = "current_fixed_head_rerun"

            metrics = metric_dict(y_true, y_pred)
            max_jack = jackknife_max_delta_r2(y_true, y_pred)
            endpoint_metric_rows.append(
                {
                    "endpoint": endpoint.name,
                    "split_seed": seed,
                    "n_test": len(y_true),
                    **metrics,
                    "max_jackknife_delta_r2": max_jack,
                    "selected_head": endpoint.selected_head,
                    "source": source,
                }
            )
            for row_number, (orig_idx, true, pred) in enumerate(zip(orig_indices, y_true, y_pred)):
                endpoint_prediction_rows.append(
                    {
                        "endpoint": endpoint.name,
                        "split_seed": seed,
                        "row_in_test": row_number,
                        "row_id": f"{endpoint.prefix}_ORIG_{orig_idx:03d}",
                        "original_row_index": orig_idx,
                        "true_value": float(true),
                        "predicted_value": float(pred),
                        "is_seed0_anchor": seed == 0,
                        "selected_head": endpoint.selected_head,
                        "source": source,
                    }
                )
            print(
                f"[{endpoint.name} seed {seed}] n={len(y_true)} R2={metrics['r2']:.4f} "
                f"RMSE={metrics['rmse']:.4f} Spearman={metrics['spearman']:.4f}"
            )

        pred_frame = pd.DataFrame(endpoint_prediction_rows)
        metric_frame = pd.DataFrame(endpoint_metric_rows)
        pred_frame.to_csv(output_dir / (("viscosity" if endpoint.name == "viscosity" else "mouse_exposure") + "_alternate_partition_predictions.csv"), index=False)
        metric_frame.to_csv(output_dir / (("viscosity" if endpoint.name == "viscosity" else "mouse_exposure") + "_alternate_partition_metrics.csv"), index=False)
        if endpoint.name == "clearance":
            metric_frame[["split_seed", "r2", "max_jackknife_delta_r2"]].to_csv(
                output_dir / "clearance_jackknife.csv", index=False
            )
        all_prediction_frames[endpoint.name] = pred_frame
        all_metric_frames[endpoint.name] = metric_frame

    pd.DataFrame(split_manifest_rows).to_csv(output_dir / "SPLIT_MANIFEST.csv", index=False)
    summarize_outputs(output_dir, all_prediction_frames, all_metric_frames)
    write_checksums(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "run", "summarize"], default="preflight")
    parser.add_argument("--package-root", "--repo-root", dest="package_root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = discover_package_root(args.package_root)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else package_root / "ReproducedOutputs" / "robustness" / "split_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "RUN_LOG.txt"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] mode={args.mode} script={SCRIPT_VERSION}\n")
    try:
        report = preflight(package_root, output_dir)
        print(json.dumps(report, indent=2))
        if args.mode == "run":
            run_all(package_root, output_dir, args.force)
        elif args.mode == "summarize":
            pred_frames = {
                ep.name: pd.read_csv(output_dir / (("viscosity" if ep.name == "viscosity" else "mouse_exposure") + "_alternate_partition_predictions.csv")) for ep in ENDPOINTS
            }
            metric_frames = {
                ep.name: pd.read_csv(output_dir / (("viscosity" if ep.name == "viscosity" else "mouse_exposure") + "_alternate_partition_metrics.csv")) for ep in ENDPOINTS
            }
            summarize_outputs(output_dir, pred_frames, metric_frames)
            write_checksums(output_dir)
        return 0
    except ControlledStop as exc:
        message = str(exc)
        print("CONTROLLED STOP:\n" + message, file=sys.stderr)
        (output_dir / "CONTROLLED_STOP.txt").write_text(message + "\n", encoding="utf-8")
        return 2
    except Exception as exc:  # pragma: no cover - user environment dependent
        import traceback

        text = traceback.format_exc()
        print(text, file=sys.stderr)
        (output_dir / "UNEXPECTED_ERROR.txt").write_text(text, encoding="utf-8")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
