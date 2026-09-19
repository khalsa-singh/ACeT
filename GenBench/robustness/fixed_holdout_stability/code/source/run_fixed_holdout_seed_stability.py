#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-holdout multi-training-seed stability for MLHealth regression endpoints.

Scientific scope
----------------
This  producer complements the existing six composite-split
sensitivity analysis.  It keeps the manuscript train/test partitions fixed and
changes only the model/training random seed:

* viscosity: reuses the five current Full-condition runs already generated for
  Figure 2e (seeds 0-4; fixed 52/23 split; KAN head);
* mouse exposure: runs the existing current clearance producer five times on
  the unchanged 42/11 manuscript split (seeds 0-4; spline head; average
  pooling).

  It does not select heads on the
held-out test set.  It preserves the separate six-split sensitivity outputs and
uses those only for descriptive error/parity panels without aggregating R^2
across changing test cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_VERSION = "2026-08-13.1"
SEEDS = tuple(range(5))


class ControlledStop(RuntimeError):
    """Deliberate stop with a user-readable reason."""


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


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_true = float(np.mean(y_true))
    rho_result = spearmanr(y_true, y_pred)
    rho = rho_result.statistic if hasattr(rho_result, "statistic") else rho_result[0]
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / mean_true if mean_true != 0 else float("nan"),
        "nmae": mae / mean_true if mean_true != 0 else float("nan"),
        "spearman": float(rho),
    }


def summarize_metric_frame(frame: pd.DataFrame, endpoint: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "endpoint": endpoint,
        "n_training_seeds": int(frame["training_seed"].nunique()),
        "seed_range": f"{int(frame['training_seed'].min())}-{int(frame['training_seed'].max())}",
    }
    for metric in ("r2", "rmse", "mae", "nrmse", "nmae", "spearman"):
        values = frame[metric].astype(float)
        row[f"mean_{metric}"] = float(values.mean())
        row[f"sd_{metric}"] = float(values.std(ddof=1))
        row[f"min_{metric}"] = float(values.min())
        row[f"max_{metric}"] = float(values.max())
    return row


def require_same_fixed_holdout(predictions: pd.DataFrame, id_col: str, true_col: str) -> None:
    seeds = sorted(predictions["training_seed"].unique().tolist())
    if seeds != list(SEEDS):
        raise ControlledStop(f"Expected fixed-holdout seeds {list(SEEDS)}; found {seeds}")
    base = (
        predictions[predictions["training_seed"] == seeds[0]][[id_col, true_col]]
        .sort_values(id_col)
        .reset_index(drop=True)
    )
    for seed in seeds[1:]:
        other = (
            predictions[predictions["training_seed"] == seed][[id_col, true_col]]
            .sort_values(id_col)
            .reset_index(drop=True)
        )
        if len(other) != len(base):
            raise ControlledStop(f"Seed {seed}: fixed holdout row count differs from seed 0.")
        if base[id_col].astype(str).tolist() != other[id_col].astype(str).tolist():
            raise ControlledStop(f"Seed {seed}: held-out antibody identities differ from seed 0.")
        if not np.allclose(base[true_col].to_numpy(float), other[true_col].to_numpy(float), atol=1e-10, rtol=0):
            raise ControlledStop(f"Seed {seed}: held-out true values differ from seed 0.")


def load_viscosity_fixed(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_dir = package_root / "MainPack" / "viscosity" / "results" / "figure2de"
    pred_path = final_dir / "figure2e_ablation_predictions.csv"
    runs_path = final_dir / "figure2e_ablation_runs.csv"
    if not pred_path.exists() or not runs_path.exists():
        raise ControlledStop(
            "Current viscosity five-seed Full-condition outputs are missing. Expected:\n"
            f"{pred_path}\n{runs_path}"
        )
    pred = pd.read_csv(pred_path)
    runs = pd.read_csv(runs_path)
    required_pred = {"mAb_id", "condition", "seed", "true_viscosity", "predicted_viscosity"}
    required_runs = {"condition", "seed", "r2", "rmse", "mae", "spearman"}
    if not required_pred.issubset(pred.columns) or not required_runs.issubset(runs.columns):
        raise ControlledStop("Unexpected viscosity Figure 2e output schema.")
    pred = pred[(pred["condition"] == "full") & (pred["seed"].isin(SEEDS))].copy()
    runs = runs[(runs["condition"] == "full") & (runs["seed"].isin(SEEDS))].copy()
    pred = pred.rename(
        columns={
            "seed": "training_seed",
            "true_viscosity": "true_value",
            "predicted_viscosity": "predicted_value",
        }
    )
    pred["endpoint"] = "viscosity"
    pred["source"] = "current_figure2e_full_condition"
    pred = pred[["endpoint", "training_seed", "mAb_id", "true_value", "predicted_value", "source"]]
    if len(pred) != 5 * 23 or len(runs) != 5:
        raise ControlledStop(f"Viscosity fixed-holdout outputs should contain 115 predictions and 5 runs; found {len(pred)} and {len(runs)}.")
    require_same_fixed_holdout(pred, "mAb_id", "true_value")
    metric_rows = []
    for seed, group in pred.groupby("training_seed", sort=True):
        metrics = metric_dict(group["true_value"].to_numpy(float), group["predicted_value"].to_numpy(float))
        metric_rows.append({"endpoint": "viscosity", "training_seed": int(seed), "n_test": len(group), **metrics})
    metrics = pd.DataFrame(metric_rows)
    # Verify against the already saved run-level metrics.
    merged = metrics.merge(runs[["seed", "r2", "rmse", "mae", "spearman"]], left_on="training_seed", right_on="seed", suffixes=("", "_saved"))
    for name in ("r2", "rmse", "mae", "spearman"):
        if not np.allclose(merged[name], merged[f"{name}_saved"], atol=5e-6, rtol=0):
            raise ControlledStop(f"Viscosity recomputed {name} does not match the current saved Full-condition runs.")
    return pred, metrics


def clearance_input_tables(package_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    train_source = package_root / "Data" / "fixed_splits" / "mouse_exposure" / "DataS2_clearance_seed0_train.csv"
    test_source = package_root / "Data" / "fixed_splits" / "mouse_exposure" / "DataS2_clearance_seed0_test.csv"
    if not train_source.exists() or not test_source.exists():
        raise ControlledStop(f"Current clearance seed-0 train/test files are missing: {train_source}, {test_source}")
    train_raw = pd.read_csv(train_source)
    test_raw = pd.read_csv(test_source)
    cols = ["Heparin_RT", "Heparin_pB_buffer", "BVP_high", "poly_D_lysine", "AUCt"]
    missing = [c for c in cols if c not in train_raw.columns or c not in test_raw.columns]
    if missing:
        raise ControlledStop(f"Clearance fixed-holdout files are missing required columns: {missing}")
    train_numeric = train_raw[cols].apply(pd.to_numeric, errors="raise")
    test_numeric = test_raw[cols].apply(pd.to_numeric, errors="raise")
    if len(train_numeric) != 42 or len(test_numeric) != 11:
        raise ControlledStop(f"Expected clearance fixed split 42/11; found {len(train_numeric)}/{len(test_numeric)}")
    ids = test_raw["mAb_id"].astype(str) if "mAb_id" in test_raw.columns else pd.Series([f"CLR_TEST_{i:03d}" for i in range(len(test_raw))])
    inputs_dir = output_dir / "input_snapshots"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    train_path = inputs_dir / "clearance_fixed_train_numeric.csv"
    test_path = inputs_dir / "clearance_fixed_test_numeric.csv"
    train_numeric.to_csv(train_path, index=False)
    test_numeric.to_csv(test_path, index=False)
    test_meta = pd.DataFrame({"row_index": np.arange(len(test_raw)), "mAb_id": ids, "true_value": test_numeric["AUCt"]})
    test_meta.to_csv(inputs_dir / "clearance_fixed_test_row_map.csv", index=False)
    return train_numeric, test_meta, train_path, test_path


def run_clearance_seed(
    producer: Path,
    seed: int,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    force: bool,
) -> Path:
    seed_dir = output_dir / "runs" / "clearance" / f"seed_{seed}"
    result_path = seed_dir / "clearance_full_predictions.csv"
    signature_path = seed_dir / "COMPLETE_SIGNATURE.json"
    signature = {
        "script_version": SCRIPT_VERSION,
        "endpoint": "clearance",
        "training_seed": seed,
        "head": "spline",
        "pooling": "avg",
        "fixed_split": "DataS2 seed0 42/11",
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "producer_sha256": sha256_file(producer),
    }
    if not force and result_path.exists() and signature_path.exists():
        old = json.loads(signature_path.read_text(encoding="utf-8"))
        if old == signature:
            print(f"[reuse] clearance fixed holdout seed {seed}")
            return result_path
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(producer),
        "--full",
        "--pooling",
        "avg",
        "--train_file",
        str(train_path),
        "--test_file",
        str(test_path),
        "--output_dir",
        str(seed_dir),
        "--seed",
        str(seed),
        "--verbose",
        "0",
    ]
    print("[run] " + " ".join(command))
    completed = subprocess.run(command, cwd=producer.parent, check=False)
    if completed.returncode != 0:
        raise ControlledStop(
            f"Clearance fixed-holdout seed {seed} returned exit code {completed.returncode}. "
            f"Inspect {seed_dir / 'RUN_LOG.txt'}."
        )
    if not result_path.exists():
        raise ControlledStop(f"Expected clearance result was not created: {result_path}")
    write_json(signature_path, signature)
    return result_path


def parse_clearance_predictions(path: Path, seed: int, test_meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"pooling", "split", "row_index", "true", "predicted"}
    if not required.issubset(frame.columns):
        raise ControlledStop(f"Unexpected clearance pooling output schema in {path}: {frame.columns.tolist()}")
    held = frame[(frame["pooling"] == "avg") & (frame["split"] == "heldout_test")].copy()
    held["row_index"] = pd.to_numeric(held["row_index"], errors="raise").astype(int)
    held = held.sort_values("row_index").reset_index(drop=True)
    if len(held) != 11:
        raise ControlledStop(f"Clearance seed {seed}: expected 11 held-out rows; found {len(held)}")
    merged = held.merge(test_meta, on="row_index", how="left", validate="one_to_one")
    if merged["mAb_id"].isna().any():
        raise ControlledStop(f"Clearance seed {seed}: could not align held-out predictions to the fixed test IDs.")
    if not np.allclose(merged["true"].to_numpy(float), merged["true_value"].to_numpy(float), atol=1e-10, rtol=0):
        raise ControlledStop(f"Clearance seed {seed}: producer true values do not match the fixed test table.")
    pred = pd.DataFrame(
        {
            "endpoint": "clearance",
            "training_seed": seed,
            "mAb_id": merged["mAb_id"].astype(str),
            "true_value": merged["true"].astype(float),
            "predicted_value": merged["predicted"].astype(float),
            "source": "current_fixed_split_spline_avg_pooling",
        }
    )
    metrics = metric_dict(pred["true_value"].to_numpy(float), pred["predicted_value"].to_numpy(float))
    return pred, {"endpoint": "clearance", "training_seed": seed, "n_test": len(pred), **metrics}


def per_antibody_summary(pred: pd.DataFrame) -> pd.DataFrame:
    return (
        pred.groupby(["endpoint", "mAb_id", "true_value"], as_index=False)
        .agg(
            mean_predicted_value=("predicted_value", "mean"),
            sd_predicted_value=("predicted_value", lambda s: float(np.std(s, ddof=1))),
            min_predicted_value=("predicted_value", "min"),
            max_predicted_value=("predicted_value", "max"),
            n_training_seeds=("training_seed", "nunique"),
        )
    )


def split_sensitivity_rows(package_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    split_dir = package_root / "GenBench" / "robustness" / "split_sensitivity" / "results" / "locked"
    vis_path = split_dir / "viscosity_alternate_partition_predictions.csv"
    clr_path = split_dir / "mouse_exposure_alternate_partition_predictions.csv"
    if not vis_path.exists() or not clr_path.exists():
        raise ControlledStop(
            "Current six-split prediction outputs are missing. Run the existing supplementary robustness producer first:\n"
            f"{vis_path}\n{clr_path}"
        )
    vis = pd.read_csv(vis_path)
    clr = pd.read_csv(clr_path)
    for frame, endpoint in ((vis, "viscosity"), (clr, "clearance")):
        required = {"split_seed", "true_value", "predicted_value"}
        if not required.issubset(frame.columns):
            raise ControlledStop(f"Unexpected {endpoint} split-sensitivity schema: {frame.columns.tolist()}")
        if sorted(frame["split_seed"].unique().tolist()) != list(range(6)):
            raise ControlledStop(f"{endpoint}: expected composite split seeds 0-5.")
    vis_err = np.abs(np.log10(vis["predicted_value"].to_numpy(float)) - np.log10(vis["true_value"].to_numpy(float)))
    clr_err = np.abs((clr["predicted_value"].to_numpy(float) - clr["true_value"].to_numpy(float)) / clr["true_value"].to_numpy(float)) * 100
    return (
        {
            "endpoint": "viscosity",
            "n_partitions": 6,
            "n_prediction_instances": len(vis),
            "within_2fold_pct": float(np.mean(vis_err <= math.log10(2)) * 100),
            "within_3fold_pct": float(np.mean(vis_err <= math.log10(3)) * 100),
        },
        {
            "endpoint": "clearance",
            "n_partitions": 6,
            "n_prediction_instances": len(clr),
            "within_15pct": float(np.mean(clr_err <= 15) * 100),
            "within_30pct": float(np.mean(clr_err <= 30) * 100),
        },
    )


def make_table_s13_rows(fixed_summary: pd.DataFrame, vis_split: dict[str, Any], clr_split: dict[str, Any]) -> pd.DataFrame:
    vis = fixed_summary.loc[fixed_summary["endpoint"] == "viscosity"].iloc[0]
    clr = fixed_summary.loc[fixed_summary["endpoint"] == "clearance"].iloc[0]
    rows = [
        {
            "Endpoint": "Viscosity — fixed holdout",
            "Repeat basis": "Five independent training seeds on the unchanged 52/23 manuscript split; fixed KAN head",
            "Summary": (
                f"Held-out R² = {vis.mean_r2:.3f} ± {vis.sd_r2:.3f}; "
                f"RMSE = {vis.mean_rmse:.2f} ± {vis.sd_rmse:.2f} cP; "
                f"MAE = {vis.mean_mae:.2f} ± {vis.sd_mae:.2f} cP; "
                f"Spearman ρ = {vis.mean_spearman:.3f} ± {vis.sd_spearman:.3f}."
            ),
        },
        {
            "Endpoint": "Viscosity — split sensitivity",
            "Repeat basis": "Six composite-stratified train/test partitions (seeds 0-5); fixed KAN head and current workflow",
            "Summary": (
                f"Across {vis_split['n_prediction_instances']} held-out prediction instances, "
                f"{vis_split['within_2fold_pct']:.1f}% were within twofold and "
                f"{vis_split['within_3fold_pct']:.1f}% were within threefold error. "
                "Test membership changed by partition; no aggregate R² is reported."
            ),
        },
        {
            "Endpoint": "Mouse exposure — fixed holdout",
            "Repeat basis": "Five independent training seeds on the unchanged 42/11 manuscript split; fixed spline head",
            "Summary": (
                f"Held-out R² = {clr.mean_r2:.3f} ± {clr.sd_r2:.3f}; "
                f"nRMSE = {clr.mean_nrmse:.3f} ± {clr.sd_nrmse:.3f}; "
                f"nMAE = {clr.mean_nmae:.3f} ± {clr.sd_nmae:.3f}; "
                f"Spearman ρ = {clr.mean_spearman:.3f} ± {clr.sd_spearman:.3f}."
            ),
        },
        {
            "Endpoint": "Mouse exposure — split sensitivity",
            "Repeat basis": "Six composite-stratified train/test partitions (seeds 0-5); fixed spline head and current workflow",
            "Summary": (
                f"Across {clr_split['n_prediction_instances']} held-out prediction instances, "
                f"{clr_split['within_15pct']:.1f}% were within 15% and "
                f"{clr_split['within_30pct']:.1f}% were within 30% error. "
                "Each partition used a different 11-antibody test set; no aggregate R² is reported."
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_summary_md(path: Path, fixed_summary: pd.DataFrame, table_rows: pd.DataFrame) -> None:
    lines = [
        "# Combined regression robustness summary",
        "",
        "Two complementary questions are reported separately:",
        "",
        "1. **Fixed-holdout training-seed stability:** the manuscript train/test split is unchanged and only the training seed varies; mean ± SD is therefore reported across runs.",
        "2. **Composite-split sensitivity:** the held-out antibodies change across six partitions; error coverage and parity are displayed descriptively, and no aggregate R² is reported.",
        "",
        "## Fixed-holdout metrics",
        "",
    ]
    try:
        lines.append(fixed_summary.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        lines.extend(["```csv", fixed_summary.to_csv(index=False).rstrip(), "```"])
    lines.extend(["", "## Table S13 replacement regression rows", ""])
    try:
        lines.append(table_rows.to_markdown(index=False))
    except ImportError:
        lines.extend(["```csv", table_rows.to_csv(index=False).rstrip(), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(package_root: Path, output_dir: Path) -> dict[str, Any]:
    viscosity_pred = package_root / "MainPack" / "viscosity" / "results" / "figure2de" / "figure2e_ablation_predictions.csv"
    viscosity_runs = package_root / "MainPack" / "viscosity" / "results" / "figure2de" / "figure2e_ablation_runs.csv"
    clearance_producer = package_root / "MainPack" / "shared" / "pooling_ablation" / "clearance_pooling_ablation.py"
    clearance_train = package_root / "Data" / "fixed_splits" / "mouse_exposure" / "DataS2_clearance_seed0_train.csv"
    clearance_test = package_root / "Data" / "fixed_splits" / "mouse_exposure" / "DataS2_clearance_seed0_test.csv"
    split_vis = package_root / "GenBench" / "robustness" / "split_sensitivity" / "results" / "locked" / "viscosity_alternate_partition_predictions.csv"
    split_clr = package_root / "GenBench" / "robustness" / "split_sensitivity" / "results" / "locked" / "mouse_exposure_alternate_partition_predictions.csv"
    required = [viscosity_pred, viscosity_runs, clearance_producer, clearance_train, clearance_test, split_vis, split_clr]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ControlledStop("Required files are missing:\n" + "\n".join(missing))
    vis_pred, vis_metrics = load_viscosity_fixed(package_root)
    train_numeric, test_meta, train_path, test_path = clearance_input_tables(package_root, output_dir)
    vis_split, clr_split = split_sensitivity_rows(package_root)
    report = {
        "status": "passed",
        "script_version": SCRIPT_VERSION,
        "package_root": str(package_root),
        "viscosity_fixed_holdout": {"n_seeds": 5, "n_test": 23, "n_predictions": len(vis_pred)},
        "clearance_fixed_holdout": {"n_seeds_planned": 5, "n_train": len(train_numeric), "n_test": len(test_meta)},
        "split_sensitivity": {"viscosity": vis_split, "clearance": clr_split},
        "input_hashes": {str(p.relative_to(package_root) if str(p).startswith(str(package_root)) else p): sha256_file(p) for p in required},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "PREFLIGHT_REPORT.json", report)
    return report


def run_all(package_root: Path, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_pred, vis_metrics = load_viscosity_fixed(package_root)
    vis_pred.to_csv(output_dir / "viscosity_fixed_holdout_predictions.csv", index=False)
    vis_metrics.to_csv(output_dir / "viscosity_fixed_holdout_metrics.csv", index=False)
    per_antibody_summary(vis_pred).to_csv(output_dir / "viscosity_fixed_holdout_per_antibody.csv", index=False)

    _, test_meta, train_path, test_path = clearance_input_tables(package_root, output_dir)
    producer = package_root / "MainPack" / "shared" / "pooling_ablation" / "clearance_pooling_ablation.py"
    clearance_pred_frames: list[pd.DataFrame] = []
    clearance_metric_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        result_path = run_clearance_seed(producer, seed, train_path, test_path, output_dir, force)
        pred, metrics = parse_clearance_predictions(result_path, seed, test_meta)
        clearance_pred_frames.append(pred)
        clearance_metric_rows.append(metrics)
        print(
            f"[clearance fixed holdout seed {seed}] R2={metrics['r2']:.4f} "
            f"nRMSE={metrics['nrmse']:.4f} Spearman={metrics['spearman']:.4f}"
        )
    clr_pred = pd.concat(clearance_pred_frames, ignore_index=True)
    clr_metrics = pd.DataFrame(clearance_metric_rows)
    require_same_fixed_holdout(clr_pred, "mAb_id", "true_value")
    clr_pred.to_csv(output_dir / "clearance_fixed_holdout_predictions.csv", index=False)
    clr_metrics.to_csv(output_dir / "clearance_fixed_holdout_metrics.csv", index=False)
    per_antibody_summary(clr_pred).to_csv(output_dir / "clearance_fixed_holdout_per_antibody.csv", index=False)

    fixed_summary = pd.DataFrame(
        [
            summarize_metric_frame(vis_metrics, "viscosity"),
            summarize_metric_frame(clr_metrics, "clearance"),
        ]
    )
    fixed_summary.to_csv(output_dir / "fixed_holdout_stability_summary.csv", index=False)
    vis_split, clr_split = split_sensitivity_rows(package_root)
    table_rows = make_table_s13_rows(fixed_summary, vis_split, clr_split)
    table_rows.to_csv(output_dir / "TABLE_S13_COMBINED_REGRESSION_ROWS.csv", index=False)
    write_summary_md(output_dir / "COMBINED_ROBUSTNESS_SUMMARY.md", fixed_summary, table_rows)
    write_checksums(output_dir)


def summarize_existing(package_root: Path, output_dir: Path) -> None:
    vis_pred = pd.read_csv(output_dir / "viscosity_fixed_holdout_predictions.csv")
    vis_metrics = pd.read_csv(output_dir / "viscosity_fixed_holdout_metrics.csv")
    clr_pred = pd.read_csv(output_dir / "clearance_fixed_holdout_predictions.csv")
    clr_metrics = pd.read_csv(output_dir / "clearance_fixed_holdout_metrics.csv")
    require_same_fixed_holdout(vis_pred, "mAb_id", "true_value")
    require_same_fixed_holdout(clr_pred, "mAb_id", "true_value")
    fixed_summary = pd.DataFrame(
        [summarize_metric_frame(vis_metrics, "viscosity"), summarize_metric_frame(clr_metrics, "clearance")]
    )
    fixed_summary.to_csv(output_dir / "fixed_holdout_stability_summary.csv", index=False)
    per_antibody_summary(vis_pred).to_csv(output_dir / "viscosity_fixed_holdout_per_antibody.csv", index=False)
    per_antibody_summary(clr_pred).to_csv(output_dir / "clearance_fixed_holdout_per_antibody.csv", index=False)
    vis_split, clr_split = split_sensitivity_rows(package_root)
    table_rows = make_table_s13_rows(fixed_summary, vis_split, clr_split)
    table_rows.to_csv(output_dir / "TABLE_S13_COMBINED_REGRESSION_ROWS.csv", index=False)
    write_summary_md(output_dir / "COMBINED_ROBUSTNESS_SUMMARY.md", fixed_summary, table_rows)
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
        else package_root / "ReproducedOutputs" / "robustness" / "fixed_holdout_stability"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "RUN_LOG.txt").open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] mode={args.mode} script={SCRIPT_VERSION}\n")
    try:
        report = preflight(package_root, output_dir)
        print(json.dumps(report, indent=2))
        if args.mode == "run":
            run_all(package_root, output_dir, args.force)
        elif args.mode == "summarize":
            summarize_existing(package_root, output_dir)
        return 0
    except ControlledStop as exc:
        message = str(exc)
        print("CONTROLLED STOP:\n" + message, file=sys.stderr)
        (output_dir / "CONTROLLED_STOP.txt").write_text(message + "\n", encoding="utf-8")
        return 2
    except Exception:
        import traceback

        text = traceback.format_exc()
        print(text, file=sys.stderr)
        (output_dir / "UNEXPECTED_ERROR.txt").write_text(text, encoding="utf-8")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
