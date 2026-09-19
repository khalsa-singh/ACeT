#!/usr/bin/env python
"""
Reproduce Bailly et al. benchmark figures/tables from provided OOF prediction tables.

This script is designed for reproducibility without re-training models.
It regenerates:
- Parity plots (True vs OOF predictions)
- 30-min triage confusion matrices (ACeT vs Bailly-refit baseline)
- CSV summaries of continuous + binary metrics
- CV-replicate variability summaries (mean +/- SD) from *_repeat_metrics.csv

"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.stats import spearmanr, pearsonr


def _auto_find_bench_root() -> Path | None:
    """Try to locate GenBench relative to this file."""
    here = Path(__file__).resolve()
    for up in [here.parents[i] for i in range(min(8, len(here.parents)))]:
        cand = up / "GenBench"
        if cand.exists() and cand.is_dir():
            return cand
    return None


def _metrics_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr = spearmanr(y_true, y_pred).correlation
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "sklearn_R2": float(r2),
        "pearson_r2": float(pr**2),
        "spearman_rho": float(sr),
        "rmse": rmse,
        "mae": mae,
    }


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, cutoff: float = 30.0) -> dict:
    true_poor = (y_true > cutoff).astype(int)
    pred_poor = (y_pred > cutoff).astype(int)
    bal_acc = float(balanced_accuracy_score(true_poor, pred_poor))
    mcc = float(matthews_corrcoef(true_poor, pred_poor))
    cm = confusion_matrix(true_poor, pred_poor, labels=[0, 1])
    # cm rows: true {good(0), poor(1)}, cols: pred {good(0), poor(1)}
    good_acc = float(cm[0, 0] / max(1, cm[0].sum()))
    poor_acc = float(cm[1, 1] / max(1, cm[1].sum()))
    return {
        "balanced_acc": bal_acc,
        "mcc": mcc,
        "good_acc": good_acc,
        "poor_acc": poor_acc,
        "cm_true_good_poor__pred_good_poor": cm.tolist(),
        "cutoff_min": float(cutoff),
    }


def _save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(y_true, y_pred, s=18, alpha=0.85)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.xlabel("True HIC RT (min)")
    plt.ylabel("OOF predicted HIC RT (min)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _save_confusion(out_png: Path, y_true: np.ndarray, y_pred: np.ndarray, cutoff: float, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    true_poor = (y_true > cutoff).astype(int)
    pred_poor = (y_pred > cutoff).astype(int)
    cm = confusion_matrix(true_poor, pred_poor, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["good (≤30)", "poor (>30)"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _summarize_repeat_metrics(repeat_metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(repeat_metrics_csv)
    numeric_cols = [c for c in df.columns if c not in ("repeat", "seed")]
    out = []
    for c in numeric_cols:
        out.append({"metric": c, "mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1))})
    return pd.DataFrame(out).sort_values("metric")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bench_root", type=str, default=None,
                   help="Path to GenBench folder (contains hic_benchmark/results/benchmark/). "
                        "If omitted, the script will try to auto-detect it.")
    p.add_argument("--out_dir", type=str, default="results/bailly",
                   help="Output directory (created if missing).")
    p.add_argument("--cutoff", type=float, default=30.0, help="Binary triage cutoff in minutes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.bench_root is None:
        bench_root = _auto_find_bench_root()
        if bench_root is None:
            raise SystemExit("Could not auto-detect bench_root. Please pass --bench_root.")
    else:
        bench_root = Path(args.bench_root).resolve()

    bailly_pkg = bench_root / "hic_benchmark" / "results" / "benchmark"
    pred_root = bailly_pkg / "oof_predictions"
    met_root = bailly_pkg / "metrics"

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three feature sets
    feature_sets = {
        "assays_only": pred_root / "assays_only" / "hic_rt_min_OOF_assays_only_oof_predictions.csv",
        "assays_plus_patch": pred_root / "assays_plus_patch" / "hic_rt_min_OOF_all_oof_predictions.csv",
        "patch_only": pred_root / "patch_only" / "hic_rt_min_OOF_patch_only_oof_predictions.csv",
    }

    repeat_metric_files = {
        "assays_only": met_root / "assays_only" / "hic_rt_min_OOF_assays_only_repeat_metrics.csv",
        "assays_plus_patch": met_root / "assays_plus_patch" / "hic_rt_min_OOF_all_repeat_metrics.csv",
        "patch_only": met_root / "patch_only" / "hic_rt_min_OOF_patch_only_repeat_metrics.csv",
    }

    rows = []
    for name, csv_path in feature_sets.items():
        df = pd.read_csv(csv_path)
        y_true = df["True"].to_numpy(dtype=float)
        # ACeT OOF prediction column
        y_pred_acet = df["OOF_ACeT"].to_numpy(dtype=float)
        y_pred_refit = df["OOF_BaillyRefit"].to_numpy(dtype=float)

        # Continuous metrics
        m_acet = _metrics_continuous(y_true, y_pred_acet)
        m_refit = _metrics_continuous(y_true, y_pred_refit)

        # Binary metrics
        b_acet = _metrics_binary(y_true, y_pred_acet, cutoff=args.cutoff)
        b_refit = _metrics_binary(y_true, y_pred_refit, cutoff=args.cutoff)

        rows.append({"feature_set": name, "model": "ACeT", **m_acet,
                     "balanced_acc@cutoff": b_acet["balanced_acc"], "mcc@cutoff": b_acet["mcc"]})
        rows.append({"feature_set": name, "model": "Bailly_refit", **m_refit,
                     "balanced_acc@cutoff": b_refit["balanced_acc"], "mcc@cutoff": b_refit["mcc"]})

        # Plots
        _save_parity_plot(y_true, y_pred_acet, out_dir / f"bailly_{name}_parity_acet.png",
                          title=f"Bailly HIC RT OOF parity (ACeT) — {name}")
        _save_parity_plot(y_true, y_pred_refit, out_dir / f"bailly_{name}_parity_bailly_refit.png",
                          title=f"Bailly HIC RT OOF parity (Bailly-refit) — {name}")

        _save_confusion(out_dir / f"bailly_{name}_cm_acet.png", y_true, y_pred_acet,
                        cutoff=args.cutoff, title=f"30-min triage confusion (ACeT) — {name}")
        _save_confusion(out_dir / f"bailly_{name}_cm_bailly_refit.png", y_true, y_pred_refit,
                        cutoff=args.cutoff, title=f"30-min triage confusion (Bailly-refit) — {name}")

        # Repeat-metric variability summary
        rep_csv = repeat_metric_files.get(name)
        if rep_csv and rep_csv.exists():
            rep_summary = _summarize_repeat_metrics(rep_csv)
            rep_summary.to_csv(out_dir / f"bailly_{name}_repeat_metric_summary_mean_sd.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "bailly_metrics_summary.csv", index=False)

    # Also dump JSON for convenience
    (out_dir / "bailly_metrics_summary.json").write_text(json.dumps(rows, indent=2))

    print(f"[ok] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
