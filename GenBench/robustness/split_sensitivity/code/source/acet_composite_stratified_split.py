#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acet_composite_stratified_split.py

Self-contained train/test split generator for small antibody regression datasets.

This script merges the relevant functionality from:
  - antibody_splits.py (composite stratified split + coverage analysis, with rare-strata handling)
  - auto_split_best.py (search over (n_clusters, n_bins) to select the "best" split for a seed)

It implements the composite-stratified split described in the manuscript Methods for the
viscosity and clearance regression tasks:

1) Standardize numeric features
2) PCA (n_components = min(5, n_features))
3) KMeans clustering into n_clusters
4) Bin the continuous target into n_bins quantiles
5) Create composite strata = (cluster, target-bin)
6) Merge rare strata (<2 samples) to avoid StratifiedShuffleSplit failures
7) StratifiedShuffleSplit into train/test

To select split hyperparameters, it evaluates each (n_clusters, n_bins) pair with:
  - OOD rate: fraction of test points whose nearest-neighbor distance to train exceeds the
              95th percentile of train–train distances
  - mean NN distance: mean nearest-neighbor distance from test to train

It chooses the lexicographically minimal (OOD_rate, mean_NN_distance).

Outputs:
  - <out_prefix>_train.csv
  - <out_prefix>_test.csv
  - <out_prefix>_split_meta.json (chosen params + coverage stats)

Notes:
  - Console output is kept ASCII-only for Windows terminal compatibility.

cd MainPack/code/utils/splitting

python acet_composite_stratified_split.py ^
  --data-csv DataS1_viscosity_antibodies.csv ^
  --target-col Viscosity ^
  --seed 0 ^
  --test-size 0.30 ^
  --clusters 3 4 5 6 7 8 ^
  --bins 1 2 3 4 5 ^
  --out-dir . ^
  --out-prefix viscosity_seed0

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _ensure_numeric_matrix(df: pd.DataFrame, *, target_col: str, drop_cols: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      X (float64) numeric feature matrix
      y (float64) target vector
      df_out (original df)
    """
    drop_cols = list(drop_cols) if drop_cols else []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    # Keep a copy so we write out the original columns (including non-numeric, if any).
    df_out = df.copy()

    y = df[target_col].to_numpy()
    if pd.isna(y).any():
        raise ValueError(f"Target column '{target_col}' contains NaNs. Please impute/drop before splitting.")
    try:
        y = y.astype(float)
    except Exception as e:
        raise ValueError(f"Target column '{target_col}' must be numeric for this regression split generator.") from e

    feature_df = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    if feature_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping target/non-numerics.")

    if feature_df.isna().any().any():
        # Keep error message explicit; this avoids silent row drops that would break reproducibility.
        nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(f"Numeric features contain NaNs in columns: {nan_cols}. Please impute/drop before splitting.")

    X = feature_df.to_numpy(dtype=float)
    return X, y, df_out


def composite_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    n_clusters: int,
    n_bins: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Composite stratified split with rare-strata merging.

    Returns:
      train_idx, test_idx, clusters, target_bins, strata_labels
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("--test-size must be in (0, 1).")

    n_samples = X.shape[0]
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError(f"n_clusters must be in [2, n_samples-1], got {n_clusters} for n_samples={n_samples}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    # Feature clustering: Standardize -> PCA -> KMeans
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, Xs.shape[1]), random_state=seed)
    Z = pca.fit_transform(Xs)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    clusters = km.fit_predict(Z)

    # Target binning (quantiles). duplicates='drop' makes it robust to repeated values.
    target_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    # Convert to numpy array for downstream usage
    target_bins_arr = np.asarray(target_bins, dtype=int)

    # Composite strata labels: "<cluster>_<bin>"
    strata = pd.Series(clusters.astype(str)) + "_" + pd.Series(target_bins_arr.astype(str))

    # Merge rare strata (<2 samples) into broader groups
    counts = strata.value_counts()
    rare = counts[counts < 2].index
    if len(rare) > 0:
        # Replace rare composite labels with cluster-only label
        strata = strata.where(~strata.isin(rare), strata.str.split("_").str[0])

        # After merge, if any labels are still rare, merge those into the most common label
        counts2 = strata.value_counts()
        still_rare = counts2[counts2 < 2].index
        if len(still_rare) > 0:
            main = counts2.idxmax()
            strata = strata.where(~strata.isin(still_rare), main)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, strata))

    return train_idx, test_idx, clusters, target_bins_arr, strata


def coverage_analysis(X_train: np.ndarray, X_test: np.ndarray, *, percentile: float = 95.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Train->test coverage via nearest-neighbor distances.

    Returns:
      dists_test_to_train (n_test,)
      threshold (float): percentile of train-train 2nd-nearest distances
      ood_flags (bool array): True if test NN distance > threshold
    """
    if X_train.shape[0] < 2:
        raise ValueError("Need at least 2 training samples for coverage analysis.")

    # Train-train (2nd NN distance) for threshold
    nn_train = NearestNeighbors(n_neighbors=2).fit(X_train)
    dists_tt, _ = nn_train.kneighbors(X_train)
    threshold = float(np.percentile(dists_tt[:, 1], percentile))

    # Test->train (1st NN distance)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    dists_te, _ = nn.kneighbors(X_test)
    dists_te = dists_te.ravel()
    ood_flags = dists_te > threshold
    return dists_te, threshold, ood_flags


def find_best_params(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cluster_list: Iterable[int],
    bin_list: Iterable[int],
    test_size: float,
) -> Dict[str, Any]:
    """
    Search over all combinations of n_clusters and n_bins.
    Select the lexicographically minimal (ood_rate, mean_nn_dist).
    """
    best: Optional[Dict[str, Any]] = None

    for n_clusters in cluster_list:
        for n_bins in bin_list:
            try:
                tr_idx, te_idx, _, _, _ = composite_stratified_split(
                    X, y,
                    test_size=test_size,
                    n_clusters=n_clusters,
                    n_bins=n_bins,
                    seed=seed
                )
            except Exception as e:
                # Skip invalid combinations (e.g. too many clusters)
                continue

            d_te, thresh, ood = coverage_analysis(X[tr_idx], X[te_idx])
            ood_rate = float(np.mean(ood))
            mean_dist = float(np.mean(d_te))

            candidate = {
                "n_clusters": int(n_clusters),
                "n_bins": int(n_bins),
                "ood_rate": ood_rate,
                "mean_nn_dist": mean_dist,
                "nn_threshold": float(thresh),
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            }

            if best is None or (candidate["ood_rate"], candidate["mean_nn_dist"]) < (best["ood_rate"], best["mean_nn_dist"]):
                best = candidate

    if best is None:
        raise RuntimeError("No valid (n_clusters, n_bins) combinations found. Adjust --clusters/--bins.")

    return best


def _write_outputs(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    out_dir: Path,
    out_prefix: str,
    meta: Dict[str, Any],
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{out_prefix}_train.csv"
    test_path  = out_dir / f"{out_prefix}_test.csv"
    meta_path  = out_dir / f"{out_prefix}_split_meta.json"

    df.iloc[train_idx].to_csv(train_path, index=False)
    df.iloc[test_idx].to_csv(test_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return train_path, test_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composite-stratified train/test split generator (regression) with best-(k,b) search."
    )
    parser.add_argument("--data-csv", required=True, help="Path to input CSV (full dataset).")
    parser.add_argument("--target-col", required=True, help="Continuous target column name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for PCA/KMeans/StratifiedShuffleSplit.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Test fraction (e.g. 0.30 for 70/30).")

    parser.add_argument("--clusters", nargs="+", type=int, default=[3,4,5,6,7,8], help="n_clusters candidates to try.")
    parser.add_argument("--bins", nargs="+", type=int, default=[1,2,3,4,5], help="n_bins candidates to try.")

    parser.add_argument("--mode", choices=["search", "fixed"], default="search",
                        help="search: choose best (n_clusters,n_bins); fixed: use --n-clusters/--n-bins.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Fixed n_clusters (only used when --mode fixed).")
    parser.add_argument("--n-bins", type=int, default=None, help="Fixed n_bins (only used when --mode fixed).")

    parser.add_argument("--drop-cols", nargs="*", default=[], help="Optional extra columns to drop from features (e.g., IDs).")

    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--out-prefix", default="split", help="Output prefix for train/test/meta files.")

    args = parser.parse_args()

    data_path = Path(args.data_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df = pd.read_csv(data_path)

    X, y, df_out = _ensure_numeric_matrix(df, target_col=args.target_col, drop_cols=args.drop_cols)

    if args.mode == "search":
        best = find_best_params(
            X, y,
            seed=int(args.seed),
            cluster_list=args.clusters,
            bin_list=args.bins,
            test_size=float(args.test_size),
        )
        n_clusters = best["n_clusters"]
        n_bins = best["n_bins"]
        print(f"[split] seed={args.seed} best_n_clusters={n_clusters} best_n_bins={n_bins} "
              f"(ood_rate={best['ood_rate']:.4f}, mean_nn_dist={best['mean_nn_dist']:.4f})")
    else:
        if args.n_clusters is None or args.n_bins is None:
            raise ValueError("When --mode fixed, you must supply --n-clusters and --n-bins.")
        n_clusters = int(args.n_clusters)
        n_bins = int(args.n_bins)
        best = {
            "n_clusters": n_clusters,
            "n_bins": n_bins,
        }
        print(f"[split] seed={args.seed} fixed_n_clusters={n_clusters} fixed_n_bins={n_bins}")

    tr_idx, te_idx, clusters, target_bins, strata = composite_stratified_split(
        X, y,
        test_size=float(args.test_size),
        n_clusters=n_clusters,
        n_bins=n_bins,
        seed=int(args.seed),
    )
    d_te, thresh, ood = coverage_analysis(X[tr_idx], X[te_idx])

    # Build meta info
    meta: Dict[str, Any] = dict(best)
    meta.update({
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "target_col": str(args.target_col),
        "n_samples": int(X.shape[0]),
        "n_features_numeric": int(X.shape[1]),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "nn_threshold_95pct_train_train": float(thresh),
        "test_nn_dist_mean": float(np.mean(d_te)),
        "test_nn_dist_std": float(np.std(d_te)),
        "ood_rate": float(np.mean(ood)),
        "ood_count": int(np.sum(ood)),
        "strata_counts": strata.value_counts().to_dict(),
        "cluster_counts": {str(i): int(c) for i, c in enumerate(np.bincount(clusters))},
        "target_bin_counts": {str(i): int(c) for i, c in enumerate(np.bincount(target_bins))},
    })

    out_dir = Path(args.out_dir)
    train_path, test_path, meta_path = _write_outputs(df_out, tr_idx, te_idx, out_dir=out_dir, out_prefix=args.out_prefix, meta=meta)

    print(f"[write] train_csv={train_path}")
    print(f"[write] test_csv={test_path}")
    print(f"[write] meta_json={meta_path}")
    print(f"[coverage] ood_rate={meta['ood_rate']:.4f} (ood_count={meta['ood_count']}/{meta['n_test']}), "
          f"mean_nn_dist={meta['test_nn_dist_mean']:.4f}")


if __name__ == "__main__":
    main()
