#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current-pipeline Figure 2d/2e analysis for the ACeT analysis.

This script is deliberately limited to the viscosity endpoint and the already
selected KAN head. It keeps the authoritative 52/23 seed-0 train/test split,
the current four-assay panel, the current train+train+synthetic construction,
and the current five-fold ensemble logic used for the validated Figure 2a
anchor. 

Modes
-----
preflight   Validate files, rows, features, hashes, and package versions.
figure2d    Refit five current-pipeline full-model seeds (0-4 by default),
            reproduce the seed-0 Figure 2a anchor, and calculate 10-repeat
            held-out permutation importance for each seed.
figure2e    Reuse the full-model seed runs, run the three prespecified feature
            ablations across the same five seeds, and run a nested seed-0
            learning curve using 30%-100% of the 52 real training rows.
all         Run preflight, Figure 2d, and Figure 2e.
summarize   Rebuild final CSV/MAT/Markdown summaries from completed run folders
            without training.

Important statistical interpretation
------------------------------------
* Figure 2d error bars are across five prespecified training seeds. Each seed's
  point is the mean decrease in held-out R^2 across 10 feature permutations.
* Figure 2e ablation error bars are across the same five training seeds, with
  paired tests against the full condition and Holm adjustment.
* The learning curve is descriptive. By default it uses seed 0 only, fixed KAN,
  nested real-row subsets, the unchanged 23-row test set, and a proportional
  1:2 synthetic:real ratio at each size.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-08.2"
TARGET = "Viscosity"
ID_COLUMN = "mAb_id"
SPLIT_COLUMN = "split_seed0"
TOTAL_MODELING_N = 75
EXPECTED_TRAIN_N = 52
EXPECTED_TEST_N = 23
EXPECTED_FEATURES = [
    "DLS Interaction Parameter kD (mL/g)",
    "SE-UHPLC Main Peak Plates (EP)",
    "AC-SINS λmax (nm)",
    "SE-UHPLC Main Peak FWHM (min)",
]
FEATURE_SETS = {
    "full": EXPECTED_FEATURES,
    "top2": [EXPECTED_FEATURES[0], EXPECTED_FEATURES[1]],
    "ht_assays": [EXPECTED_FEATURES[0], EXPECTED_FEATURES[2]],
    "no_kd": [EXPECTED_FEATURES[1], EXPECTED_FEATURES[2], EXPECTED_FEATURES[3]],
}
DISPLAY_NAMES = {
    EXPECTED_FEATURES[0]: "DLS kD",
    EXPECTED_FEATURES[1]: "SE-UHPLC Plates",
    EXPECTED_FEATURES[2]: "AC-SINS Δλmax",
    EXPECTED_FEATURES[3]: "SE-UHPLC FWHM",
}
CONDITION_DISPLAY = {
    "full": "Full",
    "top2": "Top2 (kD, Plates)",
    "ht_assays": "HT assays",
    "no_kd": "No kD",
}


class AnalysisError(RuntimeError):
    """Controlled analysis stop with a user-readable reason."""


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    mainpack: Path
    code_dir: Path
    train_csv: Path
    test_csv: Path
    shared_train_csv: Path
    shared_test_csv: Path
    anchor_predictions_csv: Path
    anchor_mat: Path
    output_dir: Path


@dataclass
class RunResult:
    run_dir: Path
    condition: str
    seed: int
    features: list[str]
    n_real: int
    n_synthetic: int
    metrics: dict[str, Any]
    test_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    models: list[tuple[Any, Any]] | None = None
    predictor_scaler: Any | None = None
    X_test_scaled: np.ndarray | None = None
    y_test: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a small table without making `tabulate` a hard dependency."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        csv_text = frame.to_csv(index=False).rstrip()
        return "```csv\n" + csv_text + "\n```"


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Seed values must be unique")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one fraction is required")
    if any(value <= 0 or value > 1 for value in values):
        raise argparse.ArgumentTypeError("Fractions must lie in (0, 1]")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("Fractions must be in ascending order")
    return values


def discover_repo_root(start: Path, explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "Data/curated").is_dir(): return root
        raise AnalysisError("--repo-root must contain Data/curated")
    for parent in (start, *start.parents):
        if (parent / "Data/curated").is_dir() and (parent / "MainPack").is_dir(): return parent
    raise AnalysisError("Repository not found; supply --repo-root.")


def discover_mainpack(script_path: Path, repo_root: Path, explicit: str | None) -> Path:
    return Path(explicit).resolve() if explicit else repo_root / "MainPack"


def build_paths(args: argparse.Namespace) -> Paths:
    root = discover_repo_root(Path(__file__).resolve(), args.repo_root)
    mainpack = discover_mainpack(Path(__file__).resolve(), root, args.mainpack)
    shared = mainpack / "viscosity/code/source/shared"
    anchor = mainpack / "viscosity/results/primary"
    return Paths(
        repo_root=root, mainpack=mainpack, code_dir=shared,
        train_csv=Path(args.train_csv).resolve() if args.train_csv else root / "Data/fixed_splits/viscosity/DataS1_viscosity_seed0_train.csv",
        test_csv=Path(args.test_csv).resolve() if args.test_csv else root / "Data/fixed_splits/viscosity/DataS1_viscosity_seed0_test.csv",
        shared_train_csv=shared / "antibodies_train.csv", shared_test_csv=shared / "antibodies_test.csv",
        anchor_predictions_csv=anchor / "viscosity_predictions.csv", anchor_mat=anchor / "viscosity_results.mat",
        output_dir=Path(args.output_dir).resolve() if args.output_dir else root / "reruns/figure2de",
    )


def package_versions(include_ml: bool) -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ]
    if include_ml:
        names.extend(
            [
                "tensorflow",
                "keras",
                "sdv",
                "tfkan",
                "ImbalancedLearningRegression",
                "imbalanced-learn",
            ]
        )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def load_ml_modules(paths: Paths) -> dict[str, Any]:
    """Import heavy dependencies only when a training mode is requested."""
    try:
        import tensorflow as tf
        import ImbalancedLearningRegression as iblr
        from scipy.io import savemat
        from scipy.stats import spearmanr, ttest_rel
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.utils import check_random_state
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise AnalysisError(
            "The ML environment could not be imported. Run the preflight command and inspect "
            f"PACKAGE_VERSIONS.json. Original import error: {exc!r}"
        ) from exc

    shared_dir = paths.code_dir
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    try:
        from viscosity_model import build_transformer_model
    except Exception as exc:  # pragma: no cover
        raise AnalysisError(f"Could not import the current shared viscosity model: {exc!r}") from exc

    return {
        "tf": tf,
        "iblr": iblr,
        "savemat": savemat,
        "spearmanr": spearmanr,
        "ttest_rel": ttest_rel,
        "SingleTableMetadata": SingleTableMetadata,
        "GaussianCopulaSynthesizer": GaussianCopulaSynthesizer,
        "permutation_importance": permutation_importance,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "KFold": KFold,
        "QuantileTransformer": QuantileTransformer,
        "check_random_state": check_random_state,
        "build_transformer_model": build_transformer_model,
    }


def validate_inputs(paths: Paths, check_ml: bool) -> dict[str, Any]:
    required = [
        paths.train_csv,
        paths.test_csv,
        paths.shared_train_csv,
        paths.shared_test_csv,
        paths.anchor_predictions_csv,
        paths.anchor_mat,
        paths.code_dir / "viscosity_model.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AnalysisError("Required files are missing:\n" + "\n".join(missing))

    train = pd.read_csv(paths.train_csv)
    test = pd.read_csv(paths.test_csv)
    expected_columns = [ID_COLUMN, SPLIT_COLUMN, *EXPECTED_FEATURES, TARGET]
    if train.columns.tolist() != expected_columns:
        raise AnalysisError(f"Unexpected training columns: {train.columns.tolist()}")
    if test.columns.tolist() != expected_columns:
        raise AnalysisError(f"Unexpected test columns: {test.columns.tolist()}")
    if len(train) != EXPECTED_TRAIN_N or len(test) != EXPECTED_TEST_N:
        raise AnalysisError(f"Expected 52/23 rows, found {len(train)}/{len(test)}")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        raise AnalysisError("mAb_id is not unique within train or test")
    overlap = sorted(set(train[ID_COLUMN]).intersection(test[ID_COLUMN]))
    if overlap:
        raise AnalysisError(f"Train/test ID overlap detected: {overlap}")
    if train[EXPECTED_FEATURES + [TARGET]].isna().any().any() or test[EXPECTED_FEATURES + [TARGET]].isna().any().any():
        raise AnalysisError("Missing values detected in the authoritative viscosity train/test files")
    if not train[SPLIT_COLUMN].eq("train").all() or not test[SPLIT_COLUMN].eq("test").all():
        raise AnalysisError("split_seed0 labels do not match train/test membership")

    stripped_train = train[EXPECTED_FEATURES + [TARGET]].reset_index(drop=True)
    stripped_test = test[EXPECTED_FEATURES + [TARGET]].reset_index(drop=True)
    shared_train = pd.read_csv(paths.shared_train_csv)
    shared_test = pd.read_csv(paths.shared_test_csv)
    if not stripped_train.equals(shared_train):
        raise AnalysisError("DataS1 train rows do not exactly match shared/viscosity/antibodies_train.csv")
    if not stripped_test.equals(shared_test):
        raise AnalysisError("DataS1 test rows do not exactly match shared/viscosity/antibodies_test.csv")

    anchor = pd.read_csv(paths.anchor_predictions_csv)
    if anchor.columns.tolist() != ["True_Viscosity", "Predicted_Viscosity"] or len(anchor) != EXPECTED_TEST_N:
        raise AnalysisError("Unexpected authoritative viscosity_predictions.csv schema")
    if not np.allclose(anchor["True_Viscosity"].to_numpy(float), test[TARGET].to_numpy(float), rtol=0, atol=1e-12):
        raise AnalysisError("Authoritative prediction y_true does not match the current 23-row test file")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr

    y_true = anchor["True_Viscosity"].to_numpy(float)
    y_pred = anchor["Predicted_Viscosity"].to_numpy(float)
    anchor_metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }

    feature_importance = None
    try:
        from scipy.io import loadmat

        mat = loadmat(paths.anchor_mat)
        feature_importance = [float(value) for value in np.asarray(mat["featureImportance"]).reshape(-1)]
    except Exception as exc:
        raise AnalysisError(f"Could not read the current anchor MAT: {exc!r}") from exc

    report = {
        "script_version": SCRIPT_VERSION,
        "repo_root": str(paths.repo_root),
        "mainpack": str(paths.mainpack),
        "train_csv": str(paths.train_csv),
        "test_csv": str(paths.test_csv),
        "train_rows": len(train),
        "test_rows": len(test),
        "total_rows": len(train) + len(test),
        "features": EXPECTED_FEATURES,
        "train_ids": train[ID_COLUMN].tolist(),
        "test_ids": test[ID_COLUMN].tolist(),
        "anchor_metrics": anchor_metrics,
        "anchor_feature_importance_seed0": feature_importance,
        "hashes": {str(path): sha256_file(path) for path in required},
        "package_versions": package_versions(include_ml=check_ml),
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.output_dir / "PREFLIGHT_REPORT.json", report)
    write_json(paths.output_dir / "PACKAGE_VERSIONS.json", report["package_versions"])
    md = [
        "# Figure 2d/2e preflight",
        "",
        f"- Repository: `{paths.repo_root}`",
        f"- MainPack: `{paths.mainpack}`",
        f"- Rows: {len(train)} train / {len(test)} test",
        f"- IDs unique and disjoint: yes",
        f"- Shared stripped CSVs exactly match DataS1: yes",
        f"- Anchor R²: {anchor_metrics['r2']:.9f}",
        f"- Anchor RMSE: {anchor_metrics['rmse']:.9f} cP",
        f"- Anchor MAE: {anchor_metrics['mae']:.9f} cP",
        f"- Anchor Spearman: {anchor_metrics['spearman']:.9f}",
        "",
        "## Package versions",
        "",
    ]
    md.extend([f"- {name}: `{version}`" for name, version in report["package_versions"].items()])
    (paths.output_dir / "PREFLIGHT_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def set_seeds(seed: int, tf: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray, spearmanr: Any) -> float:
    result = spearmanr(y_true, y_pred)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value) if np.isfinite(value) else float("nan")


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, modules: dict[str, Any]) -> dict[str, float]:
    mse = modules["mean_squared_error"](y_true, y_pred)
    return {
        "r2": float(modules["r2_score"](y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "mae": float(modules["mean_absolute_error"](y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred, modules["spearmanr"]),
    }


def sample_synthetic(train: pd.DataFrame, rows: int, seed: int, modules: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    metadata = modules["SingleTableMetadata"]()
    metadata.detect_from_dataframe(train)
    synthesizer = modules["GaussianCopulaSynthesizer"](metadata, default_distribution="norm")
    synthesizer.fit(train)
    try:
        synthetic = synthesizer.sample(num_rows=rows, random_state=seed)
        method = "sample(num_rows, random_state=seed)"
    except TypeError:
        synthetic = synthesizer.sample(num_rows=rows)
        method = "sample(num_rows); random_state argument unsupported"
    return synthetic[train.columns].copy(), method


def rebalance_fold(df_tr: pd.DataFrame, target_col: str, fold: int, modules: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match the current viscosity script's ENN then sequential SMOTE behavior."""
    iblr = modules["iblr"]
    details: dict[str, Any] = {"fold": fold, "enn_rel_coef": 0.5, "smote_attempts": []}
    df_clean = iblr.enn(data=df_tr, y=target_col, rel_coef=0.5)
    details["rows_before_enn"] = len(df_tr)
    details["rows_after_enn_raw"] = len(df_clean)
    if df_clean.isnull().values.any():
        df_clean = df_clean.dropna()
    df_clean = df_clean.loc[:, df_clean.nunique() > 1]
    details["rows_after_enn_clean"] = len(df_clean)
    details["columns_after_enn"] = df_clean.columns.tolist()

    balanced: pd.DataFrame | None = None
    try:
        balanced = iblr.smote(data=df_clean, y=target_col, rel_coef=0.5)
        details["smote_attempts"].append({"rel_coef": 0.5, "status": "success", "rows": len(balanced)})
    except ValueError as exc:
        details["smote_attempts"].append({"rel_coef": 0.5, "status": "failed", "error": repr(exc)})
        print(f"[fold {fold}] SMOTE rel_coef=0.5 failed: {exc}")
    # The current authoritative script performs this second call even when the
    # first call succeeds; preserving it is necessary for the Figure 2a anchor.
    try:
        balanced = iblr.smote(data=df_clean, y=target_col, rel_coef=0.25)
        details["smote_attempts"].append({"rel_coef": 0.25, "status": "success", "rows": len(balanced)})
    except ValueError as exc:
        details["smote_attempts"].append({"rel_coef": 0.25, "status": "failed", "error": repr(exc)})
        print(f"[fold {fold}] SMOTE rel_coef=0.25 failed: {exc}")
    if balanced is None:
        balanced = df_clean.copy()
        details["fallback"] = "ENN-cleaned rows only"
    details["rows_final"] = len(balanced)
    return balanced, details


def condition_run_dir(output_dir: Path, condition: str, seed: int, n_real: int | None = None) -> Path:
    if condition == "learning":
        if n_real is None:
            raise ValueError("n_real is required for learning runs")
        return output_dir / "runs" / "learning" / f"n_{n_real:02d}" / f"seed_{seed}"
    return output_dir / "runs" / "conditions" / condition / f"seed_{seed}"


def build_config(
    paths: Paths,
    args: argparse.Namespace,
    condition: str,
    features: Sequence[str],
    seed: int,
    row_ids: Sequence[str],
    n_synthetic: int,
) -> dict[str, Any]:
    config = {
        "script_version": SCRIPT_VERSION,
        "condition": condition,
        "features": list(features),
        "seed": seed,
        "real_train_ids": list(row_ids),
        "n_real": len(row_ids),
        "n_synthetic": n_synthetic,
        "native_row_copies": 2,
        "head": "kan",
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "num_transformer_blocks": args.num_transformer_blocks,
        "mlp_units": args.mlp_units,
        "dropout_rate": args.dropout_rate,
        "l2_reg": args.l2_reg,
        "patience": args.patience,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "train_csv_sha256": sha256_file(paths.train_csv),
        "test_csv_sha256": sha256_file(paths.test_csv),
        "viscosity_model_sha256": sha256_file(paths.code_dir / "viscosity_model.py"),
    }
    config["config_hash"] = sha256_text(canonical_json(config))
    return config


def save_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_model(args: argparse.Namespace, num_features: int, modules: dict[str, Any]) -> Any:
    model = modules["build_transformer_model"](
        num_features=num_features,
        task="regression",
        head_type="kan",
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        mlp_units=args.mlp_units,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
    )
    model.compile(
        optimizer=modules["tf"].keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="log_cosh",
    )
    return model


def load_cached_run(
    run_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    modules: dict[str, Any],
    need_models: bool,
) -> RunResult | None:
    complete_path = run_dir / "run_complete.json"
    if args.force or not complete_path.exists():
        return None
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("config_hash") != config["config_hash"]:
            print(f"[cache] Config changed; rerunning {run_dir}")
            return None
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        test_predictions = pd.read_csv(run_dir / "test_predictions.csv")
        fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
        result = RunResult(
            run_dir=run_dir,
            condition=config["condition"],
            seed=int(config["seed"]),
            features=list(config["features"]),
            n_real=int(config["n_real"]),
            n_synthetic=int(config["n_synthetic"]),
            metrics=metrics,
            test_predictions=test_predictions,
            fold_metrics=fold_metrics,
        )
        if need_models:
            predictor_scaler = load_pickle(run_dir / "predictor_scaler.pkl")
            models: list[tuple[Any, Any]] = []
            for fold in range(1, args.n_splits + 1):
                target_scaler = load_pickle(run_dir / f"fold_{fold}_target_scaler.pkl")
                model = build_model(args, len(config["features"]), modules)
                model.load_weights(str(run_dir / f"fold_{fold}.weights.h5"))
                models.append((model, target_scaler))
            model_frame = pd.read_csv(run_dir / "test_model_frame.csv")
            X_test_scaled = predictor_scaler.transform(model_frame[config["features"]].to_numpy(float))
            result.models = models
            result.predictor_scaler = predictor_scaler
            result.X_test_scaled = X_test_scaled
            result.y_test = model_frame[TARGET].to_numpy(float)
        print(f"[cache] Reusing completed run: {run_dir}")
        return result
    except Exception as exc:
        print(f"[cache] Could not reuse {run_dir}: {exc!r}; rerunning")
        return None


def run_experiment(
    paths: Paths,
    args: argparse.Namespace,
    modules: dict[str, Any],
    condition: str,
    features: Sequence[str],
    seed: int,
    real_train: pd.DataFrame,
    test: pd.DataFrame,
    n_synthetic: int,
    need_models: bool,
) -> RunResult:
    run_dir = condition_run_dir(paths.output_dir, condition, seed, len(real_train) if condition == "learning" else None)
    config = build_config(paths, args, condition, features, seed, real_train[ID_COLUMN].tolist(), n_synthetic)
    cached = load_cached_run(run_dir, config, args, modules, need_models)
    if cached is not None:
        return cached

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", config)
    start = time.time()
    tf = modules["tf"]
    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds(seed, tf)

    train_model = real_train[list(features) + [TARGET]].reset_index(drop=True)
    test_model = test[[ID_COLUMN, *features, TARGET]].reset_index(drop=True)
    synthetic, sampling_method = sample_synthetic(train_model, n_synthetic, seed, modules)
    synthetic.to_csv(run_dir / "synthetic_data.csv", index=False)
    test_model.to_csv(run_dir / "test_model_frame.csv", index=False)

    train_mix = pd.concat([train_model, train_model, synthetic], ignore_index=True)
    origin = pd.DataFrame(
        {
            "train_mix_index": np.arange(len(train_mix)),
            "origin": ["real_copy_1"] * len(train_model)
            + ["real_copy_2"] * len(train_model)
            + ["synthetic"] * len(synthetic),
            "source_id": real_train[ID_COLUMN].tolist()
            + real_train[ID_COLUMN].tolist()
            + [f"SYN_{seed}_{i:03d}" for i in range(len(synthetic))],
        }
    )
    origin.to_csv(run_dir / "train_mix_origin.csv", index=False)
    train_mix.to_csv(run_dir / "train_mix_model_frame.csv", index=False)

    sc_x = modules["QuantileTransformer"]()
    X_all = sc_x.fit_transform(train_mix[list(features)].to_numpy(float))
    X_test = sc_x.transform(test_model[list(features)].to_numpy(float))
    y_all = train_mix[TARGET].to_numpy(float)
    y_test = test_model[TARGET].to_numpy(float)
    save_pickle(run_dir / "predictor_scaler.pkl", sc_x)

    kf = modules["KFold"](n_splits=args.n_splits, shuffle=True, random_state=seed)
    fold_models: list[tuple[Any, Any]] = []
    fold_metrics_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    fold_index_rows: list[dict[str, Any]] = []
    rebalance_details: list[dict[str, Any]] = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        print(
            f"[{condition} seed={seed}] fold {fold}/{args.n_splits}; "
            f"real={len(real_train)}, synthetic={n_synthetic}, features={len(features)}"
        )
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        df_tr = pd.DataFrame(X_tr, columns=features)
        df_tr[TARGET] = y_tr
        balanced, rebalance = rebalance_fold(df_tr, TARGET, fold, modules)
        expected_columns = [*features, TARGET]
        if balanced.columns.tolist() != expected_columns:
            raise AnalysisError(
                f"Fold {fold} changed the expected model columns. Expected {expected_columns}; "
                f"found {balanced.columns.tolist()}"
            )
        rebalance_details.append(rebalance)
        X_bal = balanced[list(features)].to_numpy(float)
        y_bal = balanced[TARGET].to_numpy(float)

        model = build_model(args, len(features), modules)
        sc_y = modules["QuantileTransformer"]()
        y_bal_scaled = sc_y.fit_transform(y_bal.reshape(-1, 1)).reshape(-1)
        y_val_scaled = sc_y.transform(y_all[val_idx].reshape(-1, 1)).reshape(-1)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.min_lr,
            ),
        ]
        history = model.fit(
            X_bal,
            y_bal_scaled,
            validation_data=(X_all[val_idx], y_val_scaled),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=args.verbose,
        )
        model.save_weights(str(run_dir / f"fold_{fold}.weights.h5"))
        save_pickle(run_dir / f"fold_{fold}_target_scaler.pkl", sc_y)
        fold_models.append((model, sc_y))

        val_pred_scaled = model.predict(X_all[val_idx], verbose=0).reshape(-1)
        val_pred = sc_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(-1)
        fold_metric = metrics_dict(y_all[val_idx], val_pred, modules)
        fold_metric.update(
            {
                "condition": condition,
                "seed": seed,
                "fold": fold,
                "n_train_mix_fold": len(tr_idx),
                "n_validation_mix_fold": len(val_idx),
                "n_balanced_fold": len(balanced),
                "epochs_ran": len(history.history.get("loss", [])),
            }
        )
        fold_metrics_rows.append(fold_metric)
        for epoch, loss in enumerate(history.history.get("loss", []), 1):
            history_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "loss": loss,
                    "val_loss": history.history.get("val_loss", [np.nan] * len(history.history.get("loss", [])))[epoch - 1],
                }
            )
        fold_index_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "fold": fold,
                "train_mix_indices": json.dumps([int(value) for value in tr_idx]),
                "validation_mix_indices": json.dumps([int(value) for value in val_idx]),
            }
        )

    # Preserve the current Figure 2a ensemble calculation exactly: average the
    # fold outputs on their transformed target scales, then inverse-transform
    # using the final fold's target transformer.
    fold_test_scaled = np.vstack(
        [model.predict(X_test, verbose=0).reshape(-1) for model, _ in fold_models]
    )
    ensemble_scaled = fold_test_scaled.mean(axis=0)
    y_pred = fold_models[-1][1].inverse_transform(ensemble_scaled.reshape(-1, 1)).reshape(-1)
    test_metrics = metrics_dict(y_test, y_pred, modules)
    test_metrics.update(
        {
            "condition": condition,
            "condition_display": CONDITION_DISPLAY.get(condition, condition),
            "seed": seed,
            "n_real": len(real_train),
            "n_synthetic": n_synthetic,
            "n_train_mix": len(train_mix),
            "n_test": len(test),
            "features": list(features),
            "synthetic_sampling_method": sampling_method,
            "cv_r2_mean": float(pd.DataFrame(fold_metrics_rows)["r2"].mean()),
            "cv_r2_sd_population": float(pd.DataFrame(fold_metrics_rows)["r2"].std(ddof=0)),
            "runtime_seconds": time.time() - start,
            "config_hash": config["config_hash"],
        }
    )
    test_predictions = pd.DataFrame(
        {
            ID_COLUMN: test_model[ID_COLUMN],
            "condition": condition,
            "seed": seed,
            "true_viscosity": y_test,
            "predicted_viscosity": y_pred,
        }
    )
    for fold in range(args.n_splits):
        fold_pred = fold_models[fold][1].inverse_transform(
            fold_test_scaled[fold].reshape(-1, 1)
        ).reshape(-1)
        test_predictions[f"fold_{fold + 1}_prediction_own_scale"] = fold_pred
        test_predictions[f"fold_{fold + 1}_prediction_scaled"] = fold_test_scaled[fold]

    pd.DataFrame(fold_metrics_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(fold_index_rows).to_csv(run_dir / "fold_indices.csv", index=False)
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_json(run_dir / "rebalance_details.json", rebalance_details)
    write_json(
        run_dir / "run_complete.json",
        {
            "config_hash": config["config_hash"],
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": time.time() - start,
        },
    )

    result = RunResult(
        run_dir=run_dir,
        condition=condition,
        seed=seed,
        features=list(features),
        n_real=len(real_train),
        n_synthetic=n_synthetic,
        metrics=test_metrics,
        test_predictions=test_predictions,
        fold_metrics=pd.DataFrame(fold_metrics_rows),
        models=fold_models if need_models else None,
        predictor_scaler=sc_x if need_models else None,
        X_test_scaled=X_test if need_models else None,
        y_test=y_test if need_models else None,
    )
    if not need_models:
        del fold_models
        tf.keras.backend.clear_session()
        gc.collect()
    return result


def anchor_comparison(
    paths: Paths,
    args: argparse.Namespace,
    result: RunResult,
) -> dict[str, Any]:
    anchor = pd.read_csv(paths.anchor_predictions_csv)
    current_pred = result.test_predictions["predicted_viscosity"].to_numpy(float)
    anchor_pred = anchor["Predicted_Viscosity"].to_numpy(float)
    comparison = {
        "seed": result.seed,
        "anchor_r2": float(args.anchor_r2),
        "run_r2": float(result.metrics["r2"]),
        "abs_r2_delta": abs(float(result.metrics["r2"]) - float(args.anchor_r2)),
        "anchor_rmse": float(args.anchor_rmse),
        "run_rmse": float(result.metrics["rmse"]),
        "abs_rmse_delta": abs(float(result.metrics["rmse"]) - float(args.anchor_rmse)),
        "max_abs_prediction_delta": float(np.max(np.abs(current_pred - anchor_pred))),
        "mean_abs_prediction_delta": float(np.mean(np.abs(current_pred - anchor_pred))),
        "r2_tolerance": args.anchor_r2_tolerance,
        "rmse_tolerance": args.anchor_rmse_tolerance,
        "prediction_tolerance": args.anchor_prediction_tolerance,
    }
    comparison["passed"] = bool(
        comparison["abs_r2_delta"] <= args.anchor_r2_tolerance
        and comparison["abs_rmse_delta"] <= args.anchor_rmse_tolerance
        and comparison["max_abs_prediction_delta"] <= args.anchor_prediction_tolerance
    )
    write_json(paths.output_dir / "ANCHOR_COMPARISON_SEED0.json", comparison)
    pd.DataFrame(
        {
            ID_COLUMN: result.test_predictions[ID_COLUMN],
            "true_viscosity": result.test_predictions["true_viscosity"],
            "anchor_prediction": anchor_pred,
            "current_rerun_prediction": current_pred,
            "difference": current_pred - anchor_pred,
        }
    ).to_csv(paths.output_dir / "ANCHOR_PREDICTION_COMPARISON_SEED0.csv", index=False)
    return comparison


class CurrentEnsembleRegressor:
    """Minimal fitted-estimator wrapper matching the current ensemble output."""

    def __init__(self, models: list[tuple[Any, Any]]):
        self.models = models

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CurrentEnsembleRegressor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scaled = np.mean(
            [model.predict(X, verbose=0).reshape(-1) for model, _ in self.models], axis=0
        )
        return self.models[-1][1].inverse_transform(scaled.reshape(-1, 1)).reshape(-1)


def manual_permutation_predictions(
    estimator: CurrentEnsembleRegressor,
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[str],
    features: Sequence[str],
    seed: int,
    repeats: int,
    modules: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    baseline_pred = estimator.predict(X)
    baseline_r2 = float(modules["r2_score"](y, baseline_pred))
    rng = modules["check_random_state"](seed)
    random_seed = int(rng.randint(np.iinfo(np.int32).max + 1))
    rows: list[dict[str, Any]] = []
    importances = np.zeros((len(features), repeats), dtype=float)
    for feature_index, feature in enumerate(features):
        feature_rng = np.random.RandomState(random_seed)
        X_permuted = X.copy()
        shuffle_index = np.arange(len(X_permuted))
        for repeat in range(repeats):
            feature_rng.shuffle(shuffle_index)
            X_permuted[:, feature_index] = X_permuted[shuffle_index, feature_index]
            prediction = estimator.predict(X_permuted)
            permuted_r2 = float(modules["r2_score"](y, prediction))
            importance = baseline_r2 - permuted_r2
            importances[feature_index, repeat] = importance
            for row_index, (mab_id, true, pred) in enumerate(zip(ids, y, prediction)):
                rows.append(
                    {
                        ID_COLUMN: mab_id,
                        "training_seed": seed,
                        "feature": feature,
                        "feature_display": DISPLAY_NAMES[feature],
                        "repeat": repeat + 1,
                        "row_index": row_index,
                        "true_viscosity": true,
                        "permuted_prediction": pred,
                        "baseline_r2": baseline_r2,
                        "permuted_r2": permuted_r2,
                        "importance_delta_r2": importance,
                    }
                )
    return pd.DataFrame(rows), importances


def compute_permutation_for_seed(
    paths: Paths,
    args: argparse.Namespace,
    modules: dict[str, Any],
    result: RunResult,
    test_ids: Sequence[str],
) -> None:
    if result.models is None or result.X_test_scaled is None or result.y_test is None:
        raise AnalysisError("Models were not loaded for permutation importance")
    estimator = CurrentEnsembleRegressor(result.models)
    sklearn_result = modules["permutation_importance"](
        estimator,
        result.X_test_scaled,
        result.y_test,
        n_repeats=args.permutation_repeats,
        scoring="r2",
        random_state=result.seed,
        n_jobs=1,
    )
    prediction_rows, manual_importances = manual_permutation_predictions(
        estimator,
        result.X_test_scaled,
        result.y_test,
        test_ids,
        result.features,
        result.seed,
        args.permutation_repeats,
        modules,
    )
    max_diff = float(np.max(np.abs(manual_importances - sklearn_result.importances)))
    if max_diff > 1e-10:
        raise AnalysisError(
            f"Manual permutation reconstruction differs from sklearn by {max_diff:.3g}"
        )
    prediction_rows.to_csv(result.run_dir / "permutation_predictions.csv", index=False)
    repeat_rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(result.features):
        for repeat_index in range(args.permutation_repeats):
            repeat_rows.append(
                {
                    "training_seed": result.seed,
                    "feature": feature,
                    "feature_display": DISPLAY_NAMES[feature],
                    "repeat": repeat_index + 1,
                    "importance_delta_r2": float(sklearn_result.importances[feature_index, repeat_index]),
                }
            )
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(result.run_dir / "permutation_importance_repeats.csv", index=False)
    seed_summary = (
        repeat_df.groupby(["training_seed", "feature", "feature_display"], sort=False)["importance_delta_r2"]
        .agg(importance_mean="mean", importance_sd_within_seed=lambda x: x.std(ddof=1))
        .reset_index()
    )
    seed_summary.to_csv(result.run_dir / "permutation_importance_seed_summary.csv", index=False)
    write_json(
        result.run_dir / "permutation_validation.json",
        {
            "sklearn_manual_max_abs_difference": max_diff,
            "baseline_r2": float(result.metrics["r2"]),
            "n_repeats": args.permutation_repeats,
            "random_state": result.seed,
        },
    )


def check_seed0_importance_anchor(
    paths: Paths, args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    """Gate the remaining seeds on the current saved seed-0 mean vector."""
    seed_summary_path = run_dir / "permutation_importance_seed_summary.csv"
    if not seed_summary_path.exists():
        raise AnalysisError(f"Seed-0 permutation summary is missing: {seed_summary_path}")
    seed_summary = pd.read_csv(seed_summary_path).set_index("feature")
    rerun = seed_summary.loc[EXPECTED_FEATURES, "importance_mean"].to_numpy(float)
    preflight = json.loads((paths.output_dir / "PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    anchor = np.asarray(preflight["anchor_feature_importance_seed0"], dtype=float)
    max_delta = float(np.max(np.abs(rerun - anchor)))
    report = {
        "current_anchor": anchor.tolist(),
        "rerun_seed0": rerun.tolist(),
        "max_abs_delta": max_delta,
        "tolerance": args.anchor_importance_tolerance,
        "passed": bool(max_delta <= args.anchor_importance_tolerance),
    }
    write_json(paths.output_dir / "ANCHOR_IMPORTANCE_COMPARISON_SEED0.json", report)
    return report


def holm_sidak_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = 1.0 - (1.0 - p[index]) ** (m - rank)
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = (m - rank) * p[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def aggregate_figure2d(paths: Paths, args: argparse.Namespace, modules: dict[str, Any]) -> None:
    final_dir = paths.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    repeat_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        run_dir = condition_run_dir(paths.output_dir, "full", seed)
        repeat_path = run_dir / "permutation_importance_repeats.csv"
        pred_path = run_dir / "permutation_predictions.csv"
        if not repeat_path.exists() or not pred_path.exists():
            raise AnalysisError(f"Permutation output is incomplete for seed {seed}: {run_dir}")
        repeat_frames.append(pd.read_csv(repeat_path))
        prediction_frames.append(pd.read_csv(pred_path))
    repeats = pd.concat(repeat_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    repeats.to_csv(final_dir / "figure2d_permutation_repeats.csv", index=False)
    predictions.to_csv(final_dir / "figure2d_permutation_predictions.csv", index=False)

    seed_summary = (
        repeats.groupby(["training_seed", "feature", "feature_display"], sort=False)["importance_delta_r2"]
        .agg(importance_mean="mean", importance_sd_within_seed=lambda x: x.std(ddof=1))
        .reset_index()
    )
    seed_summary.to_csv(final_dir / "figure2d_seed_importance.csv", index=False)
    overall = (
        seed_summary.groupby(["feature", "feature_display"], sort=False)["importance_mean"]
        .agg(mean_importance="mean", sd_across_seeds=lambda x: x.std(ddof=1), n_seeds="count")
        .reset_index()
    )
    overall["feature_order"] = overall["feature"].map({name: index for index, name in enumerate(EXPECTED_FEATURES)})
    overall = overall.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    overall.to_csv(final_dir / "figure2d_importance_summary.csv", index=False)

    pivot = seed_summary.pivot(index="training_seed", columns="feature", values="importance_mean")
    pair_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    pairs: list[tuple[str, str]] = []
    for left_index in range(len(EXPECTED_FEATURES)):
        for right_index in range(left_index + 1, len(EXPECTED_FEATURES)):
            left = EXPECTED_FEATURES[left_index]
            right = EXPECTED_FEATURES[right_index]
            result = modules["ttest_rel"](pivot[left], pivot[right])
            raw_p = float(result.pvalue)
            pairs.append((left, right))
            raw_p_values.append(raw_p)
    adjusted = holm_sidak_adjust(raw_p_values)
    for (left, right), raw_p, adjusted_p in zip(pairs, raw_p_values, adjusted):
        pair_rows.append(
            {
                "feature_1": left,
                "feature_1_display": DISPLAY_NAMES[left],
                "feature_2": right,
                "feature_2_display": DISPLAY_NAMES[right],
                "paired_t_statistic": float(modules["ttest_rel"](pivot[left], pivot[right]).statistic),
                "raw_p": raw_p,
                "holm_sidak_p": float(adjusted_p),
                "significance": significance_label(float(adjusted_p)),
                "n_paired_seeds": len(pivot),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(final_dir / "figure2d_pairwise_tests.csv", index=False)

    # Directly load the preflight JSON to avoid pandas' orientation assumptions.
    preflight = json.loads((paths.output_dir / "PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    current_anchor_importance = np.asarray(preflight["anchor_feature_importance_seed0"], dtype=float)
    seed0 = (
        seed_summary.loc[seed_summary["training_seed"].eq(0)]
        .set_index("feature")
        .loc[EXPECTED_FEATURES, "importance_mean"]
        .to_numpy(float)
    )
    feature_anchor_max_delta = float(np.max(np.abs(seed0 - current_anchor_importance)))
    feature_anchor_passed = feature_anchor_max_delta <= args.anchor_importance_tolerance
    write_json(
        final_dir / "figure2d_seed0_importance_anchor.json",
        {
            "current_anchor": current_anchor_importance.tolist(),
            "rerun_seed0": seed0.tolist(),
            "max_abs_delta": feature_anchor_max_delta,
            "tolerance": args.anchor_importance_tolerance,
            "passed": feature_anchor_passed,
        },
    )
    if not feature_anchor_passed and not args.allow_anchor_mismatch:
        raise AnalysisError(
            "Seed-0 permutation importance did not reproduce the current saved mean vector. "
            f"Maximum difference={feature_anchor_max_delta:.4f}; see figure2d_seed0_importance_anchor.json"
        )

    modules["savemat"](
        final_dir / "Figure2d_current.mat",
        {
            "feature_names": np.array(EXPECTED_FEATURES, dtype=object),
            "feature_display": np.array([DISPLAY_NAMES[name] for name in EXPECTED_FEATURES], dtype=object),
            "training_seeds": np.array(args.seeds, dtype=int),
            "seed_importance_means": pivot.loc[args.seeds, EXPECTED_FEATURES].to_numpy(float),
            "overall_means": overall.set_index("feature").loc[EXPECTED_FEATURES, "mean_importance"].to_numpy(float),
            "overall_sds": overall.set_index("feature").loc[EXPECTED_FEATURES, "sd_across_seeds"].to_numpy(float),
            "pairwise_raw_p": pair_df["raw_p"].to_numpy(float),
            "pairwise_holm_sidak_p": pair_df["holm_sidak_p"].to_numpy(float),
        },
    )
    dominant = overall.iloc[0]
    summary_lines = [
        "# Figure 2d current-pipeline summary",
        "",
        f"- Training seeds: {', '.join(map(str, args.seeds))}",
        f"- Permutation repeats per feature per seed: {args.permutation_repeats}",
        f"- Dominant mean feature: **{dominant['feature_display']}** ({dominant['mean_importance']:.4f} ΔR²)",
        f"- Seed-0 importance anchor max difference: {feature_anchor_max_delta:.6f}",
        "- Pairwise tests: paired two-sided t-tests across seed-level mean importances; Holm–Šidák adjusted.",
        "",
        "## Across-seed summary",
        "",
        dataframe_markdown(overall),
        "",
        "## Pairwise tests",
        "",
        dataframe_markdown(pair_df),
        "",
    ]
    (final_dir / "FIGURE2D_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def nested_learning_subsets(
    train: pd.DataFrame,
    fractions: Sequence[float],
    order_seed: int,
    order_file: Path | None,
) -> tuple[list[tuple[float, pd.DataFrame]], dict[str, Any]]:
    """Build nested real-row subsets using a prespecified row order.

    The package includes the seed-0 order recovered from the original nested
    training-size files. Because those historical files are numerically
    identical to the current 52-row development split, retaining their row
    order preserves the intended sampling design while all model fitting uses
    the current pipeline. If the order file is unavailable, a deterministic
    seed-based permutation is used and reported explicitly.
    """
    if order_file is not None and order_file.exists():
        order_table = pd.read_csv(order_file)
        required = {"order_rank", ID_COLUMN}
        if not required.issubset(order_table.columns):
            raise AnalysisError(
                f"Learning-order file must contain {sorted(required)}: {order_file}"
            )
        ordered_ids = order_table.sort_values("order_rank")[ID_COLUMN].astype(str).tolist()
        if len(ordered_ids) != len(train) or len(set(ordered_ids)) != len(train):
            raise AnalysisError("Learning-order file does not contain 52 unique IDs")
        if set(ordered_ids) != set(train[ID_COLUMN].astype(str)):
            missing = sorted(set(train[ID_COLUMN].astype(str)) - set(ordered_ids))
            extra = sorted(set(ordered_ids) - set(train[ID_COLUMN].astype(str)))
            raise AnalysisError(
                f"Learning-order IDs do not match current training IDs; missing={missing}, extra={extra}"
            )
        indexed = train.assign(**{ID_COLUMN: train[ID_COLUMN].astype(str)}).set_index(ID_COLUMN, drop=False)
        ordered_train = indexed.loc[ordered_ids].reset_index(drop=True)
        provenance = {
            "source": "packaged historical seed-0 order used for subset membership; selected rows fitted in canonical current-table order",
            "order_file": str(order_file),
            "order_file_sha256": sha256_file(order_file),
            "order_seed": order_seed,
            "ordered_ids": ordered_ids,
        }
    else:
        rng = np.random.RandomState(order_seed)
        order = rng.permutation(len(train))
        ordered_train = train.iloc[order].copy().reset_index(drop=True)
        provenance = {
            "source": "deterministic NumPy permutation fallback",
            "order_file": None,
            "order_seed": order_seed,
            "ordered_ids": ordered_train[ID_COLUMN].astype(str).tolist(),
        }

    outputs: list[tuple[float, pd.DataFrame]] = []
    previous_ids: set[str] = set()
    subset_records: list[dict[str, Any]] = []
    for fraction in fractions:
        n_rows = int(round(len(train) * fraction))
        n_rows = max(1, min(len(train), n_rows))
        # Use the prespecified historical order only to choose subset membership.
        # Fit each selected subset in the canonical current training-table order.
        # This removes an otherwise artificial row-order effect and makes the
        # 100% learning-curve point reproduce the canonical seed-0 full run.
        selected_ids = set(ordered_ids[:n_rows])
        subset = (
            train.loc[train[ID_COLUMN].astype(str).isin(selected_ids)]
            .copy()
            .reset_index(drop=True)
        )
        current_ids = set(subset[ID_COLUMN].astype(str))
        if not previous_ids.issubset(current_ids):
            raise AnalysisError("Learning subsets are not nested")
        previous_ids = current_ids
        outputs.append((fraction, subset))
        subset_records.append(
            {
                "fraction_of_development_train": fraction,
                "n_real": n_rows,
                "percent_of_total_75": 100.0 * n_rows / TOTAL_MODELING_N,
                "mAb_ids": subset[ID_COLUMN].astype(str).tolist(),
            }
        )
    provenance["subsets"] = subset_records
    return outputs, provenance


def aggregate_figure2e(paths: Paths, args: argparse.Namespace, modules: dict[str, Any]) -> None:
    final_dir = paths.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    condition_rows: list[dict[str, Any]] = []
    condition_predictions: list[pd.DataFrame] = []
    for condition in ["full", "top2", "ht_assays", "no_kd"]:
        for seed in args.seeds:
            run_dir = condition_run_dir(paths.output_dir, condition, seed)
            if not (run_dir / "run_complete.json").exists():
                raise AnalysisError(f"Missing completed ablation run: {run_dir}")
            metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
            condition_rows.append(metrics)
            condition_predictions.append(pd.read_csv(run_dir / "test_predictions.csv"))
    runs = pd.DataFrame(condition_rows)
    runs["condition_display"] = runs["condition"].map(CONDITION_DISPLAY)
    runs.to_csv(final_dir / "figure2e_ablation_runs.csv", index=False)
    pd.concat(condition_predictions, ignore_index=True).to_csv(
        final_dir / "figure2e_ablation_predictions.csv", index=False
    )
    summary = (
        runs.groupby(["condition", "condition_display"], sort=False)
        .agg(
            mean_test_r2=("r2", "mean"),
            sd_test_r2=("r2", lambda x: x.std(ddof=1)),
            mean_test_rmse=("rmse", "mean"),
            sd_test_rmse=("rmse", lambda x: x.std(ddof=1)),
            mean_test_mae=("mae", "mean"),
            sd_test_mae=("mae", lambda x: x.std(ddof=1)),
            mean_test_spearman=("spearman", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    condition_order = {condition: index for index, condition in enumerate(["full", "top2", "ht_assays", "no_kd"])}
    summary["condition_order"] = summary["condition"].map(condition_order)
    summary = summary.sort_values("condition_order").reset_index(drop=True)
    summary.to_csv(final_dir / "figure2e_ablation_summary.csv", index=False)

    pivot = runs.pivot(index="seed", columns="condition", values="r2").loc[args.seeds]
    raw_p_values: list[float] = []
    tests: list[dict[str, Any]] = []
    for condition in ["top2", "ht_assays", "no_kd"]:
        test_result = modules["ttest_rel"](pivot["full"], pivot[condition])
        raw_p_values.append(float(test_result.pvalue))
        tests.append(
            {
                "reference": "full",
                "condition": condition,
                "condition_display": CONDITION_DISPLAY[condition],
                "paired_t_statistic": float(test_result.statistic),
                "raw_p": float(test_result.pvalue),
                "mean_paired_r2_difference": float((pivot["full"] - pivot[condition]).mean()),
                "n_paired_seeds": len(pivot),
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(tests, adjusted):
        row["holm_p"] = float(adjusted_p)
        row["significance"] = significance_label(float(adjusted_p))
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(final_dir / "figure2e_ablation_tests.csv", index=False)

    learning_rows: list[dict[str, Any]] = []
    learning_predictions: list[pd.DataFrame] = []
    membership_rows: list[dict[str, Any]] = []
    for fraction in args.learning_fractions:
        n_real = int(round(EXPECTED_TRAIN_N * fraction))
        for seed in args.learning_seeds:
            run_dir = condition_run_dir(paths.output_dir, "learning", seed, n_real)
            if not (run_dir / "run_complete.json").exists():
                raise AnalysisError(f"Missing completed learning run: {run_dir}")
            metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
            metrics["requested_fraction_of_development_train"] = fraction
            metrics["percent_of_total_75"] = 100.0 * n_real / TOTAL_MODELING_N
            learning_rows.append(metrics)
            pred = pd.read_csv(run_dir / "test_predictions.csv")
            pred["requested_fraction_of_development_train"] = fraction
            pred["percent_of_total_75"] = 100.0 * n_real / TOTAL_MODELING_N
            learning_predictions.append(pred)
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            for mab_id in config["real_train_ids"]:
                membership_rows.append(
                    {
                        "seed": seed,
                        "requested_fraction_of_development_train": fraction,
                        "n_real": n_real,
                        "percent_of_total_75": 100.0 * n_real / TOTAL_MODELING_N,
                        ID_COLUMN: mab_id,
                    }
                )
    learning = pd.DataFrame(learning_rows).sort_values(["n_real", "seed"]).reset_index(drop=True)
    learning.to_csv(final_dir / "figure2e_learning_curve_runs.csv", index=False)
    pd.concat(learning_predictions, ignore_index=True).to_csv(
        final_dir / "figure2e_learning_curve_predictions.csv", index=False
    )
    pd.DataFrame(membership_rows).to_csv(final_dir / "figure2e_learning_subset_membership.csv", index=False)
    learning_summary = (
        learning.groupby(["n_real", "percent_of_total_75"], sort=True)
        .agg(
            mean_test_r2=("r2", "mean"),
            sd_test_r2=("r2", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            mean_test_rmse=("rmse", "mean"),
            mean_test_mae=("mae", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    learning_summary.to_csv(final_dir / "figure2e_learning_curve_summary.csv", index=False)

    r2_values = learning_summary["mean_test_r2"].to_numpy(float)
    monotonic_non_decreasing = bool(np.all(np.diff(r2_values) >= -1e-12))
    last_three_slope = float(np.polyfit(learning_summary["n_real"].to_numpy(float)[-3:], r2_values[-3:], 1)[0])
    full_mean = float(summary.loc[summary["condition"].eq("full"), "mean_test_r2"].iloc[0])
    no_kd_mean = float(summary.loc[summary["condition"].eq("no_kd"), "mean_test_r2"].iloc[0])
    claim_check = {
        "learning_curve_monotonic_non_decreasing": monotonic_non_decreasing,
        "learning_curve_last_three_slope_r2_per_real_row": last_three_slope,
        "full_mean_test_r2": full_mean,
        "no_kd_mean_test_r2": no_kd_mean,
        "full_minus_no_kd_mean_r2": full_mean - no_kd_mean,
        "no_kd_holm_p": float(tests_df.loc[tests_df["condition"].eq("no_kd"), "holm_p"].iloc[0]),
        "top2_mean_test_r2": float(summary.loc[summary["condition"].eq("top2"), "mean_test_r2"].iloc[0]),
        "ht_assays_mean_test_r2": float(summary.loc[summary["condition"].eq("ht_assays"), "mean_test_r2"].iloc[0]),
    }
    write_json(final_dir / "figure2e_claim_check.json", claim_check)

    modules["savemat"](
        final_dir / "Figure2e_current.mat",
        {
            "learning_n_real": learning_summary["n_real"].to_numpy(int),
            "learning_percent_total": learning_summary["percent_of_total_75"].to_numpy(float),
            "learning_r2_mean": learning_summary["mean_test_r2"].to_numpy(float),
            "learning_r2_sd": learning_summary["sd_test_r2"].to_numpy(float),
            "ablation_conditions": np.array([CONDITION_DISPLAY[c] for c in ["full", "top2", "ht_assays", "no_kd"]], dtype=object),
            "ablation_seed_r2": pivot[["full", "top2", "ht_assays", "no_kd"]].to_numpy(float),
            "ablation_r2_mean": summary.set_index("condition").loc[["full", "top2", "ht_assays", "no_kd"], "mean_test_r2"].to_numpy(float),
            "ablation_r2_sd": summary.set_index("condition").loc[["full", "top2", "ht_assays", "no_kd"], "sd_test_r2"].to_numpy(float),
            "ablation_raw_p": tests_df["raw_p"].to_numpy(float),
            "ablation_holm_p": tests_df["holm_p"].to_numpy(float),
        },
    )
    summary_lines = [
        "# Figure 2e current-pipeline summary",
        "",
        "## Feature ablations",
        "",
        dataframe_markdown(summary),
        "",
        "## Paired tests versus full",
        "",
        dataframe_markdown(tests_df),
        "",
        "## Learning curve",
        "",
        dataframe_markdown(learning_summary),
        "",
        "## Automated claim checks",
        "",
        f"- Monotonic non-decreasing mean learning curve: **{monotonic_non_decreasing}**",
        f"- Slope across last three learning points: {last_three_slope:.6f} R² per real row",
        f"- Full minus no-kD mean R²: {full_mean - no_kd_mean:.4f}",
        f"- No-kD Holm-adjusted paired P: {claim_check['no_kd_holm_p']:.6g}",
        "",
        "Do not finalize the Results wording until these current outputs are reviewed.",
        "",
    ]
    (final_dir / "FIGURE2E_RESULTS_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def write_checksums(root: Path, output_path: Path, exclude_weights: bool = True) -> None:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output_path:
            continue
        if exclude_weights and (path.name.endswith(".weights.h5") or path.suffix in {".pkl"}):
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure2d(paths: Paths, args: argparse.Namespace, modules: dict[str, Any], preflight: dict[str, Any]) -> None:
    train = pd.read_csv(paths.train_csv)
    test = pd.read_csv(paths.test_csv)
    for seed in args.seeds:
        result = run_experiment(
            paths,
            args,
            modules,
            condition="full",
            features=FEATURE_SETS["full"],
            seed=seed,
            real_train=train,
            test=test,
            n_synthetic=args.full_synthetic_rows,
            need_models=True,
        )
        if seed == 0:
            comparison = anchor_comparison(paths, args, result)
            print(f"[anchor] {json.dumps(comparison, indent=2)}")
            if not comparison["passed"] and not args.allow_anchor_mismatch:
                raise AnalysisError(
                    "The seed-0 full model did not reproduce the current Figure 2a anchor within tolerance. "
                    "Stop here and upload ANCHOR_COMPARISON_SEED0.json before running more models."
                )
        compute_permutation_for_seed(paths, args, modules, result, test[ID_COLUMN].tolist())
        if seed == 0:
            importance_comparison = check_seed0_importance_anchor(paths, args, result.run_dir)
            print(f"[importance anchor] {json.dumps(importance_comparison, indent=2)}")
            if not importance_comparison["passed"] and not args.allow_anchor_mismatch:
                raise AnalysisError(
                    "The seed-0 permutation-importance means did not reproduce the current saved vector. "
                    "Stop here before training seeds 1-4 and upload ANCHOR_IMPORTANCE_COMPARISON_SEED0.json."
                )
        result.models = None
        modules["tf"].keras.backend.clear_session()
        gc.collect()
    aggregate_figure2d(paths, args, modules)


def run_figure2e(paths: Paths, args: argparse.Namespace, modules: dict[str, Any]) -> None:
    train = pd.read_csv(paths.train_csv)
    test = pd.read_csv(paths.test_csv)
    # The full condition is reused from Figure 2d when available. If Figure 2e
    # is launched first, the same full runs are created here.
    for condition in ["full", "top2", "ht_assays", "no_kd"]:
        for seed in args.seeds:
            result = run_experiment(
                paths,
                args,
                modules,
                condition=condition,
                features=FEATURE_SETS[condition],
                seed=seed,
                real_train=train,
                test=test,
                n_synthetic=args.full_synthetic_rows,
                need_models=False,
            )
            if condition == "full" and seed == 0:
                comparison = anchor_comparison(paths, args, result)
                if not comparison["passed"] and not args.allow_anchor_mismatch:
                    raise AnalysisError(
                        "The full seed-0 run did not reproduce the Figure 2a anchor. "
                        "Do not continue the ablations."
                    )

    packaged_order = Path(__file__).resolve().with_name("learning_subset_order_seed0.csv")
    order_file = Path(args.learning_order_file).expanduser().resolve() if args.learning_order_file else packaged_order
    subsets, order_provenance = nested_learning_subsets(
        train, args.learning_fractions, args.learning_subset_order_seed, order_file
    )
    write_json(paths.output_dir / "LEARNING_SUBSET_ORDER_PROVENANCE.json", order_provenance)
    for fraction, subset in subsets:
        n_synthetic = int(round(len(subset) * args.learning_synthetic_ratio))
        n_synthetic = max(1, n_synthetic)
        for seed in args.learning_seeds:
            run_experiment(
                paths,
                args,
                modules,
                condition="learning",
                features=FEATURE_SETS["full"],
                seed=seed,
                real_train=subset,
                test=test,
                n_synthetic=n_synthetic,
                need_models=False,
            )
    aggregate_figure2e(paths, args, modules)


def summarize_existing(paths: Paths, args: argparse.Namespace, modules: dict[str, Any]) -> None:
    aggregate_figure2d(paths, args, modules)
    aggregate_figure2e(paths, args, modules)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "figure2d", "figure2e", "all", "summarize"], required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--mainpack")
    parser.add_argument("--train-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--check-ml", action="store_true", help="Import/version-check ML dependencies during preflight")
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4"))
    parser.add_argument("--learning-seeds", type=parse_int_list, default=parse_int_list("0"))
    parser.add_argument(
        "--learning-fractions",
        type=parse_float_list,
        default=parse_float_list("0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"),
    )
    parser.add_argument("--learning-subset-order-seed", type=int, default=0)
    parser.add_argument(
        "--learning-order-file",
        help="Optional CSV containing order_rank and mAb_id for the nested learning subsets; "
        "defaults to the packaged seed-0 order file.",
    )
    parser.add_argument("--learning-synthetic-ratio", type=float, default=0.5)
    parser.add_argument("--full-synthetic-rows", type=int, default=26)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=32)
    parser.add_argument("--num-transformer-blocks", type=int, default=1)
    parser.add_argument("--mlp-units", type=parse_int_list, default=parse_int_list("64"))
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--l2-reg", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--force", action="store_true", help="Rerun completed condition/seed folders")
    parser.add_argument("--allow-anchor-mismatch", action="store_true")
    parser.add_argument("--anchor-r2", type=float, default=0.740886152354125)
    parser.add_argument("--anchor-rmse", type=float, default=4.766305895799887)
    parser.add_argument("--anchor-r2-tolerance", type=float, default=0.02)
    parser.add_argument("--anchor-rmse-tolerance", type=float, default=0.25)
    parser.add_argument("--anchor-prediction-tolerance", type=float, default=1.0)
    parser.add_argument("--anchor-importance-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"figure2d", "figure2e", "all", "summarize"} and 0 not in args.seeds:
        raise SystemExit("Seed 0 must be included in --seeds so the current Figure 2a anchor can be checked.")
    if args.learning_synthetic_ratio <= 0:
        raise SystemExit("--learning-synthetic-ratio must be greater than zero.")
    paths = build_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.output_dir / "RUN_LOG.txt"
    with log_path.open("a", encoding="utf-8") as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_handle)
        sys.stderr = Tee(old_stderr, log_handle)
        try:
            print("\n" + "=" * 88)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Figure 2d/2e mode={args.mode}")
            print(f"Script version: {SCRIPT_VERSION}")
            print(f"Repository: {paths.repo_root}")
            print(f"MainPack: {paths.mainpack}")
            print(f"Output: {paths.output_dir}")
            preflight = validate_inputs(paths, check_ml=args.check_ml or args.mode != "preflight")
            if args.mode == "preflight":
                if args.check_ml and any(value == "NOT INSTALLED" for value in preflight["package_versions"].values()):
                    raise AnalysisError("One or more required ML packages are not installed; see PACKAGE_VERSIONS.json")
                print("Preflight completed successfully. No model was run.")
                return 0

            modules = load_ml_modules(paths)
            versions = package_versions(include_ml=True)
            write_json(paths.output_dir / "PACKAGE_VERSIONS.json", versions)
            required_versions = {
                "tensorflow": "2.15.0",
                "keras": "2.15.0",
                "scikit-learn": "1.4.1.post1",
                "sdv": "1.17.4",
                "tfkan": "0.1.0",
                "ImbalancedLearningRegression": "0.0.2",
            }
            version_warnings = {
                name: {"expected": expected, "found": versions.get(name)}
                for name, expected in required_versions.items()
                if versions.get(name) != expected
            }
            write_json(paths.output_dir / "VERSION_WARNINGS.json", version_warnings)
            if version_warnings:
                print("[warning] Package versions differ from the manuscript environment:")
                print(json.dumps(version_warnings, indent=2))

            if args.mode in {"figure2d", "all"}:
                run_figure2d(paths, args, modules, preflight)
            if args.mode in {"figure2e", "all"}:
                run_figure2e(paths, args, modules)
            if args.mode == "summarize":
                summarize_existing(paths, args, modules)

            write_checksums(paths.output_dir / "final", paths.output_dir / "final" / "CHECKSUMS.sha256")
            print(f"Completed mode={args.mode}")
            return 0
        except AnalysisError as exc:
            print(f"\nCONTROLLED STOP: {exc}")
            (paths.output_dir / "CONTROLLED_STOP.txt").write_text(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{exc}\n", encoding="utf-8"
            )
            return 2
        except Exception as exc:  # pragma: no cover - user environment dependent
            print(f"\nUNEXPECTED ERROR: {exc!r}")
            traceback.print_exc()
            (paths.output_dir / "UNEXPECTED_ERROR.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            return 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    raise SystemExit(main())
