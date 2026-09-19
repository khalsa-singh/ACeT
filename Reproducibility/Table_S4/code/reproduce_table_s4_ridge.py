#!/usr/bin/env python3
"""Reproduce Supplementary Table S4 Ridge baselines for viscosity.

This is a small, deterministic held-out analysis on the locked seed-0 split:
52 development rows and 23 held-out rows. Predictors are transformed using a
QuantileTransformer fitted only on the development rows. The target remains on
the native viscosity scale. Two fixed Ridge(alpha=1.0) models are evaluated:
(1) kD only and (2) kD + SE-UHPLC plate count.

No cross-validation, augmentation, resampling, target transformation, or model
selection is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer

KD = "DLS Interaction Parameter kD (mL/g)"
PLATES = "SE-UHPLC Main Peak Plates (EP)"
TARGET = "Viscosity"
ID_COL = "mAb_id"

EXPECTED_FULL = {
    "Ridge (kD-only)": {
        "r2": 0.2663894530150398,
        "rmse_cP": 8.019907938397107,
        "mae_cP": 5.893355705879815,
    },
    "Ridge (kD + SEC plates)": {
        "r2": 0.47836008439381883,
        "rmse_cP": 6.76272936223998,
        "mae_cP": 5.1618703752893,
    },
}

TABLE_S4_ROUNDED = {
    "Ridge (kD-only)": {"r2": 0.27, "rmse_cP": 8.02, "mae_cP": 5.89},
    "Ridge (kD + SEC plates)": {"r2": 0.48, "rmse_cP": 6.76, "mae_cP": 5.16},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover_package_root(explicit: Path | None) -> Path:
    """Locate the clean ACeT package without relying on a workstation path."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    env_root = os.environ.get("ACET_PACKAGE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend((start, *start.parents))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "MANIFEST.csv").is_file() and (candidate / "Data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the ACeT package root. Pass --package-root or set "
        "ACET_PACKAGE_ROOT. The root must contain MANIFEST.csv and Data/."
    )


def validate_inputs(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required = {ID_COL, KD, PLATES, TARGET}
    for label, frame, expected_n in (("train", train, 52), ("test", test, 23)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label}: missing required columns: {missing}")
        if len(frame) != expected_n:
            raise ValueError(f"{label}: expected {expected_n} rows, found {len(frame)}")
        if frame[ID_COL].duplicated().any():
            raise ValueError(f"{label}: duplicated mAb_id values")
        if frame[[KD, PLATES, TARGET]].isna().any().any():
            raise ValueError(f"{label}: missing numeric values in required columns")
    overlap = set(train[ID_COL]).intersection(test[ID_COL])
    if overlap:
        raise ValueError(f"Train/test antibody overlap detected: {sorted(overlap)}")


def fit_fixed_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, dict[str, float], QuantileTransformer, Ridge]:
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(train)),
        output_distribution="normal",
        random_state=0,
        copy=True,
    )
    x_train = transformer.fit_transform(train[features].to_numpy(dtype=float))
    x_test = transformer.transform(test[features].to_numpy(dtype=float))
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        solver="cholesky",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "rmse_cP": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae_cP": float(mean_absolute_error(y_test, predictions)),
    }
    return predictions, metrics, transformer, model


def write_checksums(output_dir: Path) -> None:
    files = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file() and p.name != "CHECKSUMS.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files]
    (output_dir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root", "--repo-root",
        dest="package_root",
        type=Path,
        default=None,
        help=(
            "Clean ACeT package root. If omitted, ACET_PACKAGE_ROOT and parent "
            "directories of this script/current directory are searched."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ReproducedOutputs/Table_S4 under the package root.",
    )
    args = parser.parse_args()

    package_root = discover_package_root(args.package_root)
    data_dir = package_root / "Data" / "fixed_splits" / "viscosity"
    train_path = data_dir / "DataS1_viscosity_seed0_train.csv"
    test_path = data_dir / "DataS1_viscosity_seed0_test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = (
            package_root / "ReproducedOutputs" / "Table_S4"
        ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_inputs(train, test)

    configurations = [
        ("Ridge (kD-only)", [KD]),
        ("Ridge (kD + SEC plates)", [KD, PLATES]),
    ]

    results_rows: list[dict[str, object]] = []
    prediction_table = pd.DataFrame(
        {
            "mAb_id": test[ID_COL].astype(str),
            "observed_viscosity_cP": test[TARGET].astype(float),
        }
    )
    verification_lines: list[str] = []
    overall_pass = True

    for name, features in configurations:
        predictions, metrics, transformer, model = fit_fixed_ridge(train, test, features)
        expected = EXPECTED_FULL[name]
        max_abs_diff = max(abs(metrics[key] - expected[key]) for key in expected)
        exact_pass = max_abs_diff <= 1e-8
        rounded_pass = all(
            round(metrics[key], 2) == TABLE_S4_ROUNDED[name][key]
            for key in ("r2", "rmse_cP", "mae_cP")
        )
        overall_pass = overall_pass and exact_pass and rounded_pass

        results_rows.append(
            {
                "model": name,
                "input_features": "; ".join(features),
                "train_n": len(train),
                "test_n": len(test),
                "predictor_transform": "QuantileTransformer(output_distribution='normal', fit on train only)",
                "target_transform": "none",
                "ridge_alpha": 1.0,
                "ridge_solver": "cholesky",
                "test_r2": metrics["r2"],
                "test_rmse_cP": metrics["rmse_cP"],
                "test_mae_cP": metrics["mae_cP"],
                "table_s4_r2_2dp": round(metrics["r2"], 2),
                "table_s4_rmse_2dp": round(metrics["rmse_cP"], 2),
                "table_s4_mae_2dp": round(metrics["mae_cP"], 2),
                "exact_reference_pass": exact_pass,
                "rounded_table_s4_pass": rounded_pass,
                "intercept": float(model.intercept_),
                "coefficients": json.dumps([float(value) for value in np.ravel(model.coef_)]),
            }
        )

        short = "kd_only" if len(features) == 1 else "kd_plus_plates"
        prediction_table[f"prediction_{short}_cP"] = predictions
        prediction_table[f"residual_{short}_cP"] = (
            test[TARGET].to_numpy(dtype=float) - predictions
        )
        prediction_table[f"absolute_error_{short}_cP"] = np.abs(
            test[TARGET].to_numpy(dtype=float) - predictions
        )

        verification_lines.extend(
            [
                f"## {name}",
                "",
                f"- Full-precision test R²: `{metrics['r2']:.15f}`",
                f"- Full-precision test RMSE: `{metrics['rmse_cP']:.15f} cP`",
                f"- Full-precision test MAE: `{metrics['mae_cP']:.15f} cP`",
                f"- Rounded Table S4 values: `{metrics['r2']:.2f}`, `{metrics['rmse_cP']:.2f}`, `{metrics['mae_cP']:.2f}`",
                f"- Full-precision reference match: `{'PASS' if exact_pass else 'FAIL'}`",
                f"- Published Table S4 rounded-row match: `{'PASS' if rounded_pass else 'FAIL'}`",
                "",
            ]
        )

    results = pd.DataFrame(results_rows)
    results.to_csv(output_dir / "TABLE_S4_RIDGE_RESULTS.csv", index=False)
    prediction_table.to_csv(output_dir / "TABLE_S4_RIDGE_PREDICTIONS.csv", index=False)

    input_hashes = pd.DataFrame(
        [
            {
                "role": "locked seed-0 development set",
                "path": str(train_path),
                "rows": len(train),
                "sha256": sha256(train_path),
            },
            {
                "role": "locked seed-0 held-out set",
                "path": str(test_path),
                "rows": len(test),
                "sha256": sha256(test_path),
            },
        ]
    )
    input_hashes.to_csv(output_dir / "INPUT_HASHES.csv", index=False)

    environment = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "training_performed": True,
        "analysis_type": "two deterministic Ridge fits only",
        "acet_or_tensorflow_run": False,
        "augmentation_or_resampling": False,
    }
    (output_dir / "ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    verification = [
        "# Supplementary Table S4 Ridge-baseline verification",
        "",
        f"Overall status: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "The locked seed-0 viscosity split contained 52 development rows and 23 held-out rows. For each baseline, a QuantileTransformer with normal output was fitted only on the development predictors and then applied unchanged to the held-out predictors. Ridge regression used alpha = 1.0 on the native viscosity target. No augmentation, resampling, cross-validation, target transformation, or test-set fitting was used.",
        "",
        *verification_lines,
        "## Decision",
        "",
        "The two Ridge rows in Supplementary Table S4 reproduce at full precision and round exactly to the existing displayed values. No numerical change to Table S4 is required.",
    ]
    (output_dir / "TABLE_S4_VERIFICATION.md").write_text(
        "\n".join(verification) + "\n", encoding="utf-8"
    )

    run_log = {
        "status": "PASS" if overall_pass else "FAIL",
        "package_root": str(package_root),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "models": results_rows,
    }
    (output_dir / "RUN_LOG.json").write_text(
        json.dumps(run_log, indent=2), encoding="utf-8"
    )

    write_checksums(output_dir)
    print(results[["model", "test_r2", "test_rmse_cP", "test_mae_cP"]].to_string(index=False))
    print(f"Overall verification: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Outputs: {output_dir}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
