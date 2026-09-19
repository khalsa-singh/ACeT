#!/usr/bin/env python
"""
Reproduce the training-only Train44 and Train42 clearance datasets.

This script reads:
  00_source/mouse2.csv
  00_source/mouse2_row_mapping.csv

For each development cohort independently:
  1. Detect 1.5*IQR outliers in all nine assay features using development rows only.
  2. Replace those feature cells with NaN.
  3. Run 10-iteration IterativeImputer on the nine features plus AUC as an
     auxiliary predictor; restore AUC unchanged.
  4. Rank the nine assays by absolute Pearson correlation with AUC.
  5. Retain the top four.
  6. Fit QuantileTransformer(normal) on the top-four development matrix.
  7. Replace cells with |normal score| > 5 by NaN in original feature scale.
  8. Run 10-iteration IterativeImputer on the four features only.
  9. Export the same raw, untouched 11-antibody manuscript test set.

No held-out feature or target value is used to fit preprocessing or rank features.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "00_source" / "mouse2.csv"
MAPPING = ROOT / "00_source" / "mouse2_row_mapping.csv"
REPRO = ROOT / "REPRODUCED_OUTPUTS"

TARGET_RAW = "AUC (ng∙h/mL)"
TARGET_MODEL = "AUCt"
FEATURES = [
    "FcRn_RT",
    "AC-SINS DiffC (µm²/s)",
    "Heparin_RT",
    "Heparin_pB_buffer",
    "BVP_high",
    "membrane_prep",
    "poly_D_lysine",
    "PEI",
    "SE-UHPLC % LMW, t = 4 wk 40C, 1 mg/mL A52Su",
]
ALIAS = {
    "FcRn_RT": "FcRn_RT",
    "AC-SINS DiffC (µm²/s)": "AC_SINS_DiffC",
    "Heparin_RT": "Heparin_RT",
    "Heparin_pB_buffer": "Heparin_pB_buffer",
    "BVP_high": "BVP_high",
    "membrane_prep": "membrane_prep",
    "poly_D_lysine": "poly_D_lysine",
    "PEI": "PEI",
    "SE-UHPLC % LMW, t = 4 wk 40C, 1 mg/mL A52Su": "SEC_LMW_4wk_40C",
}
TEST_ROWS = {2, 6, 12, 16, 26, 31, 34, 37, 41, 49, 51}
OUTLIER_ROWS = {39, 52}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[c]) for c in columns] for row in rows], dtype=float)


def first_stage(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = matrix(rows, FEATURES)
    y = matrix(rows, [TARGET_RAW]).reshape(-1)
    mask = np.zeros_like(x, dtype=bool)

    for j in range(x.shape[1]):
        q1, q3 = np.quantile(x[:, j], [0.25, 0.75], method="linear")
        iqr = q3 - q1
        mask[:, j] = (
            (x[:, j] < q1 - 1.5 * iqr)
            | (x[:, j] > q3 + 1.5 * iqr)
        )

    work = np.column_stack([x.copy(), y])
    work[:, : len(FEATURES)][mask] = np.nan
    imputer = IterativeImputer(
        max_iter=10,
        initial_strategy="mean",
        imputation_order="ascending",
        sample_posterior=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imputer.fit_transform(work)
    return imputed[:, : len(FEATURES)], y.copy()


def ranking(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for j, feature in enumerate(FEATURES):
        r_value, p_value = pearsonr(x[:, j], y)
        rows.append({
            "Feature": feature,
            "Alias": ALIAS[feature],
            "Pearson_r": float(r_value),
            "Absolute_r": float(abs(r_value)),
            "Univariate_R2": float(r_value ** 2),
            "p_value_nominal": float(p_value),
        })
    rows.sort(key=lambda row: row["Absolute_r"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["Rank"] = rank
    return rows


def second_stage(x: np.ndarray, selected: list[str]) -> np.ndarray:
    indices = [FEATURES.index(feature) for feature in selected]
    x4 = x[:, indices].copy()
    qt = QuantileTransformer(
        n_quantiles=len(x4),
        output_distribution="normal",
        random_state=0,
    )
    z = qt.fit_transform(x4)
    x4[np.abs(z) > 5.0] = np.nan

    imputer = IterativeImputer(
        max_iter=10,
        initial_strategy="mean",
        imputation_order="ascending",
        sample_posterior=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return imputer.fit_transform(x4)


def main() -> None:
    global REPRO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'rerun_outputs' / 'preprocessing')
    parser.add_argument('--include-train44', action='store_true', help='Also reproduce the 44-row sensitivity cohort.')
    args = parser.parse_args()
    REPRO = args.output_dir.resolve()
    raw = read_csv(SOURCE)
    mapping = read_csv(MAPPING)
    if len(raw) != 55 or len(mapping) != 55:
        raise RuntimeError("Expected 55 rows in both source files.")

    map_by_row = {int(row["mouse2_row"]): row for row in mapping}
    enriched = []
    for row_number, source_row in enumerate(raw, 1):
        mapped = map_by_row[row_number]
        row: dict[str, Any] = {
            "mouse2_row": row_number,
            "mAb_name": mapped["mAb_name"],
            "original_DataS2_mAb_id": mapped["original_DataS2_mAb_id"],
        }
        row.update({column: float(source_row[column]) for column in FEATURES + [TARGET_RAW]})
        enriched.append(row)

    test = [row for row in enriched if row["mouse2_row"] in TEST_ROWS]
    train44 = [row for row in enriched if row["mouse2_row"] not in TEST_ROWS]
    train42 = [
        row for row in train44
        if row["mouse2_row"] not in OUTLIER_ROWS
    ]

    if REPRO.exists() and any(REPRO.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {REPRO}. Choose a new --output-dir.")
    REPRO.mkdir(parents=True, exist_ok=True)

    cohorts = [("train42", train42)]
    if args.include_train44:
        cohorts.insert(0, ("train44", train44))
    for label, development in cohorts:
        x1, y = first_stage(development)
        ranked = ranking(x1, y)
        selected = [row["Feature"] for row in ranked[:4]]
        x2 = second_stage(x1, selected)

        folder = REPRO / label
        folder.mkdir(parents=True)

        write_csv(
            folder / f"{label}_ranking.csv",
            ranked,
            ["Rank", "Feature", "Alias", "Pearson_r", "Absolute_r",
             "Univariate_R2", "p_value_nominal"],
        )

        train_model = []
        for i in range(len(development)):
            row = {ALIAS[feature]: float(x2[i, j]) for j, feature in enumerate(selected)}
            row[TARGET_MODEL] = float(y[i])
            train_model.append(row)

        test_model = []
        for source_row in test:
            row = {ALIAS[feature]: float(source_row[feature]) for feature in selected}
            row[TARGET_MODEL] = float(source_row[TARGET_RAW])
            test_model.append(row)

        columns = [ALIAS[feature] for feature in selected] + [TARGET_MODEL]
        write_csv(folder / f"{label}_MODEL_READY_TRAIN.csv", train_model, columns)
        write_csv(folder / f"{label}_MODEL_READY_TEST11_RAW_UNTOUCHED.csv", test_model, columns)
        (folder / f"{label}_selected_top4.json").write_text(
            json.dumps({
                "development_rows": len(development),
                "selected_top4": selected,
                "aliases": [ALIAS[feature] for feature in selected],
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("Development-only preprocessing completed; archived inputs and results were not modified.")
    print(REPRO)


if __name__ == "__main__":
    main()
