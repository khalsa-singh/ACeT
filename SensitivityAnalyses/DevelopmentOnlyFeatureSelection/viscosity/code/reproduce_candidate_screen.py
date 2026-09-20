"""Reproduce the viscosity candidate set ranking.

Absolute Pearson correlations are calculated from the same 52 development
antibodies and checked against the earlier univariate-association reference.
This script neither reads a test file nor trains a prediction model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
RANK_COLUMNS = [
    "Rank", "Candidate_assay", "Pearson_r", "Absolute_r", "Univariate_R2",
    "P_value_nominal", "Development_n", "Top4_in_this_scope", "Used_in_archived_model",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def near(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def checked_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Required package input does not resolve: {relative}")
    return path


def calculate(root: Path = ROOT) -> dict[str, Any]:
    config = json.loads((root / "settings/viscosity_screen_definition.json").read_text(encoding="utf-8"))
    source = checked_file(root, config["source_export"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_export_sha256"]:
        raise ValueError("The archived candidate source has changed.")
    fields, rows = read_csv(source)
    _, definition = read_csv(checked_file(root, config["candidate_definition"]))
    candidates = [row["workbook_column"] for row in definition]
    if len(candidates) != config["candidate_count"] or len(set(candidates)) != 10:
        raise ValueError("Candidate definitions must contain distinct assay entries matching the configuration.")
    if any(row["type"] != "assay_or_descriptor" for row in definition):
        raise ValueError("Unexpected non-assay candidate.")
    target_name = config["target_source_column"]
    if fields != ["mAb_id", "split_seed0", "Name"] + candidates + [target_name]:
        raise ValueError("Unexpected candidate-source schema or column order.")
    _, mapping = read_csv(checked_file(root, config["development_mapping"]))
    _, development = read_csv(checked_file(root, config["development_reference"]))
    n = config["development_count"]
    if len(rows) != n or len(mapping) != n or len(development) != n or n != 52:
        raise ValueError("Expected 52 rows in candidate source, mapping and development reference.")
    if len({row["mAb_id"] for row in rows}) != n or len({row["Name"] for row in rows}) != n:
        raise ValueError("Development antibody identifiers are not unique.")
    y = np.array([float(row[target_name]) for row in rows], dtype=float)
    if not np.isfinite(y).all() or np.unique(y).size < 2:
        raise ValueError("Invalid development viscosity target.")
    for index, (row, mapped, reference) in enumerate(zip(rows, mapping, development), 1):
        if not (row["mAb_id"] == mapped["mAb_id"] == reference["mAb_id"]):
            raise ValueError("Development identity/order mismatch.")
        if row["Name"] != mapped["workbook_name"] or int(mapped["development_row"]) != index:
            raise ValueError("Source-to-development mapping mismatch.")
        if any(r["split_seed0"] != "train" for r in (row, mapped, reference)):
            raise ValueError("A non-development observation entered the screen.")
        if not near(y[index - 1], reference["Viscosity"], 1e-10):
            raise ValueError("Development target differs from the model reference.")
        for model_col, source_col in config["model_column_aliases"].items():
            if not near(row[source_col], reference[model_col], 1e-10):
                raise ValueError(f"Model-input alias mismatch: {source_col}")
    panel = set(config["reference_panel"])
    ranking = []
    for col in candidates:
        values = np.array([float(row[col]) for row in rows], dtype=float)
        if not np.isfinite(values).all() or np.unique(values).size < 2:
            raise ValueError(f"Incomplete or constant candidate: {col}")
        coefficient, pvalue = pearsonr(values, y)
        ranking.append({
            "Candidate_assay": col, "Pearson_r": float(coefficient),
            "Absolute_r": float(abs(coefficient)), "Univariate_R2": float(coefficient ** 2),
            "P_value_nominal": float(pvalue), "Development_n": n,
            "Used_in_archived_model": col in panel,
        })
    ranking.sort(key=lambda r: -r["Absolute_r"])
    for rank, row in enumerate(ranking, 1):
        row["Rank"] = rank
        row["Top4_in_this_scope"] = rank <= 4

    prior_path = checked_file(root, config["previous_ranking_reference"])
    if hashlib.sha256(prior_path.read_bytes()).hexdigest() != config["previous_ranking_reference_sha256"]:
        raise ValueError("The earlier univariate-association reference has changed.")
    _, previous = read_csv(prior_path)
    previous.sort(key=lambda r: int(r["Rank"]))
    if len(previous) != len(ranking):
        raise ValueError("Earlier-reference row-count mismatch.")
    checks = []
    metric_names = {"Pearson_r": "Pearson r", "Absolute_r": "|r|",
                    "Univariate_R2": "Univariate R2", "P_value_nominal": "p-value"}
    for current, earlier in zip(ranking, previous):
        if current["Rank"] != int(earlier["Rank"]) or current["Candidate_assay"] != earlier["Assay / descriptor"]:
            raise ValueError("Earlier-reference rank or candidate mismatch.")
        if int(earlier["n"]) != n:
            raise ValueError("Earlier-reference development-count mismatch.")
        for current_name, earlier_name in metric_names.items():
            if not near(current[current_name], earlier[earlier_name]):
                raise ValueError(f"Earlier-reference numerical mismatch: {current['Candidate_assay']}")
        if current["Used_in_archived_model"] != (earlier["Manuscript top four"] == "Yes"):
            raise ValueError("Earlier-reference model-panel mismatch.")
        checks.append({"Rank": current["Rank"], "Candidate_assay": current["Candidate_assay"],
                       "Pearson_r_recomputed": current["Pearson_r"],
                       "Pearson_r_previous": float(earlier["Pearson r"]),
                       "Absolute_difference": abs(current["Pearson_r"] - float(earlier["Pearson r"]))})

    matrix_fields = ["mAb_id", "split_seed0", "Name"] + candidates + ["Viscosity"]
    matrix = [{"mAb_id": row["mAb_id"], "split_seed0": "train", "Name": row["Name"],
               **{col: row[col] for col in candidates}, "Viscosity": row[target_name]} for row in rows]
    summary = {
        "scope": config["scope"], "candidate_assays": len(candidates),
        "development_antibodies": n, "statistic": config["statistic"],
        "selected_top4": [row["Candidate_assay"] for row in ranking[:4]],
        "top4_matches_archived_model_panel": {row["Candidate_assay"] for row in ranking[:4]} == panel,
        "previous_reference_rows_reproduced": len(checks),
        "max_absolute_pearson_difference_from_previous": max(row["Absolute_difference"] for row in checks),
        "heldout_file_read": False, "model_retrained": False,
    }
    return {"ranking": ranking, "matrix": matrix, "matrix_fields": matrix_fields,
            "reference_checks": checks, "summary": summary}


def write_outputs(result: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("viscosity_development_only_candidate_screen.csv", "viscosity_development_only_ranking.csv"):
        write_csv(destination / name, result["ranking"], RANK_COLUMNS)
    write_csv(destination / "development_candidate_assays.csv", result["matrix"], result["matrix_fields"])
    write_csv(destination / "previous_rankings_reproduction_check.csv", result["reference_checks"],
              list(result["reference_checks"][0]))
    (destination / "screen_summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_archived_screen(root: Path = ROOT) -> dict[str, Any]:
    result = calculate(root)
    for filename in ("viscosity_development_only_candidate_screen.csv", "viscosity_development_only_ranking.csv"):
        fields, archived = read_csv(root / "results" / filename)
        if fields != RANK_COLUMNS or len(archived) != len(result["ranking"]):
            raise ValueError("Archived ranking schema/count mismatch.")
        for a, b in zip(result["ranking"], archived):
            if a["Candidate_assay"] != b["Candidate_assay"] or a["Rank"] != int(b["Rank"]) or a["Development_n"] != int(b["Development_n"]):
                raise ValueError("Archived ranking identity/count mismatch.")
            for metric in ["Pearson_r", "Absolute_r", "Univariate_R2", "P_value_nominal"]:
                if not near(a[metric], b[metric]):
                    raise ValueError("Archived ranking numerical mismatch.")
            for flag in ["Top4_in_this_scope", "Used_in_archived_model"]:
                if a[flag] != (b[flag].lower() == "true"):
                    raise ValueError("Archived panel-membership mismatch.")
    fields, matrix = read_csv(root / "data/development_candidate_assays.csv")
    if fields != result["matrix_fields"] or len(matrix) != len(result["matrix"]):
        raise ValueError("Candidate matrix schema/count mismatch.")
    for a, b in zip(result["matrix"], matrix):
        for col in fields:
            if col in {"mAb_id", "split_seed0", "Name"}:
                if a[col] != b[col]:
                    raise ValueError("Candidate matrix identity mismatch.")
            elif not near(a[col], b[col], 1e-10):
                raise ValueError("Candidate matrix value mismatch.")
    archived_summary = json.loads((root / "results/screen_summary.json").read_text(encoding="utf-8"))
    # The aggregate discrepancy is a floating-point diagnostic, not an identity.
    # Use the same absolute tolerance as the per-assay checks above. Keep counts,
    # labels, selected-assay order and Boolean flags exact; do not edit the archive.
    current_summary = result["summary"]
    if not isinstance(archived_summary, dict) or set(archived_summary) != set(current_summary):
        raise ValueError("Archived screen summary keys differ from the recalculated summary.")
    for key, current_value in current_summary.items():
        archived_value = archived_summary[key]
        if key == "max_absolute_pearson_difference_from_previous":
            tolerance = 1e-12
            valid_values = all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0.0 <= value <= tolerance
                for value in (archived_value, current_value)
            )
            matches = valid_values and near(archived_value, current_value, tolerance)
        else:
            matches = type(archived_value) is type(current_value) and archived_value == current_value
        if not matches:
            raise ValueError(
                f"Archived screen summary mismatch for {key!r}: "
                f"archived={archived_value!r}; recalculated={current_value!r}."
            )
    _, table = read_csv(root.parents[1] / "TableSources/Table_S15a_development_only_assay_rankings.csv")
    table_rows = [row for row in table if row["Endpoint"] == "Viscosity"]
    if len(table_rows) != 10:
        raise ValueError("Candidate set row count does not match the S15a viscosity source.")
    for a, b in zip(result["ranking"], table_rows):
        if a["Candidate_assay"] != b["Candidate_assay"] or a["Rank"] != int(b["Rank"]):
            raise ValueError("Table S15a candidate/rank mismatch.")
        if not near(a["Pearson_r"], b["Pearson_r"]) or not near(a["Absolute_r"], b["Absolute_r"]):
            raise ValueError("Table S15a coefficient mismatch.")
        if (b["Selected"] == "Yes") != a["Top4_in_this_scope"]:
            raise ValueError("Table S15a selected-panel mismatch.")
    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rerun_outputs/candidate_screen")
    parser.add_argument("--verify", action="store_true", help="Recalculate and compare archived evidence without writing output.")
    args = parser.parse_args()
    if args.verify:
        summary = verify_archived_screen()
    else:
        destination = args.output_dir.resolve()
        protected = [ROOT / name for name in ["code", "data", "results", "settings", "reference", "00_source"]]
        if any(destination == path or path in destination.parents for path in protected):
            raise ValueError("Choose a separate rerun output directory.")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Output directory is not empty; choose a new directory.")
        result = calculate()
        write_outputs(result, destination)
        summary = result["summary"]
    print(f"Candidate set: {summary['development_antibodies']} development antibodies.")
    print("Top four within this candidate set: " + "; ".join(summary["selected_top4"]))
    print("Candidate set reference rankings reproduced; no held-out file read and no model retrained.")


if __name__ == "__main__":
    main()
