#!/usr/bin/env python3
"""Extract and verify Figure 3b locked plotting arrays without training."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


LABEL = "DERIVED VERBATIM FROM THE LOCKED MATLAB PLOTTING SOURCE; NOT AN ORIGINAL MODEL-OUTPUT FILE."
EXPECTED = {"augmented": 0.7979254, "unaugmented": 0.5164624}


def extract_vector(text: str, name: str) -> list[float]:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find MATLAB vector {name}")
    body = re.sub(r"\.\.\..*?(?:\r?\n|$)", " ", match.group(1))
    values = [float(token) for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", body)]
    if len(values) != 11:
        raise ValueError(f"{name} has {len(values)} values; expected exactly 11")
    return values


def r2(observed: list[float], predicted: list[float]) -> float:
    mean_observed = sum(observed) / len(observed)
    return 1.0 - sum((o - p) ** 2 for o, p in zip(observed, predicted)) / sum(
        (o - mean_observed) ** 2 for o in observed
    )


def main() -> None:
    package_root = next(p for p in Path(__file__).resolve().parents if (p/"Data/curated").is_dir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matlab-source",
        type=Path,
        default=package_root / "FigureSources/main/Figure_3/panels_b_d/code/plot_mouse_exposure_panels.m",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "reruns/figure3b/FIGURE3B_LOCKED_PLOTTING_ARRAYS_DERIVED.csv",
    )
    args = parser.parse_args()

    text = args.matlab_source.read_text(encoding="utf-8")
    observed = extract_vector(text, "obsB")
    augmented = extract_vector(text, "predAugB")
    unaugmented = extract_vector(text, "predNoAugB")

    augmented_r2 = r2(observed, augmented)
    unaugmented_r2 = r2(observed, unaugmented)
    if not math.isclose(augmented_r2, EXPECTED["augmented"], abs_tol=1e-6):
        raise ValueError(f"Augmented R2 {augmented_r2:.10f} does not match the locked anchor")
    if not math.isclose(unaugmented_r2, EXPECTED["unaugmented"], abs_tol=1e-6):
        raise ValueError(f"Unaugmented R2 {unaugmented_r2:.10f} does not match the locked anchor")

    if args.output.exists():raise FileExistsError("Use a new output filename.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# {LABEL}\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "row_index",
                "observed",
                "augmented_prediction",
                "unaugmented_prediction",
                "augmented_absolute_error",
                "unaugmented_absolute_error",
                "augmentation_improved_absolute_error",
            ]
        )
        for index, (obs, aug, no_aug) in enumerate(zip(observed, augmented, unaugmented), start=1):
            aug_error = abs(obs - aug)
            no_aug_error = abs(obs - no_aug)
            writer.writerow([index, f"{obs:.17g}", f"{aug:.17g}", f"{no_aug:.17g}", f"{aug_error:.17g}", f"{no_aug_error:.17g}", str(aug_error <= no_aug_error).upper()])

    print(LABEL)
    print(f"Rows: {len(observed)}")
    print(f"Augmented R2: {augmented_r2:.10f}")
    print(f"Unaugmented R2: {unaugmented_r2:.10f}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
