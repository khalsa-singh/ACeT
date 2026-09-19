#!/usr/bin/env python3
"""Mouse IV exposure quick start using archived Figure 3a predictions.

No model training is performed. The script uses only the Python standard
library, recomputes the reported held-out metrics, and writes an SVG.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

EXPECTED = {
    "n": 11,
    "r2": 0.8100192636753217,
    "rmse": 107.75076495769723,
    "nrmse": 0.14322599500240224,
    "mae": 79.29346974875709,
    "nmae": 0.1053995867817487,
    "spearman_rho": 0.9090909090909091,
    "percentage_error_median": 6.0102016927682165,
    "percentage_error_p75": 16.584972137515315,
    "percentage_error_p90": 40.42320446649249,
}


def find_package_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "MANIFEST.csv").is_file() and (candidate / "FigureSources").is_dir():
            return candidate
    raise FileNotFoundError("Could not find package root containing MANIFEST.csv and FigureSources.")


def empirical_cdf_rank_marker(values: list[float], percent: float) -> float:
    """Match the final Figure 3a MATLAB CDF-marker convention.

    The manuscript producer sorts the observed errors and uses the 1-based
    index round(n * percent / 100). For the 11-antibody test set this gives
    the 6th, 8th, and 10th ordered errors for 50%, 75%, and 90%.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a CDF marker from an empty list.")
    one_based = max(1, math.floor(len(ordered) * percent / 100.0 + 0.5))
    one_based = min(len(ordered), one_based)
    return ordered[one_based - 1]


def rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        average_rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[order[k]] = average_rank
        pos = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mx = statistics.mean(x)
    my = statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return numerator / denominator


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float | int]:
    n = len(y_true)
    mean_true = statistics.mean(y_true)
    sse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    sst = sum((a - mean_true) ** 2 for a in y_true)
    rmse = math.sqrt(sse / n)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    pct = [abs(a - b) / abs(a) * 100.0 for a, b in zip(y_true, y_pred)]
    return {
        "n": n,
        "r2": 1.0 - sse / sst,
        "rmse": rmse,
        "nrmse": rmse / mean_true,
        "mae": mae,
        "nmae": mae / mean_true,
        "spearman_rho": pearson(rank_average(y_true), rank_average(y_pred)),
        "percentage_error_median": empirical_cdf_rank_marker(pct, 50),
        "percentage_error_p75": empirical_cdf_rank_marker(pct, 75),
        "percentage_error_p90": empirical_cdf_rank_marker(pct, 90),
    }


def validate_metrics(metrics: dict[str, float | int]) -> None:
    for key, expected in EXPECTED.items():
        actual = metrics[key]
        tolerance = 0.0 if key == "n" else 1e-9 * max(1.0, abs(float(expected)))
        if abs(float(actual) - float(expected)) > tolerance:
            raise RuntimeError(f"Metric mismatch for {key}: expected {expected}, got {actual}")


def write_svg(path: Path, y_true: list[float], y_pred: list[float], metrics: dict[str, float | int]) -> None:
    width, height = 980, 500
    left_x, top_y, panel_w, panel_h = 75, 80, 360, 330
    right_x = 565
    values = y_true + y_pred
    log_min = math.log10(min(values) * 0.88)
    log_max = math.log10(max(values) * 1.08)

    def map_x(v: float) -> float:
        return left_x + (math.log10(v) - log_min) / (log_max - log_min) * panel_w

    def map_y(v: float) -> float:
        return top_y + panel_h - (math.log10(v) - log_min) / (log_max - log_min) * panel_h

    pct = sorted(abs(a - b) / abs(a) * 100.0 for a, b in zip(y_true, y_pred))
    pct_max = max(50.0, math.ceil(max(pct) / 10.0) * 10.0)

    def cdf_x(v: float) -> float:
        return right_x + v / pct_max * panel_w

    def cdf_y(v: float) -> float:
        return top_y + panel_h - v / 100.0 * panel_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.title{font-size:20px;font-weight:bold}.axis{font-size:13px}.small{font-size:12px}.metric{font-size:14px}</style>',
        '<text x="40" y="34" class="title">ACeT mouse IV exposure — quick start</text>',
        '<text x="40" y="57" class="small">Metrics recalculated from 11 archived held-out predictions; parity axes use a logarithmic scale.</text>',
        f'<rect x="{left_x}" y="{top_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222"/>',
        f'<rect x="{right_x}" y="{top_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222"/>',
        f'<line x1="{map_x(10**log_min):.2f}" y1="{map_y(10**log_min):.2f}" x2="{map_x(10**log_max):.2f}" y2="{map_y(10**log_max):.2f}" stroke="#555" stroke-width="1.5"/>',
    ]

    for observed, predicted in zip(y_true, y_pred):
        lines.append(
            f'<circle cx="{map_x(observed):.2f}" cy="{map_y(predicted):.2f}" r="5" fill="#2f6f9f" stroke="white" stroke-width="1"/>'
        )

    for tick in [300, 500, 800, 1200]:
        if 10**log_min <= tick <= 10**log_max:
            x = map_x(tick)
            y = map_y(tick)
            lines.extend([
                f'<line x1="{x:.2f}" y1="{top_y+panel_h}" x2="{x:.2f}" y2="{top_y+panel_h+6}" stroke="#222"/>',
                f'<text x="{x:.2f}" y="{top_y+panel_h+23}" text-anchor="middle" class="axis">{tick}</text>',
                f'<line x1="{left_x-6}" y1="{y:.2f}" x2="{left_x}" y2="{y:.2f}" stroke="#222"/>',
                f'<text x="{left_x-10}" y="{y+4:.2f}" text-anchor="end" class="axis">{tick}</text>',
            ])

    lines.extend([
        f'<text x="{left_x+panel_w/2}" y="{height-25}" text-anchor="middle" class="axis">Observed AUCt (stored scale)</text>',
        f'<text x="20" y="{top_y+panel_h/2}" text-anchor="middle" class="axis" transform="rotate(-90 20 {top_y+panel_h/2})">Predicted AUCt (stored scale)</text>',
        f'<text x="{left_x+12}" y="{top_y+24}" class="metric">R² = {float(metrics["r2"]):.3f}</text>',
        f'<text x="{left_x+12}" y="{top_y+45}" class="metric">nRMSE = {float(metrics["nrmse"]):.3f}</text>',
        f'<text x="{left_x+12}" y="{top_y+66}" class="metric">Spearman ρ = {float(metrics["spearman_rho"]):.3f}</text>',
    ])

    points = [(0.0, 0.0)] + [(value, (i + 1) / len(pct) * 100.0) for i, value in enumerate(pct)]
    path_d = " ".join(
        ("M" if i == 0 else "L") + f" {cdf_x(x):.2f} {cdf_y(y):.2f}"
        for i, (x, y) in enumerate(points)
    )
    lines.append(f'<path d="{path_d}" fill="none" stroke="#2f6f9f" stroke-width="2.5"/>')

    for tick in range(0, int(pct_max) + 1, 10):
        x = cdf_x(float(tick))
        lines.extend([
            f'<line x1="{x:.2f}" y1="{top_y+panel_h}" x2="{x:.2f}" y2="{top_y+panel_h+6}" stroke="#222"/>',
            f'<text x="{x:.2f}" y="{top_y+panel_h+23}" text-anchor="middle" class="axis">{tick}</text>',
        ])
    for tick in [0, 25, 50, 75, 100]:
        y = cdf_y(float(tick))
        lines.extend([
            f'<line x1="{right_x-6}" y1="{y:.2f}" x2="{right_x}" y2="{y:.2f}" stroke="#222"/>',
            f'<text x="{right_x-10}" y="{y+4:.2f}" text-anchor="end" class="axis">{tick}</text>',
        ])

    lines.extend([
        f'<text x="{right_x+panel_w/2}" y="{height-25}" text-anchor="middle" class="axis">Absolute percentage error (%)</text>',
        f'<text x="{right_x-48}" y="{top_y+panel_h/2}" text-anchor="middle" class="axis" transform="rotate(-90 {right_x-48} {top_y+panel_h/2})">Cumulative antibodies (%)</text>',
        f'<text x="{right_x+12}" y="{top_y+24}" class="metric">Median = {float(metrics["percentage_error_median"]):.2f}%</text>',
        f'<text x="{right_x+12}" y="{top_y+45}" class="metric">75% CDF marker = {float(metrics["percentage_error_p75"]):.2f}%</text>',
        f'<text x="{right_x+12}" y="{top_y+66}" class="metric">90% CDF marker = {float(metrics["percentage_error_p90"]):.2f}%</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate clearance held-out metrics and an SVG from locked predictions; no model training."
    )
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--formats", default="svg", help="Comma-separated svg,png,jpg formats.")
    parser.add_argument("--raster-engine", choices=["auto","cairo","pillow"], default="auto")
    args = parser.parse_args()

    formats = set(x.strip().lower() for x in args.formats.split(","))
    if not formats or not formats <= {"svg","png","jpg","jpeg"}:
        parser.error("Formats must be selected from svg,png,jpg.")
    engine = args.raster_engine
    if formats & {"png","jpg","jpeg"}:
        try:
            from PIL import Image
        except ImportError:
            parser.error("Raster output needs: python -m pip install -r requirements-demo.txt")
        if engine in {"auto","cairo"}:
            try:
                import cairosvg
                engine = "cairo"
            except (ImportError, OSError):
                if engine == "cairo": parser.error("CairoSVG/Cairo unavailable; use --raster-engine pillow.")
                engine = "pillow"
    package_root = args.package_root.resolve() if args.package_root else find_package_root(Path(__file__).resolve())
    input_csv = package_root / "FigureSources/main/Figure_3/panel_a/data/figure3a_current_source_predictions.csv"
    if not input_csv.is_file():
        raise FileNotFoundError(f"Locked prediction table not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y_true = [float(row["y_true"]) for row in rows]
    y_pred = [float(row["y_pred"]) for row in rows]
    metrics = compute_metrics(y_true, y_pred)
    validate_metrics(metrics)

    print("Mouse-exposure demonstration: archived metrics reproduced.")
    print(f"n = {metrics['n']}")
    print(f"R2 = {float(metrics['r2']):.6f}")
    print(f"RMSE = {float(metrics['rmse']):.6f}")
    print(f"nRMSE = {float(metrics['nrmse']):.6f}")
    print(f"MAE = {float(metrics['mae']):.6f}")
    print(f"Spearman rho = {float(metrics['spearman_rho']):.6f}")
    print(
        "CDF markers (50/75/90%) = "
        f"{float(metrics['percentage_error_median']):.2f}% / "
        f"{float(metrics['percentage_error_p75']):.2f}% / "
        f"{float(metrics['percentage_error_p90']):.2f}%"
    )

    if args.check_only:
        return 0

    output_dir = args.output_dir.resolve() if args.output_dir else package_root / "reruns/demo"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "clearance_demo_metrics.json"
    svg_path = output_dir / "clearance_demo_parity_and_cdf.svg"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(svg_path, y_true, y_pred, metrics)
    if formats & {"png","jpg","jpeg"}:
        if engine == "cairo":
            import io
            raster = cairosvg.svg2png(url=str(svg_path), output_width=2940, output_height=1500)
            image = Image.open(io.BytesIO(raster)).convert("RGB")
        else:
            from rasterize_demo import render
            image = render(svg_path)
        if "png" in formats: image.save(svg_path.with_suffix(".png"))
        if formats & {"jpg","jpeg"}: image.save(svg_path.with_suffix(".jpg"), quality=95, subsampling=0)
        print("Raster renderer:", engine)
    # The SVG is kept as the vector master for all requested raster renditions.
    print(f"Metrics: {metrics_path}")
    print(f"Figure:  {svg_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CLEARANCE QUICK DEMO: FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
