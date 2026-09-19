""" Gaussian Copula real-vs-synthetic diagnostics.

This script fits SDV Gaussian Copula synthesizers only on seed0 training rows
for viscosity and clearance, then compares the generated rows with the real
training assay manifold. It does not import TensorFlow or train ACeT models.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 20260630


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    train_path: Path
    columns: tuple[str, ...]
    synthetic_multiplier_num: int
    synthetic_multiplier_den: int


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Data/curated").is_dir():
            return parent
    raise RuntimeError("Could not locate ACeT repository root")


ROOT = repo_root()
DATA_ROOT = ROOT / "Data/fixed_splits"
OUT_DIR = ROOT / "reruns/gaussian_copula"


ENDPOINTS = (
    EndpointConfig(
        name="viscosity",
        train_path=DATA_ROOT / "viscosity" / "DataS1_viscosity_seed0_train.csv",
        columns=(
            "DLS Interaction Parameter kD (mL/g)",
            "SE-UHPLC Main Peak Plates (EP)",
            "AC-SINS λmax (nm)",
            "SE-UHPLC Main Peak FWHM (min)",
            "Viscosity",
        ),
        synthetic_multiplier_num=1,
        synthetic_multiplier_den=2,
    ),
    EndpointConfig(
        name="clearance",
        train_path=DATA_ROOT / "mouse_exposure" / "DataS2_clearance_seed0_train.csv",
        columns=(
            "Heparin_RT",
            "Heparin_pB_buffer",
            "BVP_high",
            "poly_D_lysine",
            "AUCt",
        ),
        synthetic_multiplier_num=2,
        synthetic_multiplier_den=1,
    ),
)


def import_sdv():
    try:
        import sdv
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        return sdv, SingleTableMetadata, GaussianCopulaSynthesizer
    except Exception as exc:  # pragma: no cover - used for controlled stop
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "missing_dependency",
            "message": "SDV/GaussianCopulaSynthesizer is unavailable; no diagnostics were run.",
            "error": repr(exc),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (OUT_DIR / "missing_dependency_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "run_log.txt").write_text(
            "Gaussian Copula diagnostics stopped before synthesis.\n"
            f"Reason: {repr(exc)}\n",
            encoding="utf-8",
        )
        raise SystemExit(2) from exc


def require_columns(df: pd.DataFrame, columns: Iterable[str], endpoint: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{endpoint}: missing required columns: {missing}")


def synthetic_n(real_n: int, cfg: EndpointConfig) -> int:
    return int(real_n * cfg.synthetic_multiplier_num / cfg.synthetic_multiplier_den)


def make_synthetic(
    real: pd.DataFrame,
    n_rows: int,
    SingleTableMetadata,
    GaussianCopulaSynthesizer,
) -> pd.DataFrame:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real)
    synth = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
    )
    synth.fit(real)
    synthetic = synth.sample(num_rows=n_rows)
    return synthetic.loc[:, real.columns].apply(pd.to_numeric, errors="coerce")


def save_heatmap(matrix: pd.DataFrame, title: str, path_png: Path, path_pdf: Path | None = None) -> None:
    fig_w = max(7.0, 0.8 * len(matrix.columns) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    if path_pdf is not None:
        fig.savefig(path_pdf)
    plt.close(fig)


def save_difference_heatmap(matrix: pd.DataFrame, title: str, path_png: Path) -> None:
    fig_w = max(7.0, 0.8 * len(matrix.columns) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako_r",
        vmin=0,
        vmax=max(0.5, float(np.nanmax(matrix.to_numpy()))),
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Absolute Spearman rho difference"},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def save_pca_plot(real: pd.DataFrame, synthetic: pd.DataFrame, endpoint: str) -> dict[str, float]:
    combined = pd.concat([real, synthetic], ignore_index=True)
    labels = np.array(["Real training"] * len(real) + ["Synthetic"] * len(synthetic))
    scaled = StandardScaler().fit_transform(combined)
    pca = PCA(n_components=2, random_state=SEED)
    scores = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "source": labels,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="source",
        style="source",
        s=72,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title(f"{endpoint.capitalize()} real-vs-synthetic PCA")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{endpoint}_real_vs_synthetic_pca.png", dpi=300)
    fig.savefig(OUT_DIR / f"{endpoint}_real_vs_synthetic_pca.pdf")
    plt.close(fig)

    return {
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_variance": float(pca.explained_variance_ratio_[1]),
    }


def marginal_statistics(real: pd.DataFrame, synthetic: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    for col in real.columns:
        real_values = real[col].dropna().to_numpy()
        synthetic_values = synthetic[col].dropna().to_numpy()
        ks = stats.ks_2samp(real_values, synthetic_values, method="auto")
        rows.append(
            {
                "endpoint": endpoint,
                "variable": col,
                "real_n": len(real_values),
                "synthetic_n": len(synthetic_values),
                "real_mean": float(np.mean(real_values)),
                "synthetic_mean": float(np.mean(synthetic_values)),
                "real_sd": float(np.std(real_values, ddof=1)),
                "synthetic_sd": float(np.std(synthetic_values, ddof=1)),
                "real_median": float(np.median(real_values)),
                "synthetic_median": float(np.median(synthetic_values)),
                "real_iqr": float(np.percentile(real_values, 75) - np.percentile(real_values, 25)),
                "synthetic_iqr": float(
                    np.percentile(synthetic_values, 75) - np.percentile(synthetic_values, 25)
                ),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_distance": float(
                    stats.wasserstein_distance(real_values, synthetic_values)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_marginal_plot(real: pd.DataFrame, synthetic: pd.DataFrame, endpoint: str) -> None:
    plot_df = pd.concat(
        [
            real.assign(source="Real training"),
            synthetic.assign(source="Synthetic"),
        ],
        ignore_index=True,
    )
    long_df = plot_df.melt(id_vars="source", var_name="variable", value_name="value")
    n_cols = 2
    n_rows = int(np.ceil(real.shape[1] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.5 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, variable in zip(axes_flat, real.columns):
        subset = long_df[long_df["variable"] == variable]
        sns.histplot(
            data=subset,
            x="value",
            hue="source",
            stat="density",
            common_norm=False,
            element="step",
            bins="auto",
            alpha=0.35,
            ax=ax,
        )
        ax.set_title(variable)
        ax.set_xlabel("")
    for ax in axes_flat[len(real.columns) :]:
        ax.axis("off")
    fig.suptitle(f"{endpoint.capitalize()} marginal distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{endpoint}_marginal_distributions.png", dpi=300)
    plt.close(fig)


def nearest_neighbor_statistics(real: pd.DataFrame, synthetic: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    scaler = StandardScaler().fit(real)
    real_scaled = scaler.transform(real)
    synthetic_scaled = scaler.transform(synthetic)

    real_nn = NearestNeighbors(n_neighbors=2).fit(real_scaled)
    real_distances = real_nn.kneighbors(real_scaled, return_distance=True)[0][:, 1]

    synth_nn = NearestNeighbors(n_neighbors=1).fit(real_scaled)
    synthetic_distances = synth_nn.kneighbors(synthetic_scaled, return_distance=True)[0][:, 0]

    rows = []
    for label, values in (
        ("real_to_real", real_distances),
        ("synthetic_to_real", synthetic_distances),
    ):
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        rows.append(
            {
                "endpoint": endpoint,
                "distance_type": label,
                "n": len(values),
                "median": float(np.median(values)),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "p95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def save_nearest_neighbor_plot(real: pd.DataFrame, synthetic: pd.DataFrame, endpoint: str) -> None:
    scaler = StandardScaler().fit(real)
    real_scaled = scaler.transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    real_distances = NearestNeighbors(n_neighbors=2).fit(real_scaled).kneighbors(
        real_scaled, return_distance=True
    )[0][:, 1]
    synthetic_distances = NearestNeighbors(n_neighbors=1).fit(real_scaled).kneighbors(
        synthetic_scaled, return_distance=True
    )[0][:, 0]
    plot_df = pd.DataFrame(
        {
            "distance": np.concatenate([real_distances, synthetic_distances]),
            "distance_type": ["Real-to-real"] * len(real_distances)
            + ["Synthetic-to-real"] * len(synthetic_distances),
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    sns.boxplot(data=plot_df, x="distance_type", y="distance", width=0.5, ax=ax)
    sns.stripplot(
        data=plot_df,
        x="distance_type",
        y="distance",
        color="0.2",
        alpha=0.55,
        size=4,
        jitter=0.18,
        ax=ax,
    )
    ax.set_title(f"{endpoint.capitalize()} nearest-neighbor distances")
    ax.set_xlabel("")
    ax.set_ylabel("Standardized Euclidean distance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{endpoint}_nearest_neighbor_distances.png", dpi=300)
    plt.close(fig)


def coverage_fraction(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    scaled = StandardScaler().fit_transform(pd.concat([real, synthetic], ignore_index=True))
    scores = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
    real_scores = scores[: len(real)]
    synthetic_scores = scores[len(real) :]
    mins = real_scores.min(axis=0)
    maxs = real_scores.max(axis=0)
    inside = np.logical_and(synthetic_scores >= mins, synthetic_scores <= maxs).all(axis=1)
    return float(np.mean(inside))


def mean_pairwise_distance_ratio(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    scaler = StandardScaler().fit(real)
    real_scaled = scaler.transform(real)
    synthetic_scaled = scaler.transform(synthetic)
    real_pairwise = pairwise_distances(real_scaled)
    real_pairwise = real_pairwise[np.triu_indices_from(real_pairwise, k=1)]
    synthetic_to_real = pairwise_distances(synthetic_scaled, real_scaled).ravel()
    return float(np.mean(synthetic_to_real) / np.mean(real_pairwise))


def interpret_endpoint(summary_row: dict[str, float], marginal_df: pd.DataFrame, nn_df: pd.DataFrame) -> str:
    endpoint = summary_row["endpoint"]
    coverage = summary_row["pca_synthetic_inside_real_pc_range_fraction"]
    mean_abs_corr = summary_row["mean_absolute_correlation_difference"]
    max_ks = marginal_df["ks_statistic"].max()
    median_real = float(nn_df.loc[nn_df["distance_type"] == "real_to_real", "median"].iloc[0])
    median_synth = float(nn_df.loc[nn_df["distance_type"] == "synthetic_to_real", "median"].iloc[0])
    distance_ratio = median_synth / median_real if median_real > 0 else np.nan

    pca_text = (
        "Synthetic points largely overlap the real training PCA range"
        if coverage >= 0.8
        else "Synthetic points partially overlap the real training PCA range"
    )
    marginal_text = (
        "marginal distributions are broadly preserved"
        if max_ks <= 0.35
        else "some marginals show visible distributional shifts"
    )
    corr_text = (
        "pairwise rank correlations are closely preserved"
        if mean_abs_corr <= 0.2
        else "some pairwise rank correlations differ enough to warrant cautious wording"
    )
    nn_text = (
        "synthetic-to-real distances are comparable to real-to-real distances"
        if distance_ratio <= 1.5
        else "synthetic-to-real distances are larger than real-to-real distances"
    )
    return (
        f"For {endpoint}, {pca_text.lower()} in the first two principal components "
        f"({coverage:.2f} of synthetic rows inside the real PC1/PC2 range). The "
        f"{marginal_text}; the largest KS statistic is {max_ks:.2f}. The mean "
        f"absolute Spearman correlation difference is {mean_abs_corr:.2f}, so "
        f"{corr_text}. Median nearest-neighbor distances indicate that {nn_text} "
        f"(synthetic/real median ratio {distance_ratio:.2f}). These diagnostics are "
        "descriptive and limited by the small endpoint-specific training sets."
    )


def analyze_endpoint(cfg: EndpointConfig, SingleTableMetadata, GaussianCopulaSynthesizer) -> tuple[dict, str]:
    train = pd.read_csv(cfg.train_path)
    require_columns(train, cfg.columns, cfg.name)
    real = train.loc[:, cfg.columns].apply(pd.to_numeric, errors="coerce")
    if real.isna().any().any():
        raise ValueError(f"{cfg.name}: required analysis columns contain non-numeric or missing values")

    n_synthetic = synthetic_n(len(real), cfg)
    synthetic = make_synthetic(real, n_synthetic, SingleTableMetadata, GaussianCopulaSynthesizer)

    sample_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "endpoint": cfg.name,
                    "source": "real_training",
                    "row_id": train.get("mAb_id", pd.Series(range(len(real)))).astype(str).to_numpy(),
                }
            ).join(real.reset_index(drop=True)),
            pd.DataFrame(
                {
                    "endpoint": cfg.name,
                    "source": "synthetic",
                    "row_id": [f"{cfg.name}_synthetic_{idx:03d}" for idx in range(len(synthetic))],
                }
            ).join(synthetic.reset_index(drop=True)),
        ],
        ignore_index=True,
    )
    sample_table.to_csv(OUT_DIR / f"{cfg.name}_real_vs_synthetic_samples.csv", index=False)

    pca_metrics = save_pca_plot(real, synthetic, cfg.name)
    save_marginal_plot(real, synthetic, cfg.name)
    save_nearest_neighbor_plot(real, synthetic, cfg.name)

    real_corr = real.corr(method="spearman")
    synthetic_corr = synthetic.corr(method="spearman")
    corr_diff = (real_corr - synthetic_corr).abs()
    real_corr.to_csv(OUT_DIR / f"{cfg.name}_correlation_real.csv")
    synthetic_corr.to_csv(OUT_DIR / f"{cfg.name}_correlation_synthetic.csv")
    corr_diff.to_csv(OUT_DIR / f"{cfg.name}_correlation_difference.csv")

    save_heatmap(
        real_corr,
        f"{cfg.name.capitalize()} real Spearman correlations",
        OUT_DIR / f"{cfg.name}_correlation_real.png",
    )
    save_heatmap(
        synthetic_corr,
        f"{cfg.name.capitalize()} synthetic Spearman correlations",
        OUT_DIR / f"{cfg.name}_correlation_synthetic.png",
    )
    save_difference_heatmap(
        corr_diff,
        f"{cfg.name.capitalize()} absolute correlation differences",
        OUT_DIR / f"{cfg.name}_correlation_difference.png",
    )

    marginal_df = marginal_statistics(real, synthetic, cfg.name)
    nn_df = nearest_neighbor_statistics(real, synthetic, cfg.name)
    marginal_df.to_csv(OUT_DIR / f"{cfg.name}_marginal_statistics.csv", index=False)
    nn_df.to_csv(OUT_DIR / f"{cfg.name}_nearest_neighbor_statistics.csv", index=False)

    summary_row = {
        "endpoint": cfg.name,
        "train_path": str(cfg.train_path.relative_to(ROOT)),
        "real_training_n": len(real),
        "synthetic_n": len(synthetic),
        "synthetic_ratio": f"{cfg.synthetic_multiplier_num}:{cfg.synthetic_multiplier_den}",
        "mean_absolute_correlation_difference": float(
            corr_diff.where(~np.eye(len(corr_diff), dtype=bool)).stack().mean()
        ),
        "maximum_absolute_correlation_difference": float(
            corr_diff.where(~np.eye(len(corr_diff), dtype=bool)).stack().max()
        ),
        "maximum_ks_statistic": float(marginal_df["ks_statistic"].max()),
        "mean_ks_statistic": float(marginal_df["ks_statistic"].mean()),
        "mean_wasserstein_distance": float(marginal_df["wasserstein_distance"].mean()),
        "pca_synthetic_inside_real_pc_range_fraction": coverage_fraction(real, synthetic),
        "standardized_synthetic_to_real_pairwise_distance_ratio": mean_pairwise_distance_ratio(
            real, synthetic
        ),
        **pca_metrics,
    }
    interpretation = interpret_endpoint(summary_row, marginal_df, nn_df)
    return summary_row, interpretation


def write_markdown_summary(summary_df: pd.DataFrame, interpretations: list[str]) -> None:
    lines = [
        "# Gaussian Copula real-vs-synthetic diagnostics",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Scope: SDV Gaussian Copula diagnostic samples were fit only on seed0 training rows for each endpoint. No ACeT model, TensorFlow training, or `model.fit` call was used.",
        "",
        "## Summary metrics",
        "",
        summary_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "##  interpretation",
        "",
    ]
    lines.extend(f"- {text}" for text in interpretations)
    lines.extend(
        [
            "",
            "Overall interpretation: these diagnostics support whether the synthetic rows occupy the same broad assay manifold as the real training rows, but they should not be treated as proof of perfect generative fidelity. The small viscosity and clearance training sets limit the precision of KS, Wasserstein, correlation, and nearest-neighbor summaries.",
            "",
            "## Files",
            "",
            "Each endpoint has PCA, correlation, marginal-distribution, nearest-neighbor, sample, and statistic outputs saved in this directory as CSV plus PNG/PDF figures where requested.",
        ]
    )
    (OUT_DIR / "synthetic_diagnostics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_log(summary_df: pd.DataFrame, sdv_version: str) -> None:
    payload = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "sdv_version": sdv_version,
        "python_version": sys.version,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "outputs_dir": str(OUT_DIR),
        "training_performed": False,
        "tensorflow_imported": False,
        "acet_model_trained": False,
        "synthesizer_fit_scope": "seed0 training rows only for each endpoint",
        "endpoints": summary_df.to_dict(orient="records"),
    }
    lines = [
        "Gaussian Copula real-vs-synthetic diagnostics run log",
        json.dumps(payload, indent=2),
    ]
    (OUT_DIR / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir.resolve()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise SystemExit("Choose an empty output directory.")
    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="notebook")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sdv, SingleTableMetadata, GaussianCopulaSynthesizer = import_sdv()

    summary_rows = []
    interpretations = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for cfg in ENDPOINTS:
            summary_row, interpretation = analyze_endpoint(
                cfg, SingleTableMetadata, GaussianCopulaSynthesizer
            )
            summary_rows.append(summary_row)
            interpretations.append(interpretation)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "synthetic_diagnostics_summary.csv", index=False)
    write_markdown_summary(summary_df, interpretations)
    write_run_log(summary_df, getattr(sdv, "__version__", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
