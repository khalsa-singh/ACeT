#!/usr/bin/env python3
"""Regenerate Figure 2b, 3a and 3c numeric plot sources from archived predictions and metrics."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = next(p for p in Path(__file__).resolve().parents if (p/'Data/curated').is_dir())
OUT = None  # Set only from an explicit output directory in main().
VISCOSITY_CSV = ROOT/'MainPack/viscosity/results/primary/viscosity_predictions.csv'
CLEARANCE_3A_MAT = ROOT/'MainPack/mouse_exposure/results/primary/clearance_panel3A_data.mat'
CLEARANCE_3C_MAT = ROOT/'FigureSources/main/Figure_3/panel_c/data/figure3c_current_baselines.mat'


def array_fingerprint(a: np.ndarray) -> str:
    """SHA256 over dtype, shape, and canonical C-order array bytes."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot, float(np.sqrt(np.mean(residual**2))), float(np.mean(np.abs(residual)))


def build_figure2b() -> dict[str, str]:
    out = OUT / "figure2b"
    out.mkdir(parents=True, exist_ok=True)
    packaged_csv = out / "figure2b_current_source_predictions.csv"
    shutil.copyfile(VISCOSITY_CSV, packaged_csv)

    data = np.genfromtxt(VISCOSITY_CSV, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    y_true = np.asarray(data["True_Viscosity"], dtype=np.float64)
    y_pred = np.asarray(data["Predicted_Viscosity"], dtype=np.float64)
    r2, rmse, mae = metric_values(y_true, y_pred)

    labels = ["Low (<=15 cP)", "Mid (>15 to <=30 cP)", "High (>30 cP)"]
    masks = [y_true <= 15.0, (y_true > 15.0) & (y_true <= 30.0), y_true > 30.0]
    rng = np.random.default_rng(0)
    samples = np.empty((1000, 3), dtype=np.float64)
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for j, (label, mask) in enumerate(zip(labels, masks)):
        yt, yp = y_true[mask], y_pred[mask]
        for i in range(1000):
            idx = rng.integers(0, yt.size, size=yt.size)
            samples[i, j] = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2))
            bootstrap_rows.append({"range": label, "bootstrap_index": i + 1, "rmse_cp": f"{samples[i, j]:.17g}"})
        summary_rows.append({
            "range": label,
            "n": yt.size,
            "true_mean_cp": f"{np.mean(yt):.17g}",
            "predicted_mean_cp": f"{np.mean(yp):.17g}",
            "observed_rmse_cp": f"{np.sqrt(np.mean((yt - yp) ** 2)):.17g}",
            "bootstrap_rmse_mean_cp": f"{np.mean(samples[:, j]):.17g}",
            "bootstrap_rmse_sd_cp": f"{np.std(samples[:, j], ddof=1):.17g}",
            "bootstrap_rmse_ci2_5_cp": f"{np.percentile(samples[:, j], 2.5):.17g}",
            "bootstrap_rmse_ci97_5_cp": f"{np.percentile(samples[:, j], 97.5):.17g}",
        })
    write_csv(out / "figure2b_current_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(out / "figure2b_current_bootstrap_samples.csv", ["range", "bootstrap_index", "rmse_cp"], bootstrap_rows)

    savemat(out / "figure2b_current_fixed_ranges.mat", {
        "y_true": y_true[:, None], "y_pred": y_pred[:, None],
        "range_labels": np.asarray(labels, dtype=object)[None, :],
        "bin_counts": np.asarray([m.sum() for m in masks], dtype=np.int64)[None, :],
        "true_means": np.asarray([y_true[m].mean() for m in masks])[None, :],
        "pred_means": np.asarray([y_pred[m].mean() for m in masks])[None, :],
        "observed_rmse": np.asarray([np.sqrt(np.mean((y_true[m] - y_pred[m])**2)) for m in masks])[None, :],
        "bootstrap_rmse_samples": samples,
        "bootstrap_rmse_mean": samples.mean(axis=0)[None, :],
        "bootstrap_seed": np.asarray([[0]], dtype=np.int64),
        "bootstrap_replicates_per_range": np.asarray([[1000]], dtype=np.int64),
        "overall_r2": np.asarray([[r2]]), "overall_rmse": np.asarray([[rmse]]), "overall_mae": np.asarray([[mae]]),
    }, do_compression=True)

    matlab = """%% Figure 2b: reference fixed viscosity ranges
% No training. Input values come only from the packaged current prediction CSV.
data = load('figure2b_current_fixed_ranges.mat');
labels = cellstr(data.range_labels); rmseMean = data.bootstrap_rmse_mean(:);
figure('Color','w','Units','pixels','Position',[100 100 760 650]); hold on; grid on;
plot([0 50],[0 50],'k--','LineWidth',1.5);
errorbar(data.true_means(:),data.pred_means(:),rmseMean,'o','MarkerSize',12, ...
 'MarkerFaceColor',[0.20 0.70 0.90],'MarkerEdgeColor','k','LineWidth',2, ...
 'Color',[0 0.45 0.74],'CapSize',15,'LineStyle','none');
rng(double(data.bootstrap_seed));
for i=1:3
 s=data.bootstrap_rmse_samples(:,i); jitter=(rand(size(s))-0.5)*0.10;
 scatter(data.true_means(i)+jitter,data.pred_means(i)+s,10,'k','filled','MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05);
 scatter(data.true_means(i)+jitter,data.pred_means(i)-s,10,'k','filled','MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05);
 text(data.true_means(i)+1.4,data.pred_means(i),sprintf('%s (n=%d)',labels{i},data.bin_counts(i)), ...
  'FontSize',16,'FontWeight','bold','HorizontalAlignment','left');
end
axis square; xlim([0 50]); ylim([0 50]); xlabel('Mean true viscosity (cP)'); ylabel('Mean predicted viscosity (cP)');
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
"""
    (out / "Figure2b_section_replacement_current.m").write_text(matlab, encoding="utf-8", newline="\n")
    report = f"""# Figure 2b validation report

- Authoritative source: `{VISCOSITY_CSV.relative_to(ROOT).as_posix()}`
- Source/package CSV SHA256: `{file_sha256(VISCOSITY_CSV)}` / `{file_sha256(packaged_csv)}` (exact byte copy: **{file_sha256(VISCOSITY_CSV) == file_sha256(packaged_csv)}**)
- `y_true` fingerprint: `{array_fingerprint(y_true)}`
- `y_pred` fingerprint: `{array_fingerprint(y_pred)}`
- Rows: {y_true.size}; overall R2: {r2:.12f}; RMSE: {rmse:.12f} cP; MAE: {mae:.12f} cP
- Ranges: Low `<=15`, Mid `>15 and <=30`, High `>30` cP.
- Bootstrap: 1,000 paired within-range resamples per range; NumPy `default_rng`, seed 0.
- Validation: generated MAT arrays equal the parsed current CSV arrays element-for-element. The archived prediction CSV is the numerical input.
- Reference-source: **yes**. Training run: **no**.
"""
    (out / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    check = loadmat(out / "figure2b_current_fixed_ranges.mat")
    assert np.array_equal(check["y_true"].ravel(), y_true) and np.array_equal(check["y_pred"].ravel(), y_pred)
    return {"y_true": array_fingerprint(y_true), "y_pred": array_fingerprint(y_pred)}


def empirical_cdf_rank_marker(values: np.ndarray, percent: float) -> float:
    """Match the final Figure 3a MATLAB empirical-CDF marker convention."""
    ordered = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if ordered.size == 0:
        raise ValueError("Cannot calculate a CDF marker from an empty array.")
    one_based = max(1, int(np.floor(ordered.size * percent / 100.0 + 0.5)))
    one_based = min(int(ordered.size), one_based)
    return float(ordered[one_based - 1])


def build_figure3a() -> dict[str, str]:
    out = OUT / "figure3a"; out.mkdir(parents=True, exist_ok=True)
    source = loadmat(CLEARANCE_3A_MAT)
    keys = ["trues", "preds", "x", "y", "r2", "rmse", "nrmse", "dens", "pct_err", "sorted_pct", "cum_percent"]
    copied = {k: source[k].copy() for k in keys}
    trues, preds = source["trues"].ravel(), source["preds"].ravel()
    pct = np.abs(preds.astype(np.float64) - trues) / trues * 100.0
    percentiles = np.asarray([empirical_cdf_rank_marker(pct, q) for q in (50, 75, 90)], dtype=np.float64)
    copied.update({"pct_error_median": percentiles[0], "pct_error_p75": percentiles[1], "pct_error_p90": percentiles[2]})
    mat_path = out / "figure3a_current_parity_cdf.mat"; savemat(mat_path, copied, do_compression=True)
    write_csv(out / "figure3a_current_source_predictions.csv", ["index", "y_true", "y_pred", "percentage_error"], [
        {"index": i + 1, "y_true": f"{trues[i]:.17g}", "y_pred": f"{preds[i]:.17g}", "percentage_error": f"{pct[i]:.17g}"} for i in range(trues.size)])
    write_csv(out / "figure3a_current_summary.csv", ["n", "r2", "rmse", "nrmse", "percentage_error_median", "percentage_error_p75", "percentage_error_p90"], [{
        "n": trues.size, "r2": f"{source['r2'].item():.17g}", "rmse": f"{source['rmse'].item():.17g}", "nrmse": f"{source['nrmse'].item():.17g}",
        "percentage_error_median": f"{percentiles[0]:.17g}", "percentage_error_p75": f"{percentiles[1]:.17g}", "percentage_error_p90": f"{percentiles[2]:.17g}"}])
    matlab = """%% Figure 3a: reference clearance parity and error CDF
% No training. All plotted arrays are loaded from the reference MAT.
data=load('figure3a_current_parity_cdf.mat');
figure('Color','w','Units','normalized','Position',[0.10 0.15 0.80 0.55]);
subplot(1,2,1); scatter(data.trues,data.preds,80,data.dens,'filled','MarkerEdgeColor','k'); hold on;
lims=[min([data.trues(:);data.preds(:)]) max([data.trues(:);data.preds(:)])]; plot(lims,lims,'k--','LineWidth',1.5); axis square;
xlabel('Measured clearance'); ylabel('Predicted clearance'); title(sprintf('R^2 = %.3f',data.r2)); colorbar;
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(1,2,2); plot(data.sorted_pct,data.cum_percent,'o-','Color',[0.20 0.60 0.80],'MarkerFaceColor',[0.20 0.60 0.80],'LineWidth',2);
xlabel('Absolute percentage error (%)'); ylabel('Cumulative observations (%)'); ylim([0 100]); grid on;
set(gca,'FontSize',16,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
"""
    (out / "Figure3a_section_replacement_current.m").write_text(matlab, encoding="utf-8", newline="\n")
    reloaded = loadmat(mat_path)
    assert all(np.array_equal(reloaded[k], source[k]) for k in keys)
    assert trues.size == 11 and np.isclose(source["r2"].item(), .810019, atol=1e-6) and np.isclose(source["rmse"].item(), 107.750765, atol=1e-6) and np.isclose(source["nrmse"].item(), .143226, atol=1e-6)
    lines = [f"- `{k}`: `{array_fingerprint(source[k])}`" for k in keys]
    report = f"""# Figure 3a validation report

- Authoritative source: `{CLEARANCE_3A_MAT.relative_to(ROOT).as_posix()}`
- Source file SHA256: `{file_sha256(CLEARANCE_3A_MAT)}`
- n = 11; R2 = {source['r2'].item():.12f}; RMSE = {source['rmse'].item():.12f}; nRMSE = {source['nrmse'].item():.12f}.
- Percentage-error CDF markers (empirical-rank convention matching the final Figure 3a MATLAB producer): median {percentiles[0]:.12f}%, p75 {percentiles[1]:.12f}%, p90 {percentiles[2]:.12f}%.
- Every copied array, including every `trues`/`preds` element, is exactly equal after MAT round-trip.
- Reference-source: **yes**. Historical hard-coded predictions used: **no**. Training run: **no**.

## Array fingerprints
{chr(10).join(lines)}
"""
    (out / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return {k: array_fingerprint(source[k]) for k in keys}


def build_figure3c() -> dict[str, str]:
    out = OUT / "figure3c"; out.mkdir(parents=True, exist_ok=True)
    source = loadmat(CLEARANCE_3C_MAT)
    keys = [k for k in source if not k.startswith("__")]
    copied = {k: source[k].copy() for k in keys}
    models = ["ACeT", "Ridge", "SVR", "Random forest"]
    metrics = ["r2", "rmse", "mae", "nrmse", "nmae"]
    stems = ["transformer", "ridge", "svr", "rf"]
    for metric in metrics:
        copied[f"cv_{metric}"] = np.column_stack([source[f"cv_{metric}_{stem}"].ravel() for stem in stems])
    copied["model_labels"] = np.asarray(models, dtype=object)[None, :]
    mat_path = out / "figure3c_current_baselines.mat"; savemat(mat_path, copied, do_compression=True)
    fold_rows = []
    for metric in metrics:
        for j, model in enumerate(models):
            for fold, value in enumerate(copied[f"cv_{metric}"][:, j], 1):
                fold_rows.append({"model": model, "metric": metric.upper(), "fold": fold, "value": f"{value:.17g}"})
    write_csv(out / "figure3c_current_fold_metrics.csv", ["model", "metric", "fold", "value"], fold_rows)
    summary_rows = []
    for mi, metric in enumerate(metrics):
        held = source[f"test_{metric}_all"].ravel()
        for j, model in enumerate(models):
            vals = copied[f"cv_{metric}"][:, j]
            summary_rows.append({"model": model, "metric": metric.upper(), "cv_mean": f"{vals.mean():.17g}", "cv_sd": f"{vals.std(ddof=1):.17g}", "held_out": f"{held[j]:.17g}"})
    write_csv(out / "figure3c_current_summary.csv", ["model", "metric", "cv_mean", "cv_sd", "held_out"], summary_rows)
    matlab = """%% Figure 3c: reference clearance baseline comparison
% No training. Bars are held-out metrics; diamonds/fold points are current CV arrays.
data=load('figure3c_current_baselines.mat'); models={'ACeT','Ridge','SVR','Random forest'};
barColors=[.20 .60 .80;.75 .85 .95;.75 .85 .95;.75 .85 .95]; rng(0); jitter=(rand(5,4)-.5)*.15;
figure('Color','w','Units','normalized','Position',[.05 .10 .90 .70]);
subplot(2,2,[1 2]); hold on;
for i=1:4, bar(i,data.test_r2_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_r2,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_r2(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
minR2=min([data.cv_r2(:);data.test_r2_all(:)]); ylim([min(-0.05,floor((minR2-.02)*10)/10) 1]); ylabel('R^2'); xticks(1:4); xticklabels(models);
set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(2,2,3); hold on;
for i=1:4, bar(i,data.test_rmse_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_rmse,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_rmse(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
ylabel('RMSE'); xticks(1:4); xticklabels(models); xtickangle(45); set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
subplot(2,2,4); hold on;
for i=1:4, bar(i,data.test_mae_all(i),'FaceColor',barColors(i,:),'BarWidth',.5,'EdgeColor','none'); end
scatter(1:4,mean(data.cv_mae,1),100,'d','MarkerEdgeColor',[.3 .3 .3],'MarkerFaceColor',[.3 .3 .3]);
for i=1:4, scatter(i+jitter(:,i),data.cv_mae(:,i),80,'o','MarkerEdgeColor',[.2 .2 .2],'MarkerFaceColor','none','LineWidth',1); end
ylabel('MAE'); xticks(1:4); xticklabels(models); xtickangle(45); set(gca,'FontSize',24,'FontName','Helvetica','LineWidth',1.5,'TickDir','out','Box','off');
"""
    (out / "Figure3c_section_replacement_current.m").write_text(matlab, encoding="utf-8", newline="\n")
    reloaded = loadmat(mat_path)
    assert all(np.array_equal(reloaded[k], source[k]) for k in keys)
    assert float(np.min(copied["cv_r2"][:, 2])) < 0
    lines = [f"- `{k}`: `{array_fingerprint(source[k])}`" for k in keys]
    report = f"""# Figure 3c validation report

- Authoritative source: `{CLEARANCE_3C_MAT.relative_to(ROOT).as_posix()}`
- Source file SHA256: `{file_sha256(CLEARANCE_3C_MAT)}`
- All {len(keys)} fold-level and held-out source arrays are copied exactly and verified after MAT round-trip.
- Standardized 5-by-4 CV matrices use columns: ACeT, Ridge, SVR, Random forest; `transformer` source fields are presentation-labeled **ACeT**.
- MATLAB R2 limit includes a negative range (current SVR minimum = {np.min(copied['cv_r2'][:, 2]):.12f}). Coordinates come from the included comparator MAT source.
- Reference-source: **yes**. Training run: **no**.

## Source-array fingerprints
{chr(10).join(lines)}
"""
    (out / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return {k: array_fingerprint(source[k]) for k in keys}


def main() -> None:
    global OUT
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();OUT=args.output_dir.resolve()
    if OUT.exists() and any(OUT.iterdir()):raise ValueError('Use an empty output directory.')
    f2 = build_figure2b(); f3a = build_figure3a(); f3c = build_figure3c()
    master = f"""# Master validation report

This is a controlled, no-training regeneration. The archived CSV/MAT records are the numerical sources; MATLAB plotting conventions are retained.

| Panel | Exact authoritative source | Representative array fingerprints | Reference-source |
|---|---|---|---|
| Figure 2b | `{VISCOSITY_CSV.relative_to(ROOT).as_posix()}` | `y_true` `{f2['y_true']}`; `y_pred` `{f2['y_pred']}` | Yes |
| Figure 3a | `{CLEARANCE_3A_MAT.relative_to(ROOT).as_posix()}` | `trues` `{f3a['trues']}`; `preds` `{f3a['preds']}` | Yes |
| Figure 3c | `{CLEARANCE_3C_MAT.relative_to(ROOT).as_posix()}` | `cv_r2_transformer` `{f3c['cv_r2_transformer']}`; `cv_r2_svr` `{f3c['cv_r2_svr']}`; `test_r2_all` `{f3c['test_r2_all']}` | Yes |

Full per-array fingerprints and elementwise/MAT-round-trip checks are recorded in each panel's `VALIDATION_REPORT.md`.

## Computation scope

No TensorFlow, model fitting, SDV, Gaussian Copula, SMOTE, ENN, SHAP, permutation importance, learning curves, ablations, or baselines were run. Only current CSV/MAT reads, deterministic arithmetic summaries/bootstrap resampling, exact array copying, CSV/MAT serialization, and MATLAB plotting-source generation were performed. The supplied CSV/MAT files remain the numerical reference.
"""
    (OUT / "MASTER_VALIDATION_REPORT.md").write_text(master, encoding="utf-8", newline="\n")
    print("Built and validated reference Figure 2b, 3a, and 3c packages; no training performed.")


if __name__ == "__main__":
    main()
