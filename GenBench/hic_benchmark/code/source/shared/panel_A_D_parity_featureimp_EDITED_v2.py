"""
Edited version of panel_A_D_parity_featureimp.py

Key upgrades:
1) Adds Bailly-fair baselines for HIC RT (or any regression target):
   - "Bailly-style refit" baseline (LinearRegression or PLS) trained ONLY on the training data (or training folds).
5) Adds Bailly-aligned reporting:
   - Pearson r, Pearson r^2 (Bailly-style), Spearman rho, sklearn R^2, RMSE, MAE.
   - Binary classification at a cutoff (default 30) with per-class accuracies (good <= cutoff, poor > cutoff).
6) Optional out-of-fold (OOF) evaluation on a single dataset file for comparisons.

Examples:

# OOF CV on a single full dataset (e.g., Bailly 152)
python panel_A_D_parity_featureimp_EDITED_v2.py --task regression --analysis_mode oof_cv --data_file bailly2020_hicrt_no_leakage_all_features_numeric.csv --head_type mlp --cv_repeats 10

# OOF with assays-only features (exclude patch_*)
python panel_A_D_parity_featureimp_EDITED_v2.py --task regression --analysis_mode oof_cv --data_file bailly2020_hicrt_no_leakage_all_features_numeric.csv --feature_set assays_only --head_type mlp --cv_repeats 10

# OOF with Patch-only (apples-to-apples vs Bailly)
python panel_A_D_parity_featureimp_EDITED_v2.py --task regression --analysis_mode oof_cv --data_file bailly2020_hicrt_no_leakage_all_features_numeric.csv --feature_set patch_only --head_type mlp --cv_repeats 10


"""

import os
import random
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay,
    matthews_corrcoef
)
from scipy.stats import spearmanr, pearsonr
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

# Optional: PLS baseline
try:
    from sklearn.cross_decomposition import PLSRegression
    _HAVE_PLS = True
except Exception:
    _HAVE_PLS = False

# Optional: synthetic data + imbalanced regression tools
try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
    _HAVE_SDV = True
except Exception:
    _HAVE_SDV = False

try:
    import ImbalancedLearningRegression as iblr
    _HAVE_IBLR = True
except Exception:
    _HAVE_IBLR = False

from bailly_hic_model import build_transformer_model


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def pearson_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(y_true, y_pred)
    return float(r * r)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    sp, _ = spearmanr(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    return {
        "sklearn_R2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pr),
        "pearson_r2": float(pr * pr),
        "spearman_rho": float(sp),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def binary_cutoff_metrics(y_true: np.ndarray, y_pred: np.ndarray, cutoff: float = 30.0) -> dict:
    """
    Bailly-style: "good eluters" <= cutoff, "poor eluters" > cutoff.
    We treat "poor" as positive class for confusion matrix metrics.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_poor = (y_true > cutoff).astype(int)
    pred_poor = (y_pred > cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_poor, pred_poor, labels=[0, 1]).ravel()
    good_acc = tn / (tn + fp) if (tn + fp) else float("nan")  # good (<=cutoff) correctly classified
    poor_acc = tp / (tp + fn) if (tp + fn) else float("nan")  # poor (>cutoff) correctly classified

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    bal_acc = 0.5 * (good_acc + poor_acc) if (not np.isnan(good_acc) and not np.isnan(poor_acc)) else float("nan")
    mcc = float(matthews_corrcoef(true_poor, pred_poor)) if len(np.unique(true_poor)) > 1 else float("nan")

    return {
        "cutoff": float(cutoff),
        "n": int(len(y_true)),
        "n_good_true": int((true_poor == 0).sum()),
        "n_poor_true": int((true_poor == 1).sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "good_acc": float(good_acc),
        "poor_acc": float(poor_acc),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
    }


def maybe_drop_hic_leakage_columns(df: pd.DataFrame, target_col: str, guard: bool = True) -> pd.DataFrame:
    """
    Safety guard: if predicting a HIC target, drop any other columns containing 'hic'
    to prevent accidental leakage (except the target itself).
    """
    if not guard:
        return df
    if "hic" not in target_col.lower():
        return df
    drop_cols = [c for c in df.columns if ("hic" in c.lower() and c != target_col)]
    if drop_cols:
        print(f"[Leakage-guard] Dropping potential HIC leakage columns (not target): {drop_cols}")
        return df.drop(columns=drop_cols)
    return df


def resolve_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    id_col: str | None,
    drop_cols: list[str],
    feature_set: str,
    patch_cols: list[str] | None,
) -> tuple[list[str], list[str]]:
    """
    Returns (feature_cols, patch_cols_used).
    """
    cols = [c for c in df.columns if c != target_col]

    # drop id + user-specified drops
    if id_col and id_col in cols:
        cols.remove(id_col)
    for c in drop_cols:
        if c in cols:
            cols.remove(c)

    # detect patch columns
    if patch_cols is None or len(patch_cols) == 0:
        patch_cols_used = [c for c in cols if c.lower().startswith("patch_")]
    else:
        patch_cols_used = [c for c in patch_cols if c in df.columns and c != target_col]
    patch_cols_used = list(dict.fromkeys(patch_cols_used))  # dedupe preserve order

    if feature_set == "assays_only":
        feature_cols = [c for c in cols if c not in patch_cols_used]
    elif feature_set == "patch_only":
        feature_cols = patch_cols_used.copy()
    else:
        feature_cols = cols

    # keep numeric only
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[Info] Dropping non-numeric feature columns: {non_numeric}")
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    return feature_cols, patch_cols_used


def bailly_published_equation_pred(df: pd.DataFrame) -> np.ndarray | None:
    """
    Bailly 4-point QSPR equation (as published) for HIC RT prediction.

    HIC_RT-PRED = 42.23687
                  - 0.02859*(avg_patch_cdr_ion)
                  + 0.12656*(avg_patch_cdr_hyd)
                  - 0.02909*(avg_patch_hyd)
                  - 0.00949*(avg_patch_ion)

    In Bailly CSVs, these often correspond to:
      patch_cdr_ion, patch_cdr_hyd, patch_hyd, patch_ion

    Returns None if required columns are missing.
    """
    required = ["patch_cdr_ion", "patch_cdr_hyd", "patch_hyd", "patch_ion"]
    if not all(c in df.columns for c in required):
        print("[BaillyEq] Missing one of required patch columns for published equation:", required)
        return None

    return (
        42.23687
        - 0.02859 * df["patch_cdr_ion"].to_numpy()
        + 0.12656 * df["patch_cdr_hyd"].to_numpy()
        - 0.02909 * df["patch_hyd"].to_numpy()
        - 0.00949 * df["patch_ion"].to_numpy()
    ).astype(float)


def fit_bailly_style_refit_baseline(
    X_train_patch: np.ndarray,
    y_train: np.ndarray,
    method: str = "linear",
    pls_components: int = 2
):
    """
    Returns a fitted baseline model trained only on the training data.

    method:
      - "linear": LinearRegression
      - "pls": PLSRegression (requires sklearn.cross_decomposition)
    """
    method = method.lower()
    if method == "pls":
        if not _HAVE_PLS:
            raise RuntimeError("PLSRegression not available in this environment.")
        m = PLSRegression(n_components=min(pls_components, X_train_patch.shape[1]))
        m.fit(X_train_patch, y_train)
        return m
    else:
        m = LinearRegression()
        m.fit(X_train_patch, y_train)
        return m


def plot_parity(y_true, y_pred, title, out_png, log_scale=False):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.75)

    # Safe parity line
    min_val = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    max_val = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)

    if log_scale:
        # only if all positive
        if (np.min(y_true) > 0) and (np.min(y_pred) > 0):
            plt.xscale("log")
            plt.yscale("log")

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# Core: Train/test regression (legacy mode)
# --------------------------------------------------------------------------------------
def run_regression_train_test(args):
    print("=== Running Regression Task (train/test mode) ===")

    train = pd.read_csv(args.train_file)
    test  = pd.read_csv(args.test_file)

    target_col = args.target_col or train.columns[-1]
    id_col = args.id_col if args.id_col in train.columns else None

    # optional leakage guard
    train = maybe_drop_hic_leakage_columns(train, target_col, guard=args.guard_hic_leakage)
    test = maybe_drop_hic_leakage_columns(test, target_col, guard=args.guard_hic_leakage)

    # drop cols list
    drop_cols = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]

    # choose features
    patch_cols = [c.strip() for c in (args.patch_cols or "").split(",") if c.strip()] if args.patch_cols else None
    feature_cols, patch_cols_used = resolve_feature_columns(
        train, target_col=target_col, id_col=id_col, drop_cols=drop_cols,
        feature_set=args.feature_set, patch_cols=patch_cols
    )

    # -------------------------------------
    # 1) Optional synthetic augmentation
    # -------------------------------------
    if args.use_sdv:
        if not _HAVE_SDV:
            raise RuntimeError("SDV is not installed but --use_sdv was requested.")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train[[*feature_cols, target_col]])
        synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution="norm")
        synthesizer.fit(train[[*feature_cols, target_col]])
        synthetic_data = synthesizer.sample(num_rows=args.synth_rows)
        train_mix = pd.concat([train[[*feature_cols, target_col]], synthetic_data], ignore_index=True)
    else:
        train_mix = train[[*feature_cols, target_col]].copy()

    print("Original train shape:", train.shape)
    print("Train_mix shape:", train_mix.shape)
    print("Test shape:", test.shape)

    # -------------------------------------
    # 2) Global scaling (train->test)
    # -------------------------------------
    sc_X = QuantileTransformer()
    X_all = sc_X.fit_transform(train_mix[feature_cols].values)
    X_test = sc_X.transform(test[feature_cols].values)
    y_all = train_mix[target_col].values.astype(float)

    # -------------------------------------
    # 3) CV + model heads (ensemble)
    # -------------------------------------
    heads = [args.head_type] if args.head_type != "all" else ["mlp", "rbf", "spline", "kan", "interaction"]
    kf = KFold(n_splits=min(args.n_splits, len(y_all)), shuffle=True, random_state=args.seed)

    for head in heads:
        set_global_seed(args.seed)
        print(f"\n--- Head = {head.upper()} ---")

        ens_models = []
        cv_stats = []

        for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all, y_all), 1):
            print(f"Fold {fold}/{kf.get_n_splits()} (head={head})")

            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]

            # Rebalance per fold using IBLR (if available/desired)
            if args.use_iblr:
                if not _HAVE_IBLR:
                    raise RuntimeError("ImbalancedLearningRegression is not installed but --use_iblr was requested.")
                df_tr = pd.DataFrame(X_tr, columns=feature_cols)
                df_tr[target_col] = y_tr

                # ENN
                try:
                    df_tr_clean = iblr.enn(data=df_tr, y=target_col, rel_coef=args.enn_relcoef)
                except Exception as e:
                    print(f"[Fold {fold}] ENN failed ({e}); using uncleaned df_tr")
                    df_tr_clean = df_tr.copy()

                if df_tr_clean.isnull().values.any():
                    df_tr_clean = df_tr_clean.dropna()

                # DO NOT drop columns here (shape mismatch risk)

                # SMOTE (fallback)
                df_tr_bal = None
                for rel in (args.smote_relcoef, args.smote_relcoef_fallback):
                    try:
                        df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=rel)
                        break
                    except Exception as e:
                        print(f"[Fold {fold}] SMOTE failed (rel_coef={rel}) -> {e}")

                if df_tr_bal is None:
                    print(f"[Fold {fold}] Using df_tr_clean (no SMOTE)")
                    df_tr_bal = df_tr_clean.copy()

                # Force full feature set / order
                for c in feature_cols:
                    if c not in df_tr_bal.columns:
                        df_tr_bal[c] = 0.0
                df_tr_bal = df_tr_bal[feature_cols + [target_col]]

                X_tr_bal = df_tr_bal[feature_cols].values
                y_tr_bal = df_tr_bal[target_col].values.astype(float)
            else:
                X_tr_bal = X_tr
                y_tr_bal = y_tr

            # build model
            m = build_transformer_model(
                num_features=X_all.shape[1],
                task="regression",
                head_type=head,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg,
            )
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="log_cosh")

            # scale y per fold
            sc_y = QuantileTransformer()
            y_tr_s = sc_y.fit_transform(y_tr_bal.reshape(-1, 1)).reshape(-1)
            y_vl_s = sc_y.transform(y_all[vl_idx].reshape(-1, 1)).reshape(-1)

            cb = [
                EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr),
            ]

            m.fit(
                X_tr_bal, y_tr_s,
                validation_data=(X_all[vl_idx], y_vl_s),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=cb,
                verbose=args.verbose,
                shuffle=True,
            )

            preds_vl_s = m.predict(X_all[vl_idx], verbose=0).reshape(-1, 1)
            preds_vl = sc_y.inverse_transform(preds_vl_s).reshape(-1)

            fold_metrics = regression_metrics(y_all[vl_idx], preds_vl)
            cv_stats.append(fold_metrics)
            ens_models.append((m, sc_y))

        # aggregate CV
        def _avg(key): return float(np.mean([d[key] for d in cv_stats]))
        def _std(key): return float(np.std([d[key] for d in cv_stats]))

        print(f"{head.upper()} CV metrics (mean +/- std over folds):")
        for k in ["sklearn_R2", "pearson_r2", "spearman_rho", "rmse", "mae"]:
            print(f"  {k:>12s} = {_avg(k):.4f} +/- {_std(k):.4f}")

        # -------------------------------------
        # 4) Ensemble predict on test 
        # -------------------------------------
        preds_test = np.mean([
            sc.inverse_transform(mdl.predict(X_test, verbose=0).reshape(-1, 1)).reshape(-1)
            for mdl, sc in ens_models
        ], axis=0)

        trues_test = test[target_col].values.astype(float)

        acet_metrics = regression_metrics(trues_test, preds_test)
        acet_bin = binary_cutoff_metrics(trues_test, preds_test, cutoff=args.binary_cutoff)

        print("\n--- Final Test Set Metrics (ACeT) ---")
        for k, v in acet_metrics.items():
            if k != "n":
                print(f"{k:>12s}: {v:.4f}")
        print("\n--- Binary @ cutoff (ACeT) ---")
        print(f"cutoff={acet_bin['cutoff']}, good_acc={acet_bin['good_acc']:.3f}, poor_acc={acet_bin['poor_acc']:.3f}, acc={acet_bin['accuracy']:.3f}, mcc={acet_bin['mcc']:.3f}")

        # -------------------------------------
        # 5) Bailly-fair baselines (trained ONLY on train)
        # -------------------------------------
        bailly_refit_pred = None
        bailly_eq_pred = None
        baseline_metrics = {}

        if args.compute_bailly_baselines:
            # Published equation (reference)
            bailly_eq_pred = bailly_published_equation_pred(test)

            # Refit baseline on TRAIN only (fair)
            if len(patch_cols_used) >= 4 and all(c in train.columns for c in ["patch_cdr_ion","patch_cdr_hyd","patch_hyd","patch_ion"]):
                patch_cols_exact = ["patch_cdr_ion","patch_cdr_hyd","patch_hyd","patch_ion"]
            else:
                patch_cols_exact = patch_cols_used

            if patch_cols_exact and len(patch_cols_exact) >= 2:
                Xb_tr = train[patch_cols_exact].values
                yb_tr = train[target_col].values.astype(float)
                Xb_te = test[patch_cols_exact].values

                bmodel = fit_bailly_style_refit_baseline(
                    Xb_tr, yb_tr,
                    method=args.bailly_refit_method,
                    pls_components=args.bailly_pls_components
                )
                bailly_refit_pred = np.asarray(bmodel.predict(Xb_te)).reshape(-1)
                baseline_metrics["bailly_refit"] = regression_metrics(trues_test, bailly_refit_pred)
                baseline_metrics["bailly_refit_binary"] = binary_cutoff_metrics(trues_test, bailly_refit_pred, cutoff=args.binary_cutoff)
            else:
                print("[Baseline] No patch columns found -> skipping Bailly-style refit baseline.")

            if bailly_eq_pred is not None:
                baseline_metrics["bailly_equation"] = regression_metrics(trues_test, bailly_eq_pred)
                baseline_metrics["bailly_equation_binary"] = binary_cutoff_metrics(trues_test, bailly_eq_pred, cutoff=args.binary_cutoff)

        # -------------------------------------
        # 6) Save outputs (predictions table + metrics json)
        # -------------------------------------
        out_prefix = args.out_prefix or f"{target_col}_{head}"

        out_pred_csv = f"{out_prefix}_predictions.csv"
        out_metrics_json = f"{out_prefix}_metrics.json"
        out_parity_png = f"{out_prefix}_parity.png"

        pred_df = pd.DataFrame({
            "True": trues_test,
            "Pred_ACeT": preds_test,
        })

        if id_col and id_col in test.columns:
            pred_df.insert(0, id_col, test[id_col].astype(str).values)

        if bailly_refit_pred is not None:
            pred_df["Pred_BaillyRefit"] = bailly_refit_pred
        if bailly_eq_pred is not None:
            pred_df["Pred_BaillyEq"] = bailly_eq_pred

        # Optional binary columns
        pred_df["True_is_poor"] = (pred_df["True"] > args.binary_cutoff).astype(int)
        pred_df["ACeT_is_poor"] = (pred_df["Pred_ACeT"] > args.binary_cutoff).astype(int)
        if bailly_refit_pred is not None:
            pred_df["BaillyRefit_is_poor"] = (pred_df["Pred_BaillyRefit"] > args.binary_cutoff).astype(int)
        if bailly_eq_pred is not None:
            pred_df["BaillyEq_is_poor"] = (pred_df["Pred_BaillyEq"] > args.binary_cutoff).astype(int)

        pred_df.to_csv(out_pred_csv, index=False)
        print(f"Saved predictions to '{out_pred_csv}'.")

        # legacy filename for backwards compatibility
        try:
            pred_df.to_csv("viscosity_predictions.csv", index=False)
            print("Also saved legacy 'viscosity_predictions.csv' (same content).")
        except Exception:
            pass

        # parity plot (linear by default; enable log via flag)
        plot_parity(trues_test, preds_test, title=f"Parity: {target_col} (ACeT, head={head})", out_png=out_parity_png, log_scale=args.log_plot)
        print(f"Saved parity plot '{out_parity_png}'.")

        # Save metrics
        payload = {
            "target_col": target_col,
            "feature_set": args.feature_set,
            "feature_cols": feature_cols,
            "patch_cols_used": patch_cols_used,
            "head": head,
            "acet_test_metrics": acet_metrics,
            "acet_test_binary": acet_bin,
            "baselines": baseline_metrics,
            "cv_metrics_mean": {k: _avg(k) for k in ["sklearn_R2","pearson_r2","spearman_rho","rmse","mae"]},
            "cv_metrics_std": {k: _std(k) for k in ["sklearn_R2","pearson_r2","spearman_rho","rmse","mae"]},
        }
        with open(out_metrics_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to '{out_metrics_json}'.")

        # -------------------------------------
        # 7) Permutation importance (optional; ACeT only)
        # -------------------------------------
        if args.permutation_importance:
            class EnsembleRegressorFixed(BaseEstimator, RegressorMixin):
                def __init__(self, models_and_scalers):
                    self.models_and_scalers = models_and_scalers
                def fit(self, X, y=None):
                    return self
                def predict(self, X):
                    preds = np.mean([
                        sc.inverse_transform(m.predict(X, verbose=0).reshape(-1, 1)).reshape(-1)
                        for m, sc in self.models_and_scalers
                    ], axis=0)
                    return preds

            ens = EnsembleRegressorFixed(ens_models)
            perm = permutation_importance(
                ens, X_test, trues_test,
                n_repeats=args.perm_repeats,
                scoring="r2",
                random_state=args.seed
            )
            importances = perm.importances_mean
            sorted_idx = np.argsort(importances)[::-1]

            out_imp_png = f"{out_prefix}_perm_importance.png"
            plt.figure(figsize=(10, 6))
            plt.bar(np.array(feature_cols)[sorted_idx], importances[sorted_idx])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Mean decrease in R2")
            plt.title("Permutation Feature Importance (ACeT ensemble)")
            plt.tight_layout()
            plt.savefig(out_imp_png, dpi=200)
            plt.close()
            print(f"Saved permutation importance plot '{out_imp_png}'.")


# --------------------------------------------------------------------------------------
# Core: Out-of-fold CV regression on a single dataset file (fair comparisons)
# --------------------------------------------------------------------------------------

def run_regression_oof_cv(args):
    """
    Creates OUT-OF-FOLD (OOF) predictions for the entire dataset,
    and compares ACeT to a "Bailly-style refit" baseline under identical folds.

    New in this version:
      - Writes per-repeat metrics to <out_prefix>_repeat_metrics.csv
      - Stores mean±SD across repeats in the JSON output (in addition to the "OOF-ensemble" metrics
        computed on predictions averaged across repeats).
    """
    print("=== Running Regression Task (OOF CV mode) ===")

    df = pd.read_csv(args.data_file)

    target_col = args.target_col or df.columns[-1]
    id_col = args.id_col if args.id_col in df.columns else None

    df = maybe_drop_hic_leakage_columns(df, target_col, guard=args.guard_hic_leakage)

    drop_cols = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]
    patch_cols = [c.strip() for c in (args.patch_cols or "").split(",") if c.strip()] if args.patch_cols else None

    feature_cols, patch_cols_used = resolve_feature_columns(
        df, target_col=target_col, id_col=id_col, drop_cols=drop_cols,
        feature_set=args.feature_set, patch_cols=patch_cols
    )

    # filter rows with missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    y = df[target_col].values.astype(float)
    X = df[feature_cols].values.astype(float)

    # Output prefix (used for ALL artifacts in this run)
    out_prefix = args.out_prefix or f"{target_col}_OOF_{args.feature_set}"

    # patch features for baseline (always patch-only baseline)
    patch_cols_exact = ["patch_cdr_ion", "patch_cdr_hyd", "patch_hyd", "patch_ion"]
    if all(c in df.columns for c in patch_cols_exact):
        patch_cols_for_baseline = patch_cols_exact
    else:
        patch_cols_for_baseline = patch_cols_used

    X_patch = df[patch_cols_for_baseline].values.astype(float) if patch_cols_for_baseline else None

    n = len(df)
    n_splits = min(args.n_splits, n)
    if n_splits < 2:
        raise ValueError("Need at least 2 splits for CV.")

    # repeated CV: we store one OOF prediction per repeat and then average (bagged OOF)
    oof_acet_repeats = []
    oof_bailly_repeats = []

    # store per-repeat metrics for mean±SD reporting
    repeat_rows = []

    for rep in range(args.cv_repeats):
        seed = args.seed + rep
        print(f"\n[OOF] Repeat {rep+1}/{args.cv_repeats} (seed={seed})")
        set_global_seed(seed)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_acet = np.full(n, np.nan, dtype=float)
        oof_bailly = np.full(n, np.nan, dtype=float)

        for fold, (tr_idx, vl_idx) in enumerate(kf.split(X, y), 1):
            print(f"  Fold {fold}/{n_splits}")

            # Fold split
            df_tr = df.iloc[tr_idx].copy()
            df_vl = df.iloc[vl_idx].copy()

            # Optional synthetic augmentation (fit ONLY on training fold)
            if args.use_sdv:
                if not _HAVE_SDV:
                    raise RuntimeError("SDV is not installed but --use_sdv was requested.")
                meta = SingleTableMetadata()
                meta.detect_from_dataframe(df_tr[[*feature_cols, target_col]])
                synth = GaussianCopulaSynthesizer(meta, default_distribution="norm")
                synth.fit(df_tr[[*feature_cols, target_col]])
                syn = synth.sample(num_rows=args.synth_rows)
                df_train_aug = pd.concat([df_tr[[*feature_cols, target_col]], syn], ignore_index=True)
            else:
                df_train_aug = df_tr[[*feature_cols, target_col]].copy()

            # Foldwise X scaling (NO leakage)
            sc_X = QuantileTransformer()
            X_tr_aug = sc_X.fit_transform(df_train_aug[feature_cols].values)
            X_vl = sc_X.transform(df_vl[feature_cols].values)

            y_tr_aug = df_train_aug[target_col].values.astype(float)
            y_vl_true = df_vl[target_col].values.astype(float)

            # Optional IBLR rebalancing on scaled fold-train only
            if args.use_iblr:
                if not _HAVE_IBLR:
                    raise RuntimeError("ImbalancedLearningRegression is not installed but --use_iblr was requested.")
                df_tr_scaled = pd.DataFrame(X_tr_aug, columns=feature_cols)
                df_tr_scaled[target_col] = y_tr_aug

                # ENN
                try:
                    df_tr_clean = iblr.enn(data=df_tr_scaled, y=target_col, rel_coef=args.enn_relcoef)
                except Exception as e:
                    print(f"    [Fold {fold}] ENN failed ({e}); using uncleaned fold data")
                    df_tr_clean = df_tr_scaled.copy()

                if df_tr_clean.isnull().values.any():
                    df_tr_clean = df_tr_clean.dropna()

                # SMOTE (fallback)
                df_tr_bal = None
                for rel in (args.smote_relcoef, args.smote_relcoef_fallback):
                    try:
                        df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=rel)
                        break
                    except Exception as e:
                        print(f"    [Fold {fold}] SMOTE failed (rel_coef={rel}) -> {e}")

                if df_tr_bal is None:
                    df_tr_bal = df_tr_clean.copy()

                # Force full feature set / order (prevents shape mismatch)
                for c in feature_cols:
                    if c not in df_tr_bal.columns:
                        df_tr_bal[c] = 0.0
                df_tr_bal = df_tr_bal[feature_cols + [target_col]]

                X_train = df_tr_bal[feature_cols].values
                y_train = df_tr_bal[target_col].values.astype(float)
            else:
                X_train = X_tr_aug
                y_train = y_tr_aug

            # Build ACeT model
            m = build_transformer_model(
                num_features=X_train.shape[1],
                task="regression",
                head_type=args.head_type if args.head_type != "all" else "mlp",
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg,
            )
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="log_cosh")

            # Foldwise y scaling
            sc_y = QuantileTransformer()
            y_tr_s = sc_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_vl_s = sc_y.transform(y_vl_true.reshape(-1, 1)).reshape(-1)

            cb = [
                EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr),
            ]

            m.fit(
                X_train, y_tr_s,
                validation_data=(X_vl, y_vl_s),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=cb,
                verbose=args.verbose,
                shuffle=True,
            )

            pred_vl_s = m.predict(X_vl, verbose=0).reshape(-1, 1)
            pred_vl = sc_y.inverse_transform(pred_vl_s).reshape(-1)
            oof_acet[vl_idx] = pred_vl

            # Bailly-style refit baseline on SAME folds (fair)
            if X_patch is not None and X_patch.shape[1] >= 2:
                Xp_tr = X_patch[tr_idx]
                yp_tr = y[tr_idx]
                Xp_vl = X_patch[vl_idx]
                b = fit_bailly_style_refit_baseline(
                    Xp_tr, yp_tr,
                    method=args.bailly_refit_method,
                    pls_components=args.bailly_pls_components
                )
                oof_bailly[vl_idx] = np.asarray(b.predict(Xp_vl)).reshape(-1)

        oof_acet_repeats.append(oof_acet)
        if X_patch is not None:
            oof_bailly_repeats.append(oof_bailly)

        # per-repeat metrics (OOF, single repeat)
        m_acet = regression_metrics(y, oof_acet)
        b_acet = binary_cutoff_metrics(y, oof_acet, cutoff=args.binary_cutoff)

        print(
            f"[OOF repeat {rep+1}] ACeT pearson_r2={m_acet['pearson_r2']:.3f}, "
            f"spearman={m_acet['spearman_rho']:.3f}, rmse={m_acet['rmse']:.3f} | "
            f"good_acc={b_acet['good_acc']:.3f}, poor_acc={b_acet['poor_acc']:.3f}"
        )

        row = {
            "repeat": int(rep + 1),
            "seed": int(seed),
            "acet_pearson_r2": float(m_acet["pearson_r2"]),
            "acet_spearman_rho": float(m_acet["spearman_rho"]),
            "acet_sklearn_R2": float(m_acet["sklearn_R2"]),
            "acet_rmse": float(m_acet["rmse"]),
            "acet_mae": float(m_acet["mae"]),
            "acet_good_acc": float(b_acet["good_acc"]),
            "acet_poor_acc": float(b_acet["poor_acc"]),
            "acet_accuracy": float(b_acet["accuracy"]),
            "acet_mcc": float(b_acet["mcc"]),
        }

        if X_patch is not None:
            m_b = regression_metrics(y, oof_bailly)
            b_b = binary_cutoff_metrics(y, oof_bailly, cutoff=args.binary_cutoff)
            print(
                f"[OOF repeat {rep+1}] Bailly-refit pearson_r2={m_b['pearson_r2']:.3f}, "
                f"spearman={m_b['spearman_rho']:.3f}, rmse={m_b['rmse']:.3f} | "
                f"good_acc={b_b['good_acc']:.3f}, poor_acc={b_b['poor_acc']:.3f}"
            )
            row.update({
                "bailly_refit_pearson_r2": float(m_b["pearson_r2"]),
                "bailly_refit_spearman_rho": float(m_b["spearman_rho"]),
                "bailly_refit_sklearn_R2": float(m_b["sklearn_R2"]),
                "bailly_refit_rmse": float(m_b["rmse"]),
                "bailly_refit_mae": float(m_b["mae"]),
                "bailly_refit_good_acc": float(b_b["good_acc"]),
                "bailly_refit_poor_acc": float(b_b["poor_acc"]),
                "bailly_refit_accuracy": float(b_b["accuracy"]),
                "bailly_refit_mcc": float(b_b["mcc"]),
            })

        repeat_rows.append(row)

    # Save per-repeat metrics to CSV (easy to paste into supplement)
    repeat_df = pd.DataFrame(repeat_rows)
    out_repeat_csv = f"{out_prefix}_repeat_metrics.csv"
    repeat_df.to_csv(out_repeat_csv, index=False)
    print(f"\nSaved per-repeat metrics to '{out_repeat_csv}'.")

    # Summarize mean±SD across repeats (NOT the same as bagged OOF-ensemble)
    def _mean_std(col: str):
        return float(repeat_df[col].mean()), float(repeat_df[col].std(ddof=1)) if len(repeat_df[col]) > 1 else 0.0

    repeat_summary = {"acet": {}, "bailly_refit": {}}

    for key in ["pearson_r2", "spearman_rho", "sklearn_R2", "rmse", "mae", "good_acc", "poor_acc", "accuracy", "mcc"]:
        m, s = _mean_std(f"acet_{key}")
        repeat_summary["acet"][key] = {"mean": m, "std": s}

        if f"bailly_refit_{key}" in repeat_df.columns:
            m2, s2 = _mean_std(f"bailly_refit_{key}")
            repeat_summary["bailly_refit"][key] = {"mean": m2, "std": s2}

    print("\n=== OOF Summary (mean +/- SD across repeats; single-repeat OOF) ===")
    a = repeat_summary["acet"]
    print(
        f"ACeT: pearson_r2={a['pearson_r2']['mean']:.4f} +/- {a['pearson_r2']['std']:.4f}, "
        f"spearman={a['spearman_rho']['mean']:.4f} +/- {a['spearman_rho']['std']:.4f}, "
        f"rmse={a['rmse']['mean']:.3f} +/- {a['rmse']['std']:.3f}, "
        f"mae={a['mae']['mean']:.3f} +/- {a['mae']['std']:.3f}"
    )
    print(
        f"ACeT (binary@{args.binary_cutoff}): good_acc={a['good_acc']['mean']:.3f} +/- {a['good_acc']['std']:.3f}, "
        f"poor_acc={a['poor_acc']['mean']:.3f} +/- {a['poor_acc']['std']:.3f}, "
        f"acc={a['accuracy']['mean']:.3f} +/- {a['accuracy']['std']:.3f}, "
        f"mcc={a['mcc']['mean']:.3f} +/- {a['mcc']['std']:.3f}"
    )

    if repeat_summary["bailly_refit"]:
        b = repeat_summary["bailly_refit"]
        if b:
            print(
                f"Bailly-refit: pearson_r2={b['pearson_r2']['mean']:.4f} +/- {b['pearson_r2']['std']:.4f}, "
                f"spearman={b['spearman_rho']['mean']:.4f} +/- {b['spearman_rho']['std']:.4f}, "
                f"rmse={b['rmse']['mean']:.3f} +/- {b['rmse']['std']:.3f}, "
                f"mae={b['mae']['mean']:.3f} +/- {b['mae']['std']:.3f}"
            )
            print(
                f"Bailly-refit (binary@{args.binary_cutoff}): good_acc={b['good_acc']['mean']:.3f} +/- {b['good_acc']['std']:.3f}, "
                f"poor_acc={b['poor_acc']['mean']:.3f} +/- {b['poor_acc']['std']:.3f}, "
                f"acc={b['accuracy']['mean']:.3f} +/- {b['accuracy']['std']:.3f}, "
                f"mcc={b['mcc']['mean']:.3f} +/- {b['mcc']['std']:.3f}"
            )

    # Average OOF across repeats (bagged OOF ensemble; one prediction per sample)
    oof_acet_mean = np.nanmean(np.vstack(oof_acet_repeats), axis=0)
    oof_bailly_mean = np.nanmean(np.vstack(oof_bailly_repeats), axis=0) if oof_bailly_repeats else None

    # Published equation (reference) across full dataset
    bailly_eq = bailly_published_equation_pred(df)

    # Final metrics (on averaged OOF predictions across repeats)
    acet_m = regression_metrics(y, oof_acet_mean)
    acet_b = binary_cutoff_metrics(y, oof_acet_mean, cutoff=args.binary_cutoff)

    baseline = {}
    if oof_bailly_mean is not None:
        baseline["bailly_refit_oof_mean"] = regression_metrics(y, oof_bailly_mean)
        baseline["bailly_refit_oof_mean_binary"] = binary_cutoff_metrics(y, oof_bailly_mean, cutoff=args.binary_cutoff)

    if bailly_eq is not None:
        baseline["bailly_equation_reference"] = regression_metrics(y, bailly_eq)
        baseline["bailly_equation_reference_binary"] = binary_cutoff_metrics(y, bailly_eq, cutoff=args.binary_cutoff)

    print("\n=== OOF Summary (bagged OOF: predictions averaged across repeats) ===")
    print(
        f"ACeT: pearson_r2={acet_m['pearson_r2']:.4f}, spearman={acet_m['spearman_rho']:.4f}, "
        f"sklearn_R2={acet_m['sklearn_R2']:.4f}, rmse={acet_m['rmse']:.3f}, mae={acet_m['mae']:.3f}"
    )
    print(
        f"ACeT (binary@{args.binary_cutoff}): good_acc={acet_b['good_acc']:.3f}, poor_acc={acet_b['poor_acc']:.3f}, "
        f"acc={acet_b['accuracy']:.3f}, mcc={acet_b['mcc']:.3f}"
    )

    if oof_bailly_mean is not None:
        bm = baseline["bailly_refit_oof_mean"]
        bb = baseline["bailly_refit_oof_mean_binary"]
        print(
            f"Bailly-refit: pearson_r2={bm['pearson_r2']:.4f}, spearman={bm['spearman_rho']:.4f}, "
            f"sklearn_R2={bm['sklearn_R2']:.4f}, rmse={bm['rmse']:.3f}, mae={bm['mae']:.3f}"
        )
        print(
            f"Bailly-refit (binary@{args.binary_cutoff}): good_acc={bb['good_acc']:.3f}, poor_acc={bb['poor_acc']:.3f}, "
            f"acc={bb['accuracy']:.3f}, mcc={bb['mcc']:.3f}"
        )

    # Save OOF predictions table
    out_oof_csv = f"{out_prefix}_oof_predictions.csv"
    out_metrics_json = f"{out_prefix}_oof_metrics.json"
    out_parity_png = f"{out_prefix}_oof_parity.png"

    out = pd.DataFrame({"True": y, "OOF_ACeT": oof_acet_mean})
    if id_col and id_col in df.columns:
        out.insert(0, id_col, df[id_col].astype(str).values)

    if oof_bailly_mean is not None:
        out["OOF_BaillyRefit"] = oof_bailly_mean
    if bailly_eq is not None:
        out["BaillyEq_reference"] = bailly_eq

    out["True_is_poor"] = (out["True"] > args.binary_cutoff).astype(int)
    out["ACeT_is_poor"] = (out["OOF_ACeT"] > args.binary_cutoff).astype(int)
    if oof_bailly_mean is not None:
        out["BaillyRefit_is_poor"] = (out["OOF_BaillyRefit"] > args.binary_cutoff).astype(int)
    if bailly_eq is not None:
        out["BaillyEq_is_poor"] = (out["BaillyEq_reference"] > args.binary_cutoff).astype(int)

    out.to_csv(out_oof_csv, index=False)
    print(f"Saved OOF predictions to '{out_oof_csv}'.")

    plot_parity(y, oof_acet_mean, title=f"OOF Parity: {target_col} (ACeT)", out_png=out_parity_png, log_scale=args.log_plot)
    print(f"Saved OOF parity plot '{out_parity_png}'.")

    payload = {
        "analysis_mode": "oof_cv",
        "data_file": args.data_file,
        "target_col": target_col,
        "id_col": id_col,
        "feature_set": args.feature_set,
        "feature_cols": feature_cols,
        "patch_cols_used": patch_cols_used,
        # Bagged OOF ensemble metrics
        "acet_oof_metrics": acet_m,
        "acet_oof_binary": acet_b,
        "baselines": baseline,
        # Repeat stats (mean±SD across repeats)
        "repeat_metrics_csv": out_repeat_csv,
        "repeat_metrics_summary_mean_std": repeat_summary,
        "repeat_rows": repeat_rows if args.cv_repeats <= 50 else None,
        "cv_repeats": int(args.cv_repeats),
        "n_splits": int(n_splits),
    }
    with open(out_metrics_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved OOF metrics to '{out_metrics_json}'.")


# --------------------------------------------------------------------------------------
# Classification placeholder 
# --------------------------------------------------------------------------------------
def run_classification(args):
    raise NotImplementedError(
        "Classification pipeline is not implemented in this edited script. "
        "Use existing clinical-outcome pipeline (panels.py) for classification tasks."
    )


# --------------------------------------------------------------------------------------
# Argparse & Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["regression", "classification"], required=True)

    # mode selection
    p.add_argument("--analysis_mode", choices=["train_test", "oof_cv"], default="train_test",
                   help="train_test: use --train_file/--test_file. oof_cv: use --data_file to create OOF predictions.")

    # files
    p.add_argument("--train_file", default=None, help="Train CSV (required for train_test mode).")
    p.add_argument("--test_file", default=None, help="Test CSV (required for train_test mode).")
    p.add_argument("--data_file", default=None, help="Single CSV for OOF CV mode (e.g., Bailly full dataset).")

    # columns
    p.add_argument("--target_col", default=None, help="Target column name (default: last column).")
    p.add_argument("--id_col", default=None, help="Optional ID column to carry through outputs (dropped from features).")
    p.add_argument("--drop_cols", default="", help="Comma-separated extra columns to drop from features.")
    p.add_argument("--guard_hic_leakage", action="store_true", default=True,
                   help="Drop any non-target columns containing 'hic' to prevent leakage (default: on).")
    p.add_argument("--no_guard_hic_leakage", action="store_false", dest="guard_hic_leakage")

    # feature subsets
    p.add_argument("--feature_set", choices=["all", "assays_only", "patch_only"], default="all",
                   help="all: assays+patch; assays_only: exclude patch_*; patch_only: use only patch_*.")
    p.add_argument("--patch_cols", default=None,
                   help="Comma-separated explicit patch columns (if not provided, auto-detect columns starting with 'patch_').")

    # ACeT head selection
    p.add_argument("--head_type", choices=["mlp", "rbf", "spline", "kan", "interaction", "all"], default="mlp")

    # training hyperparams
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--cv_repeats", type=int, default=1, help="Only for oof_cv mode. Number of repeated CV runs.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--ff_dim", type=int, default=32)
    p.add_argument("--num_transformer_blocks", type=int, default=1)
    p.add_argument("--mlp_units", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default="64")
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--l2_reg", type=float, default=1e-3)

    # callbacks
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--verbose", type=int, default=1)

    # optional SDV augmentation + IBLR rebalancing
    p.add_argument("--use_sdv", action="store_true", default=True,
                   help="Use SDV GaussianCopulaSynthesizer to augment training (default: on).")
    p.add_argument("--no_sdv", action="store_false", dest="use_sdv")
    p.add_argument("--synth_rows", type=int, default=60)

    p.add_argument("--use_iblr", action="store_true", default=True,
                   help="Use ENN+SMOTE from ImbalancedLearningRegression on each fold (default: on).")
    p.add_argument("--no_iblr", action="store_false", dest="use_iblr")
    p.add_argument("--enn_relcoef", type=float, default=0.5)
    p.add_argument("--smote_relcoef", type=float, default=0.5)
    p.add_argument("--smote_relcoef_fallback", type=float, default=0.25)

    # Bailly baselines
    p.add_argument("--compute_bailly_baselines", action="store_true", default=True,
                   help="Compute Bailly-style baselines (refit + published equation) where possible.")
    p.add_argument("--no_bailly_baselines", action="store_false", dest="compute_bailly_baselines")
    p.add_argument("--bailly_refit_method", choices=["linear", "pls"], default="linear")
    p.add_argument("--bailly_pls_components", type=int, default=2)

    # binary cutoff (Bailly uses 30 min for HIC good vs poor)
    p.add_argument("--binary_cutoff", type=float, default=30.0)

    # plots + outputs
    p.add_argument("--log_plot", action="store_true", default=False,
                   help="Use log-log parity plot only if all values positive.")
    p.add_argument("--out_prefix", default=None, help="Prefix for output files.")

    # permutation importance (train_test mode only)
    p.add_argument("--permutation_importance", action="store_true", default=False)
    p.add_argument("--perm_repeats", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.task == "regression":
        if args.analysis_mode == "train_test":
            if not args.train_file or not args.test_file:
                raise ValueError("train_test mode requires --train_file and --test_file.")
            run_regression_train_test(args)
        else:
            if not args.data_file:
                # allow using --train_file as the data file (convenience)
                if args.train_file:
                    args.data_file = args.train_file
                else:
                    raise ValueError("oof_cv mode requires --data_file (or provide --train_file as the dataset).")
            run_regression_oof_cv(args)
    else:
        run_classification(args)


if __name__ == "__main__":
    main()
