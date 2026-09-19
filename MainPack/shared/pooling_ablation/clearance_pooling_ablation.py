"""
 mouse-clearance pooling ablation.

Smoke example:
python clearance_pooling_ablation.py --smoke

Full mode is implemented for a later controlled run:
python clearance_pooling_ablation.py --full
"""
import argparse
import sys
import time
from pathlib import Path

import ImbalancedLearningRegression as iblr
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
import pandas as pd

import viscosity_pooling_ablation as base


ENDPOINT = "clearance"
HEAD_TYPE = "spline"
SYNTHETIC_ROWS = 84


def default_train_file() -> Path:
    return base.repo_root() / "MainPack/mouse_exposure/code/source/shared/clearance_train.csv"


def default_test_file() -> Path:
    return base.repo_root() / "MainPack/mouse_exposure/code/source/shared/clearance_test.csv"


def validation_output_dir() -> Path:
    return base.repo_root() / "reruns" / "pooling_validation"


def sample_synthetic(train, rows, seed):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution="norm")
    synthesizer.fit(train)
    return synthesizer.sample(num_rows=rows)


def _diagnose_frame(label, df, target_col):
    missing_by_col = df.isnull().sum().to_dict()
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    return {
        f"{label}_rows": len(df),
        f"{label}_cols": len(df.columns),
        f"{label}_missing_total": int(sum(missing_by_col.values())),
        f"{label}_missing_by_col": missing_by_col,
        f"{label}_constant_cols": constant_cols,
        f"{label}_has_target": target_col in df.columns,
    }


def _clean_like_original(df):
    if df.isnull().values.any():
        df = df.dropna()
    return df.loc[:, df.nunique() > 1]


def _is_valid_rebalance_frame(df, target_col):
    return target_col in df.columns and not df.empty and not df.isnull().values.any()


def rebalance_fold_with_diagnostics(df_tr, target_col, fold):
    diagnostics = {
        "fold": fold,
        **_diagnose_frame("input", df_tr, target_col),
    }
    last_valid = df_tr
    selected_source = "input_fallback"

    try:
        df_tr_clean = iblr.smote(data=df_tr, y=target_col, rel_coef=0.5)
        diagnostics.update(_diagnose_frame("smote_0_5_raw", df_tr_clean, target_col))
        df_tr_clean = _clean_like_original(df_tr_clean)
        diagnostics.update(_diagnose_frame("smote_0_5_clean", df_tr_clean, target_col))
        if _is_valid_rebalance_frame(df_tr_clean, target_col):
            last_valid = df_tr_clean
            selected_source = "smote_0_5_clean"
        else:
            diagnostics["smote_0_5_invalid"] = True
    except ValueError as err:
        diagnostics["smote_0_5_error"] = repr(err)
        df_tr_clean = _clean_like_original(df_tr)
        diagnostics.update(_diagnose_frame("smote_0_5_fallback_clean", df_tr_clean, target_col))
        if _is_valid_rebalance_frame(df_tr_clean, target_col):
            last_valid = df_tr_clean
            selected_source = "input_clean_after_smote_failure"

    df_tr_bal = last_valid
    try:
        enn_0_5 = iblr.enn(data=df_tr_clean, y=target_col, rel_coef=0.5)
        diagnostics.update(_diagnose_frame("enn_0_5_raw", enn_0_5, target_col))
        if _is_valid_rebalance_frame(enn_0_5, target_col):
            df_tr_bal = enn_0_5
            last_valid = enn_0_5
            selected_source = "enn_0_5"
        else:
            diagnostics["enn_0_5_invalid"] = True
    except ValueError as err:
        diagnostics["enn_0_5_error"] = repr(err)
        print(f"[Fold {fold}] ENN failed ({err}); retrying with rel_coef=0.25")
    try:
        enn_0_25 = iblr.enn(data=df_tr_clean, y=target_col, rel_coef=0.25)
        diagnostics.update(_diagnose_frame("enn_0_25_raw", enn_0_25, target_col))
        if _is_valid_rebalance_frame(enn_0_25, target_col):
            df_tr_bal = enn_0_25
            last_valid = enn_0_25
            selected_source = "enn_0_25"
        else:
            diagnostics["enn_0_25_invalid"] = True
            df_tr_bal = last_valid
            print(f"[Fold {fold}] ENN retry returned invalid data; using {selected_source}")
    except ValueError as err2:
        diagnostics["enn_0_25_error"] = repr(err2)
        print(f"[Fold {fold}] ENN retry failed ({err2}); skipping ENN")
        df_tr_bal = last_valid
    diagnostics["selected_source"] = selected_source
    diagnostics.update(_diagnose_frame("selected", df_tr_bal, target_col))
    return df_tr_bal, diagnostics


def rebalance_fold(df_tr, target_col, fold):
    df_tr_bal, diagnostics = rebalance_fold_with_diagnostics(df_tr, target_col, fold)
    if diagnostics.get("smote_0_5_error"):
        print(f"[Fold {fold}] SMOTE failed ({diagnostics['smote_0_5_error']}); using {diagnostics['selected_source']}")
    return df_tr_bal


def run_debug_rebalance_only(args):
    base.set_seeds(args.seed)
    train = pd.read_csv(args.train_file)
    test = pd.read_csv(args.test_file)
    target_col = train.columns[-1]
    synthetic_data = sample_synthetic(train, SYNTHETIC_ROWS, args.seed)
    train_mix = pd.concat([train, train, synthetic_data], ignore_index=True)
    sc_x = QuantileTransformer()
    x_all = sc_x.fit_transform(train_mix.iloc[:, :-1].values)
    y_all = train_mix.iloc[:, -1].values
    feature_cols = train_mix.columns[:-1]
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []

    for fold, (tr, vl) in enumerate(kf.split(x_all, y_all), 1):
        df_tr = pd.DataFrame(x_all[tr], columns=feature_cols)
        df_tr[target_col] = y_all[tr]
        _, diagnostics = rebalance_fold_with_diagnostics(df_tr, target_col, fold)
        diagnostics["train_rows"] = len(tr)
        diagnostics["validation_rows"] = len(vl)
        diagnostics["top_level_synthetic_missing_total"] = int(synthetic_data.isnull().sum().sum())
        diagnostics["top_level_train_mix_missing_total"] = int(train_mix.isnull().sum().sum())
        diagnostics["test_rows"] = len(test)
        rows.append(diagnostics)

    output_dir = validation_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "clearance_rebalance_debug.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--debug_rebalance_only", action="store_true")
    parser.add_argument("--pooling", nargs="+", choices=base.POOLINGS, default=None)
    parser.add_argument("--train_file", default=str(default_train_file()))
    parser.add_argument("--test_file", default=str(default_test_file()))
    parser.add_argument("--output_dir", default=str(base.default_output_dir()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=32)
    parser.add_argument("--num_transformer_blocks", type=int, default=1)
    parser.add_argument("--mlp_units", type=lambda s: [int(x) for x in s.split(",")], default="64")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--l2_reg", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    base.ENDPOINT = ENDPOINT
    base.HEAD_TYPE = HEAD_TYPE
    base.SYNTHETIC_ROWS = SYNTHETIC_ROWS
    base.sample_synthetic = sample_synthetic
    base.rebalance_fold = rebalance_fold
    args = parse_args()
    output_dir = validation_output_dir() if args.debug_rebalance_only else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "RUN_LOG.txt"
    with log_path.open("a", encoding="utf-8") as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = base.Tee(old_stdout, log_handle)
        sys.stderr = base.Tee(old_stderr, log_handle)
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {ENDPOINT} pooling ablation")
            result = run_debug_rebalance_only(args) if args.debug_rebalance_only else base.run(args)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {ENDPOINT}: {result}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
