"""
 viscosity pooling ablation.

Smoke example:
python viscosity_pooling_ablation.py --smoke

Full mode is implemented for a later controlled run:
python viscosity_pooling_ablation.py --full
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import ImbalancedLearningRegression as iblr
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tfkan.layers import DenseKAN


POOLINGS = ("avg", "max", "flatten", "avgmax")
ENDPOINT = "viscosity"
HEAD_TYPE = "kan"
SYNTHETIC_ROWS = 26


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Data/curated").is_dir():
            return parent
    raise RuntimeError("Could not locate ACeT repository root")


def default_train_file() -> Path:
    return repo_root() / "MainPack/viscosity/code/source/shared/antibodies_train.csv"


def default_test_file() -> Path:
    return repo_root() / "MainPack/viscosity/code/source/shared/antibodies_test.csv"


def default_output_dir() -> Path:
    return repo_root() / "reruns" / "pooling"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class DenseKANRBF(layers.Layer):
    def __init__(self, units, grid_size=5, grid_range=(-1.0, 1.0), basis_function="rbf", mlp_units=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.basis_function = basis_function
        self.mlp_units = mlp_units or []

    def build(self, input_shape):
        in_f = int(input_shape[-1])
        low, high = self.grid_range
        centers_1d = tf.cast(tf.linspace(low, high, self.grid_size), dtype=self.dtype)
        centers = tf.tile(centers_1d[None, :], [in_f, 1])
        self.centers = self.add_weight(
            "centers",
            shape=(in_f, self.grid_size),
            initializer=tf.keras.initializers.Constant(centers),
            trainable=False,
            dtype=self.dtype,
        )
        self.basis_kernel = self.add_weight(
            "basis_kernel",
            shape=(in_f, self.grid_size, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.mlp_units:
            mlp_layers = [layers.Dense(u, activation="gelu") for u in self.mlp_units]
            mlp_layers.append(layers.Dense(self.units))
            self.mlp = models.Sequential(mlp_layers)
        self.bias = self.add_weight("bias", shape=(self.units,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        features = inputs.shape[-1]
        x_exp = tf.reshape(inputs, [batch, features, 1])
        centers = tf.reshape(self.centers, [1, features, self.grid_size])
        diff = x_exp - centers
        basis = tf.exp(-tf.square(diff)) if self.basis_function == "rbf" else tf.exp(-tf.abs(diff))
        weighted = tf.einsum("bfg,fgu->bfu", basis, self.basis_kernel)
        out = tf.reduce_sum(weighted, axis=1)
        if hasattr(self, "mlp"):
            out += self.mlp(inputs)
        return out + self.bias


def apply_pooling(x, pooling):
    if pooling == "avg":
        return layers.GlobalAveragePooling1D(name="pool_avg")(x)
    if pooling == "max":
        return layers.GlobalMaxPooling1D(name="pool_max")(x)
    if pooling == "flatten":
        return layers.Flatten(name="pool_flatten")(x)
    if pooling == "avgmax":
        avg = layers.GlobalAveragePooling1D(name="pool_avg")(x)
        max_pool = layers.GlobalMaxPooling1D(name="pool_max")(x)
        return layers.Concatenate(name="pool_avgmax")([avg, max_pool])
    raise ValueError(f"Unknown pooling: {pooling}")


def build_transformer_model(
    num_features,
    task,
    pooling,
    embed_dim=16,
    num_heads=2,
    ff_dim=32,
    num_transformer_blocks=1,
    mlp_units=None,
    dropout_rate=0.3,
    l2_reg=1e-3,
    num_classes=None,
    head_type=HEAD_TYPE,
):
    mlp_units = mlp_units or [64]
    inputs = layers.Input(shape=(num_features,))
    tokens = []
    for i in range(num_features):
        token = layers.Lambda(lambda x, i=i: x[:, i : i + 1])(inputs)
        token = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))(token)
        token = layers.Lambda(lambda x: tf.expand_dims(x, 1))(token)
        tokens.append(token)
    x = layers.Concatenate(axis=1)(tokens)
    positions = tf.range(0, num_features, 1)
    pos_emb = layers.Embedding(input_dim=num_features, output_dim=embed_dim)(positions)
    x = x + pos_emb
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
        x2 = layers.Add()([x, att])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn = layers.Dense(ff_dim, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x3)
        ffn = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))(ffn)
        x = layers.Add()([x2, ffn])
    x = apply_pooling(x, pooling)
    if head_type == "kan":
        out_dim = mlp_units[-1] if mlp_units else embed_dim
        x = DenseKAN(units=out_dim)(x)
    elif head_type == "spline":
        def spline_basis(z):
            knots = tf.constant([-1.0, 0.0, 1.0], dtype=z.dtype)
            diff = tf.nn.relu(tf.expand_dims(z, -1) - knots)
            return tf.pow(diff, 3)

        x = layers.Lambda(spline_basis)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    elif head_type == "mlp":
        for units in mlp_units:
            x = layers.Dense(units, activation="gelu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    elif head_type == "rbf":
        out_dim = mlp_units[-1] if mlp_units else embed_dim
        x = DenseKANRBF(units=out_dim, grid_size=8, grid_range=(-1, 1), basis_function="rbf", mlp_units=mlp_units)(x)
    else:
        raise ValueError(f"Unsupported head_type for pooling ablation: {head_type}")
    if task == "regression":
        outputs = layers.Dense(1, activation="linear")(x)
    elif task == "classification":
        outputs = layers.Dense(num_classes, activation="softmax")(x)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")
    return models.Model(inputs, outputs)


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def sample_synthetic(train, rows, seed):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution="norm")
    synthesizer.fit(train)
    try:
        return synthesizer.sample(num_rows=rows, random_state=seed)
    except TypeError:
        return synthesizer.sample(num_rows=rows)


def safe_spearman(y_true, y_pred):
    coef, _ = spearmanr(y_true, y_pred)
    return float(coef) if np.isfinite(coef) else np.nan


def metric_row(endpoint, mode, pooling, fold, split, y_true, y_pred, elapsed_seconds, extra=None):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    row = {
        "endpoint": endpoint,
        "mode": mode,
        "pooling": pooling,
        "fold": fold,
        "split": split,
        "n": len(y_true),
        "r2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "elapsed_seconds": elapsed_seconds,
    }
    if extra:
        row.update(extra)
    return row


def rebalance_fold(df_tr, target_col, fold):
    df_tr_clean = iblr.enn(data=df_tr, y=target_col, rel_coef=0.5)
    if df_tr_clean.isnull().values.any():
        df_tr_clean = df_tr_clean.dropna()
    df_tr_clean = df_tr_clean.loc[:, df_tr_clean.nunique() > 1]
    try:
        df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=0.5)
    except ValueError as err:
        print(f"[Fold {fold}] SMOTE failed ({err}); retrying with rel_coef=0.25")
    try:
        df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=0.25)
    except ValueError as err2:
        print(f"[Fold {fold}] SMOTE retry failed ({err2}); skipping SMOTE")
    return df_tr_bal


def run(args):
    set_seeds(args.seed)
    train = pd.read_csv(args.train_file)
    test = pd.read_csv(args.test_file)
    target_col = train.columns[-1]
    synthetic_data = sample_synthetic(train, SYNTHETIC_ROWS, args.seed)
    train_mix = pd.concat([train, train, synthetic_data], ignore_index=True)
    sc_x = QuantileTransformer()
    X_all = sc_x.fit_transform(train_mix.iloc[:, :-1].values)
    X_test = sc_x.transform(test.iloc[:, :-1].values)
    y_all = train_mix.iloc[:, -1].values
    y_test = test.iloc[:, -1].values
    feature_cols = train_mix.columns[:-1]
    fold_count = 1 if args.smoke else args.n_splits
    max_epochs = 5 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience
    mode = "smoke" if args.smoke else "full"
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(X_all, y_all))[:fold_count]
    poolings = list(args.pooling) if args.pooling else list(POOLINGS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows, history_rows, prediction_rows, fold_rows = [], [], [], []
    failures = []
    run_start = time.time()

    print(f"Endpoint={ENDPOINT}; mode={mode}; head={HEAD_TYPE}; poolings={','.join(poolings)}")
    print(f"Train={Path(args.train_file).resolve()}")
    print(f"Test={Path(args.test_file).resolve()}")
    print(f"Original train shape={train.shape}; synthetic shape={synthetic_data.shape}; train_mix shape={train_mix.shape}")

    for pooling in poolings:
        pooling_start = time.time()
        ens_models = []
        pooling_failed = False
        print(f"--- Pooling = {pooling} ---")
        set_seeds(args.seed)
        for fold, (tr, vl) in enumerate(splits, 1):
            try:
                print(f"Fold {fold} / {fold_count}")
                X_tr, y_tr = X_all[tr], y_all[tr]
                df_tr = pd.DataFrame(X_tr, columns=feature_cols)
                df_tr[target_col] = y_tr
                df_tr_bal = rebalance_fold(df_tr, target_col, fold)
                X_tr_bal = df_tr_bal.iloc[:, :-1].values
                y_tr_bal = df_tr_bal[target_col].values

                model = build_transformer_model(
                    num_features=X_all.shape[1],
                    task="regression",
                    pooling=pooling,
                    head_type=HEAD_TYPE,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    ff_dim=args.ff_dim,
                    num_transformer_blocks=args.num_transformer_blocks,
                    mlp_units=args.mlp_units,
                    dropout_rate=args.dropout_rate,
                    l2_reg=args.l2_reg,
                )
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="log_cosh")
                sc_y = QuantileTransformer()
                y_tr_s = sc_y.fit_transform(y_tr_bal.reshape(-1, 1)).flatten()
                y_vl_s = sc_y.transform(y_all[vl].reshape(-1, 1)).flatten()
                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=args.lr_factor,
                        patience=args.lr_patience,
                        min_lr=args.min_lr,
                    ),
                ]
                history = model.fit(
                    X_tr_bal,
                    y_tr_s,
                    validation_data=(X_all[vl], y_vl_s),
                    epochs=max_epochs,
                    batch_size=args.batch_size,
                    callbacks=callbacks,
                    verbose=args.verbose,
                )
                ens_models.append((model, sc_y))
                for epoch_idx, loss in enumerate(history.history.get("loss", []), 1):
                    history_rows.append(
                        {
                            "endpoint": ENDPOINT,
                            "mode": mode,
                            "pooling": pooling,
                            "fold": fold,
                            "epoch": epoch_idx,
                            "loss": loss,
                            "val_loss": history.history.get("val_loss", [np.nan] * len(history.history.get("loss", [])))[epoch_idx - 1],
                        }
                    )
                pred_vl_s = model.predict(X_all[vl], verbose=0).flatten()
                pred_vl = sc_y.inverse_transform(pred_vl_s.reshape(-1, 1)).flatten()
                elapsed = time.time() - pooling_start
                metrics_rows.append(metric_row(ENDPOINT, mode, pooling, fold, "cv_validation", y_all[vl], pred_vl, elapsed))
                for idx, true, pred in zip(vl, y_all[vl], pred_vl):
                    prediction_rows.append(
                        {
                            "endpoint": ENDPOINT,
                            "mode": mode,
                            "pooling": pooling,
                            "fold": fold,
                            "split": "cv_validation",
                            "row_index": int(idx),
                            "true": true,
                            "predicted": pred,
                        }
                    )
                fold_rows.append(
                    {
                        "endpoint": ENDPOINT,
                        "mode": mode,
                        "pooling": pooling,
                        "fold": fold,
                        "train_indices": json.dumps([int(i) for i in tr]),
                        "validation_indices": json.dumps([int(i) for i in vl]),
                    }
                )
            except Exception as exc:
                pooling_failed = True
                failures.append({"pooling": pooling, "fold": fold, "error": repr(exc)})
                print(f"[{pooling}] Fold {fold} failed: {exc}")
                break

        if ens_models and not pooling_failed:
            preds_s = np.mean([mdl.predict(X_test, verbose=0).flatten() for mdl, _ in ens_models], axis=0)
            preds = ens_models[-1][1].inverse_transform(preds_s.reshape(-1, 1)).flatten()
            metrics_rows.append(
                metric_row(
                    ENDPOINT,
                    mode,
                    pooling,
                    "ensemble",
                    "heldout_test",
                    y_test,
                    preds,
                    time.time() - pooling_start,
                    {"ensemble_folds": len(ens_models)},
                )
            )
            for idx, true, pred in zip(range(len(y_test)), y_test, preds):
                prediction_rows.append(
                    {
                        "endpoint": ENDPOINT,
                        "mode": mode,
                        "pooling": pooling,
                        "fold": "ensemble",
                        "split": "heldout_test",
                        "row_index": int(idx),
                        "true": true,
                        "predicted": pred,
                    }
                )
            print(f"[{pooling}] completed; held-out test ensemble generated from {len(ens_models)} fold(s)")

    prefix = f"{ENDPOINT}_{mode}"
    metrics = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)
    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    failures_df = pd.DataFrame(failures)
    metrics.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    history.to_csv(output_dir / f"{prefix}_training_history.csv", index=False)
    predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    folds.to_csv(output_dir / f"{prefix}_fold_indices.csv", index=False)
    failures_df.to_csv(output_dir / f"{prefix}_failures.csv", index=False)
    summary_path = output_dir / f"{prefix}_summary.md"
    completed = sorted(metrics.loc[metrics["split"].eq("heldout_test"), "pooling"].unique().tolist()) if not metrics.empty else []
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {ENDPOINT.title()} {mode.title()} Pooling Ablation Summary\n\n")
        handle.write(f"- Head: `{HEAD_TYPE}`\n")
        handle.write(f"- Synthetic rows: {SYNTHETIC_ROWS}\n")
        handle.write(f"- Fold count run: {fold_count}\n")
        handle.write(f"- Max epochs: {max_epochs}\n")
        handle.write(f"- Early-stopping patience: {patience}\n")
        handle.write(f"- Completed pooling variants: {', '.join(completed) if completed else 'none'}\n")
        handle.write(f"- Failed pooling variants: {', '.join(sorted({f['pooling'] for f in failures})) if failures else 'none'}\n")
        handle.write(f"- Runtime seconds: {time.time() - run_start:.1f}\n\n")
        handle.write("Smoke metrics are engineering checks only and are not manuscript evidence.\n\n")
        if not metrics.empty:
            handle.write("## Metrics\n\n")
            handle.write(metrics.to_markdown(index=False))
            handle.write("\n")
    print(f"Saved {summary_path}")
    return {"completed": completed, "failures": failures, "runtime_seconds": time.time() - run_start}


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--pooling", nargs="+", choices=POOLINGS, default=None)
    parser.add_argument("--train_file", default=str(default_train_file()))
    parser.add_argument("--test_file", default=str(default_test_file()))
    parser.add_argument("--output_dir", default=str(default_output_dir()))
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "RUN_LOG.txt"
    with log_path.open("a", encoding="utf-8") as log_handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_handle)
        sys.stderr = Tee(old_stderr, log_handle)
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {ENDPOINT} pooling ablation")
            result = run(args)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {ENDPOINT}: {result}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()
