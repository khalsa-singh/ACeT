"""
python panels.py --task classification --train_file InternalCohort_112mAbs_train.csv --test_file InternalCohort_112mAbs_test.csv --external_file ExternalCohort_14mAbs.csv --status_col Updated.Status --head_type all 

"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, PowerTransformer, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, recall_score,
)
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
import ImbalancedLearningRegression as iblr
from tfkan.layers import DenseKAN
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling  import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.tree import DecisionTreeClassifier, export_text
import shap 
from typing import Tuple 
import scipy.io as sio
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import seaborn as sns           
from typing import Dict
from clinical_model import DenseKANRBF, build_transformer_model

# ──────────────────────────────────────────────────────────────────────────────
# Jain‑et‑al. 2017 warning‑flag thresholds            
# ──────────────────────────────────────────────────────────────────────────────
JAIN_THRESH: Dict[str, Dict[str, float]] = {
    "PSR":        {">": 0.27},
    "ACSINS":     {">": 11.8},
    "CSI":        {">": 0.01},
    "CIC":        {">": 10.1},
    "HIC":        {">": 11.7},
    "SMAC":       {">": 12.8},
    "SGAC_SINS":  {"<": 370.0},
    "BVP":        {">": 4.3},
    "ELISA":      {">": 1.9},
    "AS":         {">": 0.08},
}
FLAG_GROUPS = {
    "grp1": ["PSR","ACSINS","CSI","CIC"],
    "grp2": ["HIC","SMAC","SGAC_SINS"],
    "grp3": ["BVP","ELISA"],
    "grp4": ["AS"],
}
def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for assay, rule in JAIN_THRESH.items():
        if assay not in df.columns: continue
        op, thr = list(rule.items())[0]
        flags[f"flag_{assay}"] = ((df[assay] > thr) if op==">"
                                  else (df[assay] < thr)).astype(int)
    for g, assays in FLAG_GROUPS.items():
        cols = [f"flag_{a}" for a in assays if f"flag_{a}" in flags.columns]
        flags[f"{g}_count"] = flags[cols].sum(axis=1)
    flags["flag_total"] = flags[[c for c in flags.columns if c.startswith("flag_")]].sum(axis=1)
    return flags

# ──────────────────────────────────────────────────────────────────────────────
# Threshold‑helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def youden_univariate_thresholds(X: pd.DataFrame,
                                 y: np.ndarray) -> pd.DataFrame:
    """Return optimal Youden‑J cut‑points per feature."""
    out = []
    for col in X.columns:
        x = X[col].values
        thresh = np.quantile(x, np.linspace(.01, .99, 99))
        tpr = lambda t: recall_score(y, x > t)          # positives = class 1
        tnr = lambda t: recall_score(1 - y, x <= t)     # negatives = class 0
        J = np.array([tpr(t) + tnr(t) - 1 for t in thresh])
        idx = int(np.argmax(J))
        out.append((col, thresh[idx], J[idx]))
    return (pd.DataFrame(out, columns=["feature", "threshold", "youden_J"])
              .sort_values("youden_J", ascending=False)
              .reset_index(drop=True))


def predict_and_cm(models, X, y_true):
    """Utility: ensemble probs → preds → balanced‑accuracy & confusion‑matrix."""
    probs = np.mean([m.predict(X) for m in models], axis=0)
    preds = np.argmax(probs, axis=1)
    bacc  = balanced_accuracy_score(y_true, preds)
    cm    = confusion_matrix(y_true, preds)
    return bacc, cm, preds, probs

# ------------------------------------------------------------------------------
# ------------------ Updated Classification Pipeline ------------------
def run_classification(args):
    print("=== Running Classification with Data‑driven Thresholds ===")
    train_df = pd.read_csv(args.train_file)
    test_df  = pd.read_csv(args.test_file)
    feature_cols = train_df.columns[:-1].tolist()
    target_col = train_df.columns[-1]

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Encode labels
    encoder = LabelEncoder()
    encoded_y_train = encoder.fit_transform(y_train_full)
    #encoded_y_train2 = encoder.fit_transform(train_df[target_col].values)
    encoded_y_test = encoder.transform(y_test)
    num_classes = len(encoder.classes_)

    # Scale features
    scaler = PowerTransformer()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    num_features = X_train_scaled.shape[1]

    heads = [args.head_type] if args.head_type != 'all' else [
        'mlp','rbf','spline','kan','interaction'
    ]

    best_models = None
    best_head = None
    best_cv_bacc  = -1.0
    fold_assignment_rows = []
    fold_metric_rows = []
    head_selection_rows = []
    history_rows = []
    oof_probs_by_head = {}
    model_paths = []

    for head in heads:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
        print(f"\n--- Head = {head.upper()} ---")
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        ensemble_models = []
        fold_baccs = []
        oof_probs = np.full((len(encoded_y_train), num_classes), np.nan)

        for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, encoded_y_train), start=1):
            print(f"\n--- Fold {fold_no}/{args.n_splits} ---")
            for idx in train_idx:
                fold_assignment_rows.append({'head': head, 'fold': fold_no, 'row_index': int(idx), 'role': 'fold_train', 'true_label': str(y_train_full[idx])})
            for idx in val_idx:
                fold_assignment_rows.append({'head': head, 'fold': fold_no, 'row_index': int(idx), 'role': 'fold_validation', 'true_label': str(y_train_full[idx])})
            X_tr = X_train_scaled[train_idx]
            y_tr_lbl = encoded_y_train[train_idx]
            X_val = X_train_scaled[val_idx]
            y_val_lbl = encoded_y_train[val_idx]
  
            
            # 1) ENN cleaning
            enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
            #enn = SMOTEENN(random_state=args.seed)
            #enn = TomekLinks()
            X_tr_clean, y_tr_clean = enn.fit_resample(X_tr, y_tr_lbl)
            class_counts_clean = np.bincount(y_tr_clean)
            print(f"After ENN: class_counts={class_counts_clean}")

            # If ENN yields too few, skip cleaning for SMOTE
            if class_counts_clean.min() < 2:
                print("ENN removed too many samples; using original training data for SMOTE.")
                X_for_smt, y_for_smt = X_tr, y_tr_lbl
            else:
                X_for_smt, y_for_smt = X_tr_clean, y_tr_clean

            # 2) SMOTE oversampling
            class_counts = np.bincount(y_for_smt)
            min_count = class_counts.min()
            k_smote = max(1, min(min_count - 1, 5))
            print(f"SMOTE k_neighbors={k_smote}, class_counts_before={class_counts}")
            smote = SMOTE(k_neighbors=k_smote, random_state=args.seed)
            X_tr_bal, y_tr_bal_lbl = smote.fit_resample(X_for_smt, y_for_smt)

            # One-hot encoding
            y_tr_bal = to_categorical(y_tr_bal_lbl, num_classes=num_classes)
            y_val_cat = to_categorical(y_val_lbl, num_classes=num_classes)

            # Build & compile model
            model = build_transformer_model(
                num_features=num_features,
                task='classification',
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg,
                num_classes=num_classes,
                head_type=head
            )
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy')

            # Callbacks & training
            early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                             patience=args.lr_patience, verbose=1, min_lr=args.min_lr)
            history = model.fit(
                X_tr_bal, y_tr_bal,
                validation_data=(X_val, y_val_cat),
                epochs=args.epochs, batch_size=args.batch_size,
                shuffle=True, callbacks=[early_stop, lr_reduction], verbose=1
            )

            # Validation Balanced Accuracy
            val_probs = model.predict(X_val)
            oof_probs[val_idx] = val_probs
            val_preds = np.argmax(val_probs, axis=1)
            bacc = balanced_accuracy_score(y_val_lbl, val_preds)
            print(f"Fold {fold_no} Balanced Accuracy: {bacc:.4f}")
            fold_baccs.append(bacc)
            ensemble_models.append(model)
            fold_metric_rows.append({'head': head, 'fold': fold_no, 'n_train_original': len(train_idx), 'n_validation': len(val_idx), 'n_after_enn': len(y_for_smt), 'n_after_smote': len(y_tr_bal_lbl), 'balanced_accuracy': bacc})
            for epoch_idx, values in enumerate(zip(history.history.get('loss', []), history.history.get('val_loss', [])), start=1):
                history_rows.append({'head': head, 'fold': fold_no, 'epoch': epoch_idx, 'loss': values[0], 'val_loss': values[1]})
            model_path = os.path.abspath(f'model_{head}_fold{fold_no}.weights.h5')
            model.save_weights(model_path)
            model_paths.append(model_path)

        # CV summary
        mean_bacc = np.mean(fold_baccs)
        oof_probs_by_head[head] = oof_probs
        head_selection_rows.append({'head': head, 'mean_cv_balanced_accuracy': mean_bacc, 'selected': False})
        print(f"\n--- {head.upper()} Mean CV Balanced Accuracy = {mean_bacc:.4f} ---")

        # Ensemble training evaluation
        def ensemble_predict(models, X):
            return np.mean([m.predict(X) for m in models], axis=0)
        train_probs = ensemble_predict(ensemble_models, X_train_scaled)
        train_preds = np.argmax(train_probs, axis=1)
        train_bacc = balanced_accuracy_score(encoded_y_train, train_preds)
        print(f"Ensemble Training Balanced Accuracy ({head.upper()}): {train_bacc:.4f}")
        cm_train = confusion_matrix(encoded_y_train, train_preds)
        print("Confusion Matrix (Training):")
        print(cm_train)
        ConfusionMatrixDisplay(cm_train, display_labels=encoder.classes_).plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"{head.upper()} Confusion Matrix (Training)")
        plt.savefig(f"cm_train_{head}.png", bbox_inches='tight')
        plt.close()


        # Test evaluation
        test_probs = ensemble_predict(ensemble_models, X_test_scaled)
        test_preds = np.argmax(test_probs, axis=1)
        test_bacc = balanced_accuracy_score(encoded_y_test, test_preds)
        print(f"Test Set Ensemble Balanced Accuracy ({head.upper()}): {test_bacc:.4f}")
        cm_test = confusion_matrix(encoded_y_test, test_preds)
        print("Confusion Matrix (Test):")
        print(cm_test)
        ConfusionMatrixDisplay(cm_test, display_labels=encoder.classes_).plot(cmap=plt.cm.Greens, colorbar=False)
        plt.title(f"{head.upper()} Confusion Matrix (Test)")
        plt.savefig(f"cm_test_{head}.png", bbox_inches='tight')
        plt.close()


        # Track best
        if mean_bacc > best_cv_bacc:
            best_cv_bacc = mean_bacc
            best_head    = head
            best_models  = list(ensemble_models)

    print(f"\n*** Best head by CV: {best_head.upper()} (CV BA = {best_cv_bacc:.4f}) ***")
    for row in head_selection_rows:
        row['selected'] = (row['head'] == best_head)

    # ─────────────────  EXTERNAL OUT‑OF‑TIME VALIDATION  ──────────────────
    if args.external_file:
        ext_df = pd.read_csv(args.external_file)

        # keep only mAbs whose Updated.Status is Approved / Terminated
        resolved = ext_df[args.status_col].isin(['Approved','Terminated'])
        ext_df   = ext_df.loc[resolved].copy()
        if ext_df.empty:
            print(">>> No resolved mAbs in external file ‑‑ skipping external eval")
        else:
            print(f">>> External cohort: {len(ext_df)} resolved mAbs")

            # feature alignment (same order as training)
            X_ext  = scaler.transform(ext_df[feature_cols].values)
            y_ext  = encoder.transform(ext_df[args.status_col].values)

            ext_bacc, cm_ext, ext_preds, ext_probs = predict_and_cm(
                best_models, X_ext, y_ext
            )
            print(f"External balanced accuracy = {ext_bacc:.4f}")
            print("External confusion matrix:\n", cm_ext)

            # save artefacts
            np.savetxt("external_probs.csv", ext_probs, delimiter=",")
            np.savetxt("external_preds.csv", ext_preds, delimiter=",")
            np.savetxt("external_labels.csv", y_ext,   delimiter=",")
            np.savetxt("cm_external.csv",     cm_ext,  delimiter=",", fmt='%d')



    # --------  NEW THRESHOLD DERIVATION  ---------------------------------- #
    print("\n>>> Deriving assay thresholds (Youden)")

    #  Univariate Youden‑J
    uni_df = youden_univariate_thresholds(
        X=train_df[feature_cols],
        y=encoder.transform(train_df[target_col].values)
    )
    uni_df.to_csv("Table_S1_thresholds.csv", index=False)
    print("  • Youden cut‑points → Table_S1_thresholds.csv")

    # ── SAVE confusion matrices, per-sample predictions & labels ──────────────────
    cm_train_arr = cm_train           # numpy arrays from last head = best_head
    cm_test_arr  = cm_test
    train_labels = encoded_y_train    # 1-D int array
    test_labels  = encoded_y_test
    train_preds  = train_preds        # from best ensemble on true train set
    test_preds   = test_preds
    class_names  = np.array(encoder.classes_, dtype=object)

    # Optional: keep antibody identifiers if column exists
    if 'mAb_name' in train_df.columns:
        mAb_train = train_df['mAb_name'].values.astype(object)
        mAb_test  = test_df ['mAb_name'].values.astype(object)
    else:
        mAb_train = mAb_test = np.array([], dtype=object)

    # Append everything to the existing .mat bundle
    sio.savemat('figvars.mat', {
        'cm_train'       : cm_train_arr,
        'cm_test'        : cm_test_arr,
        'train_labels'   : train_labels,
        'test_labels'    : test_labels,
        'train_preds'    : train_preds,
        'test_preds'     : test_preds,
        'class_names'    : class_names,
        'mAb_train'      : mAb_train,
        'cm_external'    : cm_ext,
        'external_labels': y_ext,
        'external_preds' : ext_preds,
        'mAb_test'       : mAb_test,

    }, appendmat=True)

    print("  • Added confusion matrices to figvars.mat")

    # Provenance-only serialization; no modeling calculations are changed.
    pd.DataFrame(fold_assignment_rows).to_csv('FOLD_ASSIGNMENTS.csv', index=False)
    pd.DataFrame(fold_metric_rows).to_csv('FOLD_METRICS.csv', index=False)
    pd.DataFrame(head_selection_rows).to_csv('HEAD_SELECTION_RESULTS.csv', index=False)
    pd.DataFrame(history_rows).to_csv('TRAINING_HISTORY.csv', index=False)
    selected_oof = oof_probs_by_head[best_head]
    def prediction_table(labels_text, labels_encoded, probs, preds, cohort, provenance):
        return pd.DataFrame({
            'row_index': np.arange(len(labels_encoded)),
            'true_label': labels_text,
            'positive_class': encoder.classes_[0],
            'positive_class_probability': probs[:, 0],
            'operating_threshold': 0.5,
            'predicted_label': encoder.inverse_transform(preds),
            'cohort': cohort,
            'provenance': provenance,
        })
    train_table = prediction_table(y_train_full, encoded_y_train, train_probs, train_preds, 'internal_train', 'mean_probability_across_5_fold_models_in_sample')
    train_table['oof_positive_class_probability'] = selected_oof[:, 0]
    train_table.to_csv('TRAIN_PREDICTIONS.csv', index=False)
    prediction_table(y_test, encoded_y_test, test_probs, test_preds, 'internal_test', 'mean_probability_across_5_fold_models').to_csv('INTERNAL_TEST_PREDICTIONS.csv', index=False)
    if args.external_file and 'ext_probs' in locals():
        prediction_table(ext_df[args.status_col].values, y_ext, ext_probs, ext_preds, 'external_resolved', 'mean_probability_across_selected_head_5_fold_models').to_csv('EXTERNAL_PREDICTIONS.csv', index=False)
    pd.DataFrame([{'selected_head': best_head, 'positive_class': encoder.classes_[0], 'operating_threshold': 0.5, 'procedure': 'two-class softmax argmax; equivalent to positive-class probability >= 0.5 absent exact ties', 'derivation_cohort': 'fixed decision rule; not optimized on any cohort'}]).to_csv('MODEL_OPERATING_THRESHOLD.csv', index=False)
    pd.DataFrame({'model_path': model_paths}).to_csv('MODEL_PATHS.csv', index=False)

# 5) Argparse & Main
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['regression','classification'], required=True)
    p.add_argument('--train_file', required=True, help='Path to training CSV file')
    p.add_argument('--test_file',  required=True, help='Path to test CSV file')
    p.add_argument('--head_type',
                   choices=['mlp','rbf','spline','kan','interaction','all'],
                   default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--ff_dim', type=int, default=32)
    p.add_argument('--num_transformer_blocks', type=int, default=1)
    p.add_argument('--mlp_units',
                   type=lambda s: [int(x) for x in s.split(',')], default='64')
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--l2_reg', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr_factor', type=float, default=0.5)
    p.add_argument('--lr_patience', type=int, default=10)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--external_file',  default=None,
                   help='CSV of additional mAbs for out‑of‑time validation')
    p.add_argument('--status_col',     default='Updated.Status',
                   help='Column in --external_file containing final outcome')
    p.add_argument('--output_dir', default='.', help='Directory for serialized outputs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    if args.task == 'regression':
        run_regression(args)
    else:
        run_classification(args)

if __name__ == '__main__':
    main()
