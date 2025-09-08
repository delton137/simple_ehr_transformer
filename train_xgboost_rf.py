#!/usr/bin/env python3
"""
Train and evaluate an XGBoost classifier using RF-style matrices.

Loads train matrices from a directory (e.g., processed_data_.../train_rf) and
test matrices from another directory (e.g., processed_data_.../test_rf),
handling either dense (X.npy) or sparse (X_sparse.npz) formats.

Outputs:
- Saves trained model to <train_rf_dir>/xgb_model.json
- Prints accuracy, ROC-AUC, PR-AUC, F1, and confusion matrix on the test set
"""

import os
import argparse
import json
import numpy as np
from typing import Tuple

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

try:
    import scipy.sparse as sp
except Exception:
    sp = None

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is required. Please install with: pip install xgboost")


def load_matrix_set(dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X and y from a directory. Supports dense .npy or sparse .npz."""
    x_sparse_path = os.path.join(dir_path, 'X_sparse.npz')
    x_dense_path = os.path.join(dir_path, 'X.npy')
    y_path = os.path.join(dir_path, 'y.npy')
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing y.npy in {dir_path}")
    y = np.load(y_path)
    if os.path.exists(x_sparse_path):
        if sp is None:
            raise RuntimeError("Found X_sparse.npz but scipy is not available. Install scipy or save dense X.npy")
        X = sp.load_npz(x_sparse_path)
    elif os.path.exists(x_dense_path):
        X = np.load(x_dense_path)
    else:
        raise FileNotFoundError(f"Missing X.npy or X_sparse.npz in {dir_path}")
    return X, y


def main():
    ap = argparse.ArgumentParser(description='Train and evaluate XGBoost on RF matrices')
    ap.add_argument('--train_dir', required=True, help='Directory with train RF matrices (X.npy/X_sparse.npz, y.npy)')
    ap.add_argument('--test_dir', required=True, help='Directory with test RF matrices (X.npy/X_sparse.npz, y.npy)')
    ap.add_argument('--model_out', default=None, help='Path to save model (default: <train_dir>/xgb_model.json)')
    ap.add_argument('--max_depth', type=int, default=8)
    ap.add_argument('--n_estimators', type=int, default=500)
    ap.add_argument('--learning_rate', type=float, default=0.05)
    ap.add_argument('--subsample', type=float, default=0.8)
    ap.add_argument('--colsample_bytree', type=float, default=0.8)
    ap.add_argument('--reg_lambda', type=float, default=1.0)
    ap.add_argument('--reg_alpha', type=float, default=0.0)
    ap.add_argument('--n_jobs', type=int, default=-1)
    args = ap.parse_args()

    X_train, y_train = load_matrix_set(args.train_dir)
    X_test, y_test = load_matrix_set(args.test_dir)

    # XGBoost supports scipy.sparse CSR directly
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        n_jobs=args.n_jobs,
        tree_method='hist',
        eval_metric='logloss',
        verbosity=1,
    )

    model.fit(X_train, y_train)

    # Predict probabilities for metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.uint8)

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = float('nan')
    try:
        prauc = average_precision_score(y_test, y_prob)
    except Exception:
        prauc = float('nan')
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(json.dumps({
        'accuracy': acc,
        'roc_auc': roc,
        'pr_auc': prauc,
        'f1': f1,
        'confusion_matrix': cm,
        'positives_test': int(np.sum(y_test)),
        'negatives_test': int(len(y_test) - np.sum(y_test)),
    }, indent=2))

    model_out = args.model_out or os.path.join(args.train_dir, 'xgb_model.json')
    model.save_model(model_out)
    print(f"Saved XGBoost model to {model_out}")


if __name__ == '__main__':
    main()


