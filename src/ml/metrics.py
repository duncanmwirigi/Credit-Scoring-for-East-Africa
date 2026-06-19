from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "gini": float(gini_coefficient(y_true, y_prob)),
        "ks": float(ks_statistic(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
