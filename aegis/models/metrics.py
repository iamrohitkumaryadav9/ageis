"""Classification metrics shared by training, monitoring and remediation validation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Equal-width ECE: mean |confidence - observed frequency| weighted by bin mass."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent and threshold-free metrics for a binary classifier.

    ROC-AUC is ``None`` when only one class is present, because it is undefined
    there and a fabricated number would be worse than an honest gap.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba),
        "positive_rate_true": float(y_true.mean()),
        "positive_rate_pred": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Decision threshold maximising F1 on a held-out set (grid of 0.01 steps)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
