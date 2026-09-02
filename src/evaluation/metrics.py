# type: ignore
"""Evaluation metrics for MRI quality classification."""

from typing import TypedDict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


class ClassificationMetrics(TypedDict):
    """Metrics returned for a binary classification task."""

    accuracy: float
    auc: float
    sensitivity: float
    specificity: float
    confusion_matrix: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray
) -> ClassificationMetrics:
    """
    Compute binary classification metrics from labels and scores.

    Args:
        y_true: ground-truth binary labels.
        y_pred: predicted binary labels.
        y_score: predicted probabilities or scores for the positive class.
    """
    accuracy = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': matrix
    }
