"""Metrics for Evaluating Machine Unlearning"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_unlearning_metrics(
    original_predictions: np.ndarray,
    unlearned_predictions: np.ndarray,
    targets: np.ndarray
) -> dict:
    """
    Compute metrics for unlearning evaluation.

    Args:
        original_predictions: Predictions before unlearning
        unlearned_predictions: Predictions after unlearning
        targets: Ground truth labels

    Returns:
        Dictionary with unlearning metrics
    """
    original_acc = accuracy_score(targets, original_predictions)
    unlearned_acc = accuracy_score(targets, unlearned_predictions)

    performance_loss = original_acc - unlearned_acc

    return {
        "original_accuracy": original_acc,
        "unlearned_accuracy": unlearned_acc,
        "performance_loss": performance_loss,
        "retention_rate": unlearned_acc / original_acc if original_acc > 0 else 0,
    }


def compute_class_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute per-class metrics"""
    return {
        "precision": precision_score(targets, predictions, average='weighted'),
        "recall": recall_score(targets, predictions, average='weighted'),
        "f1": f1_score(targets, predictions, average='weighted'),
    }
