"""Utility Functions for SISA++"""

import torch
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def check_unlearning_verification(
    model_predictions_before: torch.Tensor,
    model_predictions_after: torch.Tensor,
    unlearned_sample_indices: list
) -> bool:
    """
    Verify that unlearning was successful.

    Args:
        model_predictions_before: Predictions before unlearning
        model_predictions_after: Predictions after unlearning
        unlearned_sample_indices: Indices of unlearned samples

    Returns:
        True if significant difference for unlearned samples
    """
    unlearned_before = model_predictions_before[unlearned_sample_indices]
    unlearned_after = model_predictions_after[unlearned_sample_indices]

    # Check if predictions changed for unlearned samples
    difference = torch.mean(torch.abs(unlearned_before - unlearned_after))

    return difference > 0.1  # Threshold for significant change
