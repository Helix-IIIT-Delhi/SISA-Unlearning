"""Model Merging for SISA++"""

import torch
import torch.nn as nn
from typing import List
from copy import deepcopy


class ModelMerger:
    """
    SISA++ Model Merger using Weight Averaging.

    Merges shard models through weighted averaging of parameters.
    """

    def __init__(self, shard_models: List[nn.Module]):
        self.shard_models = shard_models
        self.num_shards = len(shard_models)

    def merge(
        self,
        models: List[nn.Module] = None,
        weights: List[float] = None
    ) -> nn.Module:
        """
        Merge shard models using weight averaging.

        Args:
            models: Models to merge (default: self.shard_models)
            weights: Weights for each model (default: uniform)

        Returns:
            Merged model
        """
        if models is None:
            models = self.shard_models

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Create merged model as deep copy of first model
        merged_model = deepcopy(models[0])

        # Average parameters across all models
        with torch.no_grad():
            for param in merged_model.parameters():
                param.zero_()

            # Add weighted contributions from each model
            for model, weight in zip(models, weights):
                for merged_param, model_param in zip(
                    merged_model.parameters(),
                    model.parameters()
                ):
                    merged_param.add_(model_param, alpha=weight)

        return merged_model

    def fine_tune_merged(
        self,
        merged_model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 2,
        lr: float = 1e-4
    ) -> nn.Module:
        """
        Light fine-tuning after merging.

        Args:
            merged_model: Merged model
            data: Training data
            labels: Training labels
            epochs: Fine-tuning epochs
            lr: Learning rate

        Returns:
            Fine-tuned merged model
        """
        optimizer = torch.optim.Adam(merged_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = merged_model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        return merged_model
