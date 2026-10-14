"""SISA++ Unlearner Implementation"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .sharding import DataSharding
from .merger import ModelMerger


class SISAPlusUnlearner(nn.Module):
    """
    SISA++ Unlearner for Paralinguistic Speech Processing

    Implements the SISA++ algorithm with weight averaging for
    machine unlearning in Speech Emotion Recognition and Depression Detection.

    Args:
        num_shards: Number of data shards
        model_type: 'SER' or 'DD'
        num_classes: Number of output classes
        feature_dim: Feature dimension
    """

    def __init__(
        self,
        num_shards: int = 4,
        model_type: str = 'SER',
        num_classes: int = 7,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.num_shards = num_shards
        self.model_type = model_type
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Shard models
        self.shard_models = nn.ModuleList([
            self._create_model(num_classes, feature_dim)
            for _ in range(num_shards)
        ])

        # Track which data belongs to which shard
        self.data_sharding = DataSharding(num_shards)

        # Model merger
        self.merger = ModelMerger(self.shard_models)

        # Merged model (will be created after training)
        self.merged_model = None

        # Unlearning history
        self.unlearned_ids = set()

    def _create_model(self, num_classes: int, feature_dim: int) -> nn.Module:
        """Create a simple speech processing model"""
        return nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def train_shards(self, train_data, labels, epochs: int = 10, lr: float = 1e-3):
        """
        Train each shard model independently.

        Args:
            train_data: Training features
            labels: Training labels
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Shard the data
        shard_indices = self.data_sharding.shard(train_data)

        # Train each shard
        for shard_id, indices in enumerate(shard_indices):
            shard_data = train_data[indices]
            shard_labels = labels[indices]

            self._train_shard(
                shard_id,
                shard_data,
                shard_labels,
                epochs=epochs,
                lr=lr
            )

    def _train_shard(
        self,
        shard_id: int,
        data: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 10,
        lr: float = 1e-3
    ):
        """Train a single shard model"""
        model = self.shard_models[shard_id]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def unlearn(self, target: str):
        """
        Unlearn specific data from the models.

        Args:
            target: Specification of what to unlearn
                   Format: 'speaker:SPEAKER_ID' or 'sample:SAMPLE_ID'
        """
        # Parse unlearning target
        target_type, target_id = target.split(':')

        # Find affected shards
        affected_shards = self.data_sharding.find_affected_shards(target_type, target_id)

        # Retrain affected shards (without the target data)
        for shard_id in affected_shards:
            shard_data, shard_labels = self.data_sharding.get_shard_data(
                shard_id,
                exclude=target
            )

            if len(shard_data) > 0:
                self._train_shard(shard_id, shard_data, shard_labels, epochs=5)

        # Merge models with weight averaging
        self.merged_model = self.merger.merge(self.shard_models)

        # Track unlearned data
        self.unlearned_ids.add(target)

    def get_merged_model(self) -> nn.Module:
        """Get the merged/unlearned model"""
        if self.merged_model is None:
            self.merged_model = self.merger.merge(self.shard_models)
        return self.merged_model

    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict:
        """
        Evaluate the unlearned model.

        Args:
            test_data: Test features
            test_labels: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        model = self.get_merged_model()
        model.eval()

        with torch.no_grad():
            logits = model(test_data)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()

        return {
            "accuracy": accuracy,
            "unlearned_count": len(self.unlearned_ids),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using merged model"""
        model = self.get_merged_model()
        return model(x)
