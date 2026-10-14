"""Data Sharding for SISA++"""

import torch
import numpy as np
from typing import List, Dict, Tuple


class DataSharding:
    """
    Data sharding strategy for SISA++.

    Partitions data into shards for independent training.
    """

    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shard_assignment = {}
        self.data_metadata = {}

    def shard(self, data: torch.Tensor, metadata: Dict = None) -> List[List[int]]:
        """
        Partition data into shards.

        Args:
            data: Training data
            metadata: Optional metadata (speaker IDs, etc.)

        Returns:
            List of shard indices
        """
        n = len(data)
        shard_size = n // self.num_shards

        indices = np.arange(n)
        np.random.shuffle(indices)

        shard_indices = []
        for i in range(self.num_shards):
            start = i * shard_size
            end = start + shard_size if i < self.num_shards - 1 else n
            shard_indices.append(indices[start:end].tolist())

            # Track assignment
            for idx in indices[start:end]:
                self.shard_assignment[int(idx)] = i

        # Store metadata
        self.data_metadata = metadata or {}

        return shard_indices

    def find_affected_shards(self, target_type: str, target_id: str) -> List[int]:
        """
        Find which shards contain the target data.

        Args:
            target_type: 'speaker' or 'sample'
            target_id: ID of speaker/sample to unlearn

        Returns:
            List of affected shard indices
        """
        affected = set()

        for idx, shard_id in self.shard_assignment.items():
            if target_type == 'speaker':
                # Check if sample belongs to target speaker
                if self.data_metadata.get(f'speaker_{idx}') == target_id:
                    affected.add(shard_id)
            elif target_type == 'sample':
                if str(idx) == target_id:
                    affected.add(shard_id)

        return list(affected)

    def get_shard_data(
        self,
        shard_id: int,
        data: torch.Tensor = None,
        labels: torch.Tensor = None,
        exclude: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data for a specific shard.

        Args:
            shard_id: Shard index
            data: Full dataset
            labels: Full labels
            exclude: Data to exclude (format: 'speaker:ID')

        Returns:
            Shard data and labels
        """
        # Find indices for this shard
        indices = [idx for idx, s in self.shard_assignment.items() if s == shard_id]

        # Filter excluded data
        if exclude:
            target_type, target_id = exclude.split(':')
            indices = [
                idx for idx in indices
                if not (target_type == 'speaker' and
                       self.data_metadata.get(f'speaker_{idx}') == target_id)
            ]

        if len(indices) == 0:
            return torch.tensor([]), torch.tensor([])

        # Return shard data if provided
        if data is not None and labels is not None:
            return data[indices], labels[indices]

        return indices
