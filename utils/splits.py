from __future__ import annotations

from typing import Tuple

import torch


def random_node_split(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    train_count = max(1, int(num_nodes * train_ratio))
    val_count = max(1, int(num_nodes * val_ratio))
    if train_count + val_count >= num_nodes:
        val_count = max(1, num_nodes - train_count - 1)
    test_count = max(1, num_nodes - train_count - val_count)
    train_end = train_count
    val_end = train_count + val_count

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[permutation[:train_end]] = True
    val_mask[permutation[train_end:val_end]] = True
    test_mask[permutation[val_end : val_end + test_count]] = True
    return train_mask, val_mask, test_mask
