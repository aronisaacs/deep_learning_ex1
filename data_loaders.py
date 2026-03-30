from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def create_weighted_sampler(train_labels: torch.Tensor, num_classes: int) -> WeightedRandomSampler:
    """Create a sampler that oversamples minority classes to balance training."""
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sample_weights = sample_weights / sample_weights.sum() * len(train_labels)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )


def create_data_loaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    train_labels: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/test loaders with oversampling for optimization only."""
    sampler = create_weighted_sampler(train_labels=train_labels, num_classes=num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )

    return train_loader, train_eval_loader, test_loader


def create_basic_data_loaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/test loaders without oversampling."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )
    return train_loader, train_eval_loader, test_loader


