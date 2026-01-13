"""
Data Loaders for Training.

This module provides utilities for creating PyTorch DataLoaders
with proper train/val splits and batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .augmentation import get_train_transforms, get_val_transforms
from .dataset import ModeDataset, SimulatedModeDataset


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle training data
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Create reproducible split
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_mode_dataloaders(
    data_dir: Optional[str | Path] = None,
    batch_size: int = 32,
    image_size: int = 256,
    max_mode: int = 10,
    val_split: float = 0.2,
    num_workers: int = 4,
    use_simulated: bool = True,
    simulated_size: int = 10000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for mode classification.

    Args:
        data_dir: Directory with real data (if not using simulated)
        batch_size: Batch size
        image_size: Image size
        max_mode: Maximum mode index
        val_split: Validation split fraction
        num_workers: Number of workers
        use_simulated: Use simulated data
        simulated_size: Size of simulated dataset

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)

    if use_simulated:
        # Create separate train and val datasets with different transforms
        train_dataset = SimulatedModeDataset(
            size=int(simulated_size * (1 - val_split)),
            max_mode=max_mode,
            image_size=image_size,
            transform=train_transform,
            include_variations=True,
        )

        val_dataset = SimulatedModeDataset(
            size=int(simulated_size * val_split),
            max_mode=max_mode,
            image_size=image_size,
            transform=val_transform,
            include_variations=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader
    else:
        if data_dir is None:
            raise ValueError("data_dir must be provided when not using simulated data")

        # Load real data
        full_dataset = ModeDataset(
            root=data_dir,
            max_mode=max_mode,
        )

        return create_dataloaders(
            full_dataset,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
        )


def create_inference_loader(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 256,
    max_mode: int = 10,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create dataloader for inference (no augmentation).

    Args:
        data_dir: Directory with images
        batch_size: Batch size
        image_size: Image size
        max_mode: Maximum mode index
        num_workers: Number of workers

    Returns:
        DataLoader for inference
    """
    transform = get_val_transforms(image_size=image_size)

    dataset = ModeDataset(
        root=data_dir,
        transform=transform,
        max_mode=max_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


class InfiniteDataLoader:
    """
    DataLoader that loops infinitely.

    Useful for RL training where we need continuous data.
    """

    def __init__(self, dataloader: DataLoader):
        """
        Initialize infinite loader.

        Args:
            dataloader: Base DataLoader
        """
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)

        return batch

    def __len__(self):
        return len(self.dataloader)
