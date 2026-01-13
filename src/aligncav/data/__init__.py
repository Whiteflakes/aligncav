"""
Data module for AlignCav.

Provides dataset classes, augmentation, and data loading utilities.
"""

from .augmentation import (
    BeamAugmentation,
    CutoutAugmentation,
    MixupAugmentation,
    get_train_transforms,
    get_val_transforms,
)
from .dataset import (
    ModalDecompositionDataset,
    ModeDataset,
    RealModeDataset,
    SimulatedModeDataset,
)
from .loaders import (
    InfiniteDataLoader,
    create_dataloaders,
    create_inference_loader,
    create_mode_dataloaders,
)

__all__ = [
    # Datasets
    "ModeDataset",
    "SimulatedModeDataset",
    "ModalDecompositionDataset",
    "RealModeDataset",
    # Augmentation
    "get_train_transforms",
    "get_val_transforms",
    "BeamAugmentation",
    "MixupAugmentation",
    "CutoutAugmentation",
    # Loaders
    "create_dataloaders",
    "create_mode_dataloaders",
    "create_inference_loader",
    "InfiniteDataLoader",
]
