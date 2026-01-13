"""
Dataset Classes for Mode Classification.

This module provides PyTorch Dataset implementations for
beam mode image data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset


class ModeDataset(Dataset):
    """
    Dataset for beam mode images.

    Loads images from disk with corresponding mode labels.
    Expected directory structure:
        root/
            class_0/
                image_0.png
                image_1.png
            class_1/
                ...
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        max_mode: int = 10,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory containing class folders
            transform: Transform to apply to images
            max_mode: Maximum mode index
            extensions: Valid image file extensions
        """
        self.root = Path(root)
        self.transform = transform
        self.max_mode = max_mode
        self.extensions = extensions
        self.num_classes = (max_mode + 1) ** 2

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load sample paths and labels."""
        if not self.root.exists():
            return

        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue

            try:
                class_idx = int(class_dir.name.split("_")[-1])
            except ValueError:
                continue

            if class_idx >= self.num_classes:
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("L")  # Grayscale
        image = np.array(image, dtype=np.float32) / 255.0

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0)

        return image, label


class SimulatedModeDataset(Dataset):
    """
    Dataset that generates mode images on-the-fly.

    Useful for training without pre-generated data.
    """

    def __init__(
        self,
        size: int = 10000,
        max_mode: int = 10,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        include_variations: bool = True,
        cache: bool = False,
    ):
        """
        Initialize simulated dataset.

        Args:
            size: Number of samples to generate
            max_mode: Maximum mode index
            image_size: Output image size
            transform: Transform to apply
            include_variations: Add random variations
            cache: Cache generated images
        """
        self.size = size
        self.max_mode = max_mode
        self.image_size = image_size
        self.transform = transform
        self.include_variations = include_variations
        self.cache = cache
        self.num_classes = (max_mode + 1) ** 2

        # Generator
        from ..simulation.mode_generator import HGModeGenerator, ModeParameters

        params = ModeParameters(image_size=image_size)
        self.generator = HGModeGenerator(
            params=params,
            max_mode=max_mode,
            add_noise=True,
        )

        # Cache storage
        self._cache: dict = {} if cache else None

        # Pre-assign labels to ensure balanced classes
        self._labels = np.random.randint(0, self.num_classes, size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Get pre-assigned label
        label = int(self._labels[idx])
        m, n = self.generator.get_indices_from_class(label)

        # Generate image
        if self.include_variations:
            image = self.generator.generate_mode(
                m,
                n,
                waist_scale=np.random.uniform(0.8, 1.2),
                rotation=np.random.uniform(-0.1, 0.1),
                offset=(np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
            )
        else:
            image = self.generator.generate_mode(m, n)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0)

        result = (image, label)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = result

        return result


class ModalDecompositionDataset(Dataset):
    """
    Dataset for modal decomposition training.

    Provides images with their mode coefficient labels
    (continuous values instead of discrete classes).
    """

    def __init__(
        self,
        size: int = 10000,
        max_mode: int = 10,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        superposition_prob: float = 0.3,
    ):
        """
        Initialize modal decomposition dataset.

        Args:
            size: Number of samples
            max_mode: Maximum mode index
            image_size: Output image size
            transform: Transform to apply
            superposition_prob: Probability of generating superposition
        """
        self.size = size
        self.max_mode = max_mode
        self.image_size = image_size
        self.transform = transform
        self.superposition_prob = superposition_prob
        self.num_modes = (max_mode + 1) ** 2

        # Generator
        from ..simulation.mode_generator import HGModeGenerator, ModeParameters

        params = ModeParameters(image_size=image_size)
        self.generator = HGModeGenerator(params=params, max_mode=max_mode)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample with mode coefficients."""
        coefficients = np.zeros(self.num_modes, dtype=np.float32)

        if np.random.random() < self.superposition_prob:
            # Generate superposition
            num_modes = np.random.randint(2, 5)
            modes = []
            weights = np.random.dirichlet(np.ones(num_modes))

            for i in range(num_modes):
                m = np.random.randint(0, self.max_mode + 1)
                n = np.random.randint(0, self.max_mode + 1)
                class_idx = self.generator.get_class_from_indices(m, n)
                coefficients[class_idx] += weights[i]
                modes.append((m, n, weights[i]))

            image = self.generator.generate_superposition(modes)
        else:
            # Single mode
            m = np.random.randint(0, self.max_mode + 1)
            n = np.random.randint(0, self.max_mode + 1)
            class_idx = self.generator.get_class_from_indices(m, n)
            coefficients[class_idx] = 1.0

            image = self.generator.generate_mode(
                m,
                n,
                waist_scale=np.random.uniform(0.9, 1.1),
            )

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0)

        coefficients = torch.from_numpy(coefficients)

        return image, coefficients


class RealModeDataset(Dataset):
    """
    Dataset for real experimental beam images.

    Handles loading and preprocessing of experimental data.
    """

    def __init__(
        self,
        image_dir: str | Path,
        labels_file: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        crop_center: bool = True,
        crop_size: int = 256,
    ):
        """
        Initialize real data dataset.

        Args:
            image_dir: Directory containing images
            labels_file: CSV file with image labels
            transform: Transform to apply
            crop_center: Whether to crop center of image
            crop_size: Size of center crop
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.crop_center = crop_center
        self.crop_size = crop_size

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples(labels_file)

    def _load_samples(self, labels_file: Optional[str | Path]) -> None:
        """Load samples from directory and labels file."""
        if labels_file and Path(labels_file).exists():
            import csv

            with open(labels_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = self.image_dir / row["filename"]
                    if img_path.exists():
                        label = int(row["label"])
                        self.samples.append((img_path, label))
        else:
            # Try to infer from filenames
            for img_path in self.image_dir.glob("*.png"):
                try:
                    # Assume format: mode_m_n_*.png
                    parts = img_path.stem.split("_")
                    m, n = int(parts[1]), int(parts[2])
                    label = m * 11 + n  # Assuming max_mode=10
                    self.samples.append((img_path, label))
                except (IndexError, ValueError):
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32)

        # Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Center crop if requested
        if self.crop_center:
            h, w = image.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            image = image[start_h : start_h + self.crop_size, start_w : start_w + self.crop_size]

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0)

        return image, label
