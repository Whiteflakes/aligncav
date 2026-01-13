"""
Data Augmentation for Beam Images.

This module provides augmentation transforms specifically
designed for beam mode images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def get_train_transforms(
    image_size: int = 256,
    rotation_limit: float = 15.0,
    scale_limit: float = 0.1,
    noise_level: float = 0.05,
) -> Any:
    """
    Get training augmentation transforms.

    Args:
        image_size: Output image size
        rotation_limit: Maximum rotation in degrees
        scale_limit: Maximum scale change (relative)
        noise_level: Gaussian noise standard deviation

    Returns:
        Albumentations Compose transform
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("albumentations is required for data augmentation")

    return A.Compose(
        [
            # Geometric transforms
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=scale_limit,
                rotate_limit=rotation_limit,
                border_mode=0,
                value=0,
                p=0.5,
            ),
            # Brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3,
            ),
            # Noise
            A.GaussNoise(
                var_limit=(0, noise_level * 255),
                p=0.3,
            ),
            # Blur (simulates defocus)
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),
            # Resize to target size
            A.Resize(image_size, image_size),
        ]
    )


def get_val_transforms(image_size: int = 256) -> Any:
    """
    Get validation transforms (no augmentation).

    Args:
        image_size: Output image size

    Returns:
        Albumentations Compose transform
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("albumentations is required for data transforms")

    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ]
    )


class BeamAugmentation:
    """
    Custom augmentation class for beam images.

    Provides physics-aware augmentations.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float] = (-10, 10),
        noise_level: float = 0.05,
        intensity_range: Tuple[float, float] = (0.9, 1.1),
    ):
        """
        Initialize augmentation.

        Args:
            rotation_range: (min, max) rotation in degrees
            scale_range: (min, max) scale factor
            translation_range: (min, max) translation in pixels
            noise_level: Noise standard deviation
            intensity_range: (min, max) intensity scale
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_level = noise_level
        self.intensity_range = intensity_range

    def __call__(self, image: NDArray) -> Dict[str, NDArray]:
        """
        Apply augmentation.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            Dictionary with 'image' key
        """
        import cv2

        h, w = image.shape[:2]

        # Random rotation
        angle = np.random.uniform(*self.rotation_range)
        center = (w / 2, h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Random scale
        scale = np.random.uniform(*self.scale_range)
        M_rot[0, 0] *= scale
        M_rot[0, 1] *= scale
        M_rot[1, 0] *= scale
        M_rot[1, 1] *= scale

        # Random translation
        tx = np.random.uniform(*self.translation_range)
        ty = np.random.uniform(*self.translation_range)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply geometric transform
        image = cv2.warpAffine(image, M_rot, (w, h), borderValue=0)

        # Random intensity
        intensity = np.random.uniform(*self.intensity_range)
        image = np.clip(image * intensity, 0, 1)

        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)

        return {"image": image.astype(np.float32)}


class MixupAugmentation:
    """
    Mixup augmentation for mode classification.

    Mixes two images and their labels.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        image1: NDArray,
        label1: int,
        image2: NDArray,
        label2: int,
        num_classes: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        Apply mixup.

        Args:
            image1, image2: Images to mix
            label1, label2: Labels to mix
            num_classes: Total number of classes

        Returns:
            Tuple of (mixed_image, mixed_label)
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2

        # Mix labels (one-hot)
        label_onehot1 = np.zeros(num_classes, dtype=np.float32)
        label_onehot1[label1] = 1.0

        label_onehot2 = np.zeros(num_classes, dtype=np.float32)
        label_onehot2[label2] = 1.0

        mixed_label = lam * label_onehot1 + (1 - lam) * label_onehot2

        return mixed_image.astype(np.float32), mixed_label


class CutoutAugmentation:
    """
    Cutout augmentation (random erasing).

    Randomly masks square regions of the image.
    """

    def __init__(
        self,
        num_holes: int = 1,
        max_h_size: int = 32,
        max_w_size: int = 32,
        fill_value: float = 0.0,
    ):
        """
        Initialize cutout.

        Args:
            num_holes: Number of holes to cut
            max_h_size: Maximum hole height
            max_w_size: Maximum hole width
            fill_value: Value to fill holes with
        """
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def __call__(self, image: NDArray) -> Dict[str, NDArray]:
        """Apply cutout."""
        h, w = image.shape[:2]
        image = image.copy()

        for _ in range(self.num_holes):
            hole_h = np.random.randint(1, self.max_h_size + 1)
            hole_w = np.random.randint(1, self.max_w_size + 1)

            y = np.random.randint(0, h)
            x = np.random.randint(0, w)

            y1 = max(0, y - hole_h // 2)
            y2 = min(h, y + hole_h // 2)
            x1 = max(0, x - hole_w // 2)
            x2 = min(w, x + hole_w // 2)

            image[y1:y2, x1:x2] = self.fill_value

        return {"image": image}
