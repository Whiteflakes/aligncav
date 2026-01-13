"""
AlignCav - Fabry-Pérot Optical Cavity Beam Alignment using Deep Learning.

This package provides tools for:
- Simulating Hermite-Gaussian beam modes
- Classifying transverse modes using CNNs
- Aligning cavities using Deep Reinforcement Learning
- Interfacing with hardware (motors, cameras, power meters)

Example Usage:
    >>> from aligncav.simulation import HGModeGenerator, ModeParameters
    >>> params = ModeParameters(wavelength=1064e-9, waist=100e-6)
    >>> generator = HGModeGenerator(params=params, max_mode=10)
    >>> mode_00 = generator.generate_mode(0, 0)

    >>> from aligncav.models import ModeClassifier, DQNAgent
    >>> classifier = ModeClassifier(num_classes=121)
    >>> agent = DQNAgent(input_size=256, num_actions=81)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__author__ = "Haraprasad Nandi"
__email__ = "haraprasadnandi@iisertvm.ac.in"

# Lazy imports to handle optional dependencies gracefully
_LAZY_IMPORTS = {
    # Simulation module
    "HGModeGenerator": "aligncav.simulation",
    "ModeParameters": "aligncav.simulation",
    "CavitySimulator": "aligncav.simulation",
    "CavityConfig": "aligncav.simulation",
    "CavityEnvironment": "aligncav.simulation",
    "AlignmentState": "aligncav.simulation",
    "BeamProfileEvaluator": "aligncav.simulation",
    "BeamMetrics": "aligncav.simulation",
    "FresnelPropagator": "aligncav.simulation",
    "compute_beam_quality": "aligncav.simulation",
    "propagate_beam": "aligncav.simulation",
    
    # Models module
    "ModeClassifier": "aligncav.models",
    "DeepModeClassifier": "aligncav.models",
    "DQN": "aligncav.models",
    "DQNAgent": "aligncav.models",
    "ReplayBuffer": "aligncav.models",
    "PrioritizedReplayBuffer": "aligncav.models",
    "Transition": "aligncav.models",
    "decode_action": "aligncav.models",
    "encode_action": "aligncav.models",
    "ResNetFeatureExtractor": "aligncav.models",
    "ModalDecompositionCNN": "aligncav.models",
    
    # Training module
    "ClassifierTrainer": "aligncav.training",
    "RLTrainer": "aligncav.training",
    "TrainingConfig": "aligncav.training",
    "FocalLoss": "aligncav.training",
    "LabelSmoothingLoss": "aligncav.training",
    "HuberLoss": "aligncav.training",
    "AccuracyMetric": "aligncav.training",
    "TopKAccuracyMetric": "aligncav.training",
    "ModeDistanceMetric": "aligncav.training",
    "ConfusionMatrix": "aligncav.training",
    "MetricCollection": "aligncav.training",
    "RewardTracker": "aligncav.training",
    
    # Hardware module
    "MotorController": "aligncav.hardware",
    "ArduinoMotorController": "aligncav.hardware",
    "SimulatedMotorController": "aligncav.hardware",
    "VideoStream": "aligncav.hardware",
    "OpenCVVideoStream": "aligncav.hardware",
    "HTTPVideoStream": "aligncav.hardware",
    "SimulatedVideoStream": "aligncav.hardware",
    "PowerMeter": "aligncav.hardware",
    "ThorlabsPM100A": "aligncav.hardware",
    "SimulatedPowerMeter": "aligncav.hardware",
    
    # Data module
    "ModeDataset": "aligncav.data",
    "SimulatedModeDataset": "aligncav.data",
    "ModalDecompositionDataset": "aligncav.data",
    "create_train_val_loaders": "aligncav.data",
    "get_default_train_transforms": "aligncav.data",
    "get_default_eval_transforms": "aligncav.data",
}

# Submodules
_SUBMODULES = frozenset([
    "simulation",
    "models",
    "training",
    "hardware",
    "data",
    "cli",
])


def __getattr__(name: str) -> Any:
    """Lazy import handler for package attributes."""
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    
    if name in _SUBMODULES:
        return importlib.import_module(f"aligncav.{name}")
    
    raise AttributeError(f"module 'aligncav' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return available attributes for autocomplete."""
    return list(_LAZY_IMPORTS.keys()) + list(_SUBMODULES) + [
        "__version__",
        "__author__",
        "__email__",
    ]


if TYPE_CHECKING:
    # Type hints for static analysis
    from aligncav.data import (
        ModalDecompositionDataset,
        ModeDataset,
        SimulatedModeDataset,
        create_train_val_loaders,
        get_default_eval_transforms,
        get_default_train_transforms,
    )
    from aligncav.hardware import (
        ArduinoMotorController,
        HTTPVideoStream,
        MotorController,
        OpenCVVideoStream,
        PowerMeter,
        SimulatedMotorController,
        SimulatedPowerMeter,
        SimulatedVideoStream,
        ThorlabsPM100A,
        VideoStream,
    )
    from aligncav.models import (
        DQN,
        DQNAgent,
        DeepModeClassifier,
        ModalDecompositionCNN,
        ModeClassifier,
        PrioritizedReplayBuffer,
        ReplayBuffer,
        ResNetFeatureExtractor,
        Transition,
        decode_action,
        encode_action,
    )
    from aligncav.simulation import (
        AlignmentState,
        BeamMetrics,
        BeamProfileEvaluator,
        CavityConfig,
        CavityEnvironment,
        CavitySimulator,
        FresnelPropagator,
        HGModeGenerator,
        ModeParameters,
        compute_beam_quality,
        propagate_beam,
    )
    from aligncav.training import (
        AccuracyMetric,
        ClassifierTrainer,
        ConfusionMatrix,
        FocalLoss,
        HuberLoss,
        LabelSmoothingLoss,
        MetricCollection,
        ModeDistanceMetric,
        RewardTracker,
        RLTrainer,
        TopKAccuracyMetric,
        TrainingConfig,
    )


__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Simulation
    "HGModeGenerator",
    "ModeParameters",
    "CavitySimulator",
    "CavityConfig",
    "CavityEnvironment",
    "AlignmentState",
    "BeamProfileEvaluator",
    "BeamMetrics",
    "FresnelPropagator",
    "compute_beam_quality",
    "propagate_beam",
    # Models
    "ModeClassifier",
    "DeepModeClassifier",
    "DQN",
    "DQNAgent",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "Transition",
    "decode_action",
    "encode_action",
    "ResNetFeatureExtractor",
    "ModalDecompositionCNN",
    # Training
    "ClassifierTrainer",
    "RLTrainer",
    "TrainingConfig",
    "FocalLoss",
    "LabelSmoothingLoss",
    "HuberLoss",
    "AccuracyMetric",
    "TopKAccuracyMetric",
    "ModeDistanceMetric",
    "ConfusionMatrix",
    "MetricCollection",
    "RewardTracker",
    # Hardware
    "MotorController",
    "ArduinoMotorController",
    "SimulatedMotorController",
    "VideoStream",
    "OpenCVVideoStream",
    "HTTPVideoStream",
    "SimulatedVideoStream",
    "PowerMeter",
    "ThorlabsPM100A",
    "SimulatedPowerMeter",
    # Data
    "ModeDataset",
    "SimulatedModeDataset",
    "ModalDecompositionDataset",
    "create_train_val_loaders",
    "get_default_train_transforms",
    "get_default_eval_transforms",
]
