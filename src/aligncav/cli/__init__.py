"""
Command-line interface for AlignCav.

Provides CLI tools for training, inference, and simulation.
"""

from . import predict, simulate, train_classifier, train_rl

__all__ = [
    "train_classifier",
    "train_rl",
    "predict",
    "simulate",
]
