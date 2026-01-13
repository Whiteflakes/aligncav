"""
Simulation module for Fabry-Pérot cavity and beam physics.

This module provides tools for simulating Hermite-Gaussian beam modes,
Fabry-Pérot cavity physics, and beam propagation.
"""

from .beam_profile import BeamMetrics, BeamProfileEvaluator, compute_beam_quality
from .cavity import AlignmentState, CavityConfig, CavityEnvironment, CavitySimulator
from .fresnel import FresnelPropagator, propagate_beam
from .mode_generator import HGModeGenerator, ModeParameters

__all__ = [
    # Mode generation
    "HGModeGenerator",
    "ModeParameters",
    # Cavity simulation
    "CavitySimulator",
    "CavityConfig",
    "CavityEnvironment",
    "AlignmentState",
    # Beam analysis
    "BeamProfileEvaluator",
    "BeamMetrics",
    "compute_beam_quality",
    # Fresnel propagation
    "FresnelPropagator",
    "propagate_beam",
]
