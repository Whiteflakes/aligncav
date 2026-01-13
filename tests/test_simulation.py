"""
Tests for the simulation module.
"""

import numpy as np
import pytest

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


class TestModeParameters:
    """Tests for ModeParameters."""

    def test_default_values(self):
        """Test default parameter values."""
        params = ModeParameters()
        assert params.wavelength == 1064e-9
        assert params.waist == 100e-6
        assert params.image_size == 256

    def test_wave_number(self):
        """Test wave number calculation."""
        params = ModeParameters()
        expected_k = 2 * np.pi / params.wavelength
        assert np.isclose(params.k, expected_k)

    def test_rayleigh_range(self):
        """Test Rayleigh range calculation."""
        params = ModeParameters()
        expected_zr = np.pi * params.waist**2 / params.wavelength
        assert np.isclose(params.rayleigh_range, expected_zr)


class TestHGModeGenerator:
    """Tests for HGModeGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator fixture."""
        params = ModeParameters(image_size=64)
        return HGModeGenerator(params=params, max_mode=5, add_noise=False)

    def test_generate_fundamental_mode(self, generator):
        """Test generating TEM00 mode."""
        mode = generator.generate_mode(0, 0)
        
        assert mode.shape == (64, 64)
        assert mode.dtype == np.float32
        assert mode.min() >= 0
        assert mode.max() <= 1

    def test_generate_higher_order_mode(self, generator):
        """Test generating higher order modes."""
        mode_10 = generator.generate_mode(1, 0)
        mode_01 = generator.generate_mode(0, 1)
        
        # Higher order modes should have different patterns
        assert not np.allclose(mode_10, mode_01)

    def test_class_encoding(self, generator):
        """Test class index encoding/decoding."""
        for m in range(6):
            for n in range(6):
                class_idx = generator.get_class_from_indices(m, n)
                decoded_m, decoded_n = generator.get_indices_from_class(class_idx)
                assert decoded_m == m
                assert decoded_n == n

    def test_num_classes(self, generator):
        """Test number of classes."""
        assert generator.num_classes == 36  # (5+1)^2

    def test_generate_with_variations(self, generator):
        """Test mode generation with variations."""
        mode1 = generator.generate_mode(0, 0, waist_scale=1.0)
        mode2 = generator.generate_mode(0, 0, waist_scale=1.2)
        
        # Different waist scales should produce different patterns
        assert not np.allclose(mode1, mode2)

    def test_generate_superposition(self, generator):
        """Test superposition generation."""
        modes = [(0, 0, 0.5), (1, 0, 0.3), (0, 1, 0.2)]
        superposition = generator.generate_superposition(modes)
        
        assert superposition.shape == (64, 64)
        assert superposition.max() <= 1


class TestCavityConfig:
    """Tests for CavityConfig."""

    def test_finesse(self):
        """Test cavity finesse calculation."""
        config = CavityConfig(mirror1_reflectivity=0.99, mirror2_reflectivity=0.99)
        # Finesse should be high for high reflectivity
        assert config.finesse > 100

    def test_stability(self):
        """Test cavity stability check."""
        # Symmetric confocal cavity should be stable
        config = CavityConfig(
            mirror1_radius=0.1,
            mirror2_radius=0.1,
            length=0.1,
        )
        assert config.is_stable

    def test_g_parameters(self):
        """Test g-parameter calculation."""
        config = CavityConfig(
            mirror1_radius=0.5,
            mirror2_radius=0.5,
            length=0.1,
        )
        g1, g2 = config.g_parameters
        assert np.isclose(g1, 0.8)
        assert np.isclose(g2, 0.8)


class TestCavitySimulator:
    """Tests for CavitySimulator."""

    @pytest.fixture
    def simulator(self):
        """Create simulator fixture."""
        config = CavityConfig(image_size=64)
        return CavitySimulator(config)

    def test_reset(self, simulator):
        """Test simulator reset."""
        simulator.reset(random_misalignment=True)
        state = simulator.state.as_array()
        
        # Should have some non-zero misalignment
        assert not np.allclose(state, 0)

    def test_apply_action(self, simulator):
        """Test action application."""
        simulator.reset()
        initial_state = simulator.state.as_array().copy()
        
        action = np.array([1, 0, -1, 0], dtype=np.float32)
        simulator.apply_action(action)
        
        new_state = simulator.state.as_array()
        # State should change after applying action
        assert not np.array_equal(initial_state, new_state)

    def test_get_transmitted_beam(self, simulator):
        """Test transmitted beam calculation."""
        simulator.reset()
        beam = simulator.get_transmitted_beam()
        
        assert beam.shape == (64, 64)
        assert beam.dtype == np.float32

    def test_alignment_quality(self, simulator):
        """Test alignment quality metric."""
        simulator.reset()
        quality = simulator.get_alignment_quality()
        
        assert 0 <= quality <= 1


class TestCavityEnvironment:
    """Tests for CavityEnvironment."""

    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        config = CavityConfig(image_size=64)
        return CavityEnvironment(config, max_steps=50)

    def test_reset(self, env):
        """Test environment reset."""
        obs = env.reset()
        
        assert obs.shape == (64, 64)
        assert env.current_step == 0

    def test_step(self, env):
        """Test environment step."""
        env.reset()
        obs, reward, done, info = env.step(40)  # Middle action
        
        assert obs.shape == (64, 64)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "alignment_quality" in info

    def test_action_space(self, env):
        """Test action space size."""
        assert env.num_actions == 81  # 3^4


class TestBeamProfileEvaluator:
    """Tests for BeamProfileEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator fixture."""
        return BeamProfileEvaluator()

    @pytest.fixture
    def gaussian_image(self):
        """Create test Gaussian image."""
        x = np.linspace(-1, 1, 64)
        y = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, y)
        return np.exp(-(X**2 + Y**2) / 0.2).astype(np.float32)

    def test_compute_centroid(self, evaluator, gaussian_image):
        """Test centroid computation."""
        cx, cy = evaluator.compute_centroid(gaussian_image)
        
        # Should be near center
        assert np.isclose(cx, 32, atol=1)
        assert np.isclose(cy, 32, atol=1)

    def test_compute_beam_widths(self, evaluator, gaussian_image):
        """Test beam width computation."""
        sigma_x, sigma_y = evaluator.compute_beam_widths(gaussian_image)
        
        # Should be positive and symmetric
        assert sigma_x > 0
        assert sigma_y > 0
        assert np.isclose(sigma_x, sigma_y, rtol=0.1)

    def test_analyze(self, evaluator, gaussian_image):
        """Test full analysis."""
        metrics = evaluator.analyze(gaussian_image)
        
        assert isinstance(metrics, BeamMetrics)
        assert 0 < metrics.ellipticity <= 1

    def test_compute_correlation(self, evaluator, gaussian_image):
        """Test correlation computation."""
        corr = evaluator.compute_correlation(gaussian_image, gaussian_image)
        
        # Self-correlation should be 1
        assert np.isclose(corr, 1.0, atol=0.01)

    def test_alignment_reward(self, evaluator, gaussian_image):
        """Test alignment reward computation."""
        reward = evaluator.compute_alignment_reward(gaussian_image, gaussian_image)
        
        # Perfect match should give high reward
        assert reward > 0.9


class TestFresnelPropagator:
    """Tests for FresnelPropagator."""

    @pytest.fixture
    def propagator(self):
        """Create propagator fixture."""
        return FresnelPropagator(grid_size=64)

    def test_create_gaussian_beam(self, propagator):
        """Test Gaussian beam creation."""
        beam = propagator.create_gaussian_beam(waist=50e-6)
        
        assert beam.shape == (64, 64)
        assert beam.dtype == np.complex128

    def test_propagate(self, propagator):
        """Test beam propagation."""
        beam = propagator.create_gaussian_beam(waist=50e-6)
        propagated = propagator.propagate(beam, distance=0.01)
        
        assert propagated.shape == (64, 64)
        assert propagated.dtype == np.float64

    def test_create_hermite_gaussian(self, propagator):
        """Test HG mode creation."""
        hg00 = propagator.create_hermite_gaussian(waist=50e-6, m=0, n=0)
        hg10 = propagator.create_hermite_gaussian(waist=50e-6, m=1, n=0)
        
        # Different modes should be different
        assert not np.allclose(hg00, hg10)


def test_compute_beam_quality():
    """Test convenience function."""
    # Create a simple Gaussian
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2) / 0.2).astype(np.float32)
    
    quality = compute_beam_quality(gaussian)
    assert 0 <= quality <= 1


def test_propagate_beam():
    """Test convenience propagation function."""
    # Create simple field
    field = np.zeros((64, 64), dtype=np.complex128)
    field[28:36, 28:36] = 1.0
    
    result = propagate_beam(field, distance=0.01)
    assert result.shape == (64, 64)
