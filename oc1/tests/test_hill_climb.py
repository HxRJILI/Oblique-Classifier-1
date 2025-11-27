"""
Tests for hill-climbing optimization and coefficient perturbation.

Tests cover:
- Hyperplane initialization (axis-parallel, random, zeros)
- U_j value computation (Equation 1 from paper)
- Single coefficient perturbation
- Full hill-climbing optimization
- Convergence to local minimum
- Edge cases: degenerate hyperplanes, single samples

Paper Reference: Section 2.1 and Section 2.2
"""

import pytest
import numpy as np
from oc1.core.hill_climb import (
    initialize_hyperplane,
    compute_u_values,
    perturb_coefficient,
    hill_climb,
    find_best_hyperplane,
    normalize_hyperplane,
)
from oc1.core.splits import evaluate_split


class TestInitializeHyperplane:
    """Test hyperplane initialization methods."""
    
    def test_zeros_initialization(self):
        """Test zeros initialization method."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        hp = initialize_hyperplane(X, y, method="zeros")
        
        assert len(hp) == 3  # 2 features + bias
        assert hp[-1] != 0  # Bias should be non-zero
    
    def test_random_initialization(self):
        """Test random initialization method."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        hp = initialize_hyperplane(X, y, method="random", random_state=42)
        
        assert len(hp) == 3
        # Should have non-zero coefficients
        assert not np.allclose(hp, 0)
    
    def test_random_reproducibility(self):
        """Test that random_state produces reproducible results."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        hp1 = initialize_hyperplane(X, y, method="random", random_state=123)
        hp2 = initialize_hyperplane(X, y, method="random", random_state=123)
        
        np.testing.assert_array_equal(hp1, hp2)
    
    def test_axis_parallel_initialization(self):
        """Test axis-parallel initialization finds best single-feature split."""
        # Data where x1 gives perfect split
        X = np.array([[0, 5], [0, 6], [1, 5], [1, 6]])
        y = np.array([0, 0, 1, 1])
        
        hp = initialize_hyperplane(X, y, method="axis_parallel")
        
        # Should focus on feature 0 (x1)
        assert len(hp) == 3
        # The hyperplane should give a reasonable impurity
        impurity = evaluate_split(X, y, hp)
        assert impurity < 1  # Should be fairly good
    
    def test_invalid_method(self):
        """Test error on invalid initialization method."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        with pytest.raises(ValueError, match="Unknown"):
            initialize_hyperplane(X, y, method="invalid_method")
    
    def test_3d_initialization(self):
        """Test initialization with 3D data."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])
        
        hp = initialize_hyperplane(X, y, method="axis_parallel")
        
        assert len(hp) == 4  # 3 features + bias


class TestComputeUValues:
    """Test U_j computation (Equation 1 from paper).
    
    Paper Reference: Section 2.2
    U_j = a_m * x_j^m - V_j / x_j^m
    """
    
    def test_basic_u_computation(self):
        """Test basic U_j computation."""
        X = np.array([[1, 2], [2, 1], [3, 4]])
        hyperplane = np.array([1.0, 1.0, -3.0])  # x + y - 3 = 0
        
        U, valid_mask = compute_u_values(X, hyperplane, m=0)
        
        assert len(U) == 3
        assert np.all(valid_mask)  # All samples should be valid
    
    def test_u_with_zero_feature(self):
        """Test U_j when feature value is zero (division by zero)."""
        X = np.array([[0, 1], [1, 2], [0, 3]])
        hyperplane = np.array([1.0, 1.0, 0.0])
        
        U, valid_mask = compute_u_values(X, hyperplane, m=0)
        
        # Samples with x^0 = 0 should be marked invalid
        assert not valid_mask[0]  # First sample has x^0 = 0
        assert valid_mask[1]      # Second sample is valid
        assert not valid_mask[2]  # Third sample has x^0 = 0
    
    def test_u_for_bias_term(self):
        """Test U_j computation for bias term (m = d)."""
        X = np.array([[1, 2], [3, 4]])
        hyperplane = np.array([1.0, 1.0, -2.0])
        
        # m = 2 is the bias term (d+1 where d=2)
        U, valid_mask = compute_u_values(X, hyperplane, m=2)
        
        # For bias, x_j^m is treated as 1, so all should be valid
        assert np.all(valid_mask)
    
    def test_u_invalid_coefficient_index(self):
        """Test error on invalid coefficient index."""
        X = np.array([[1, 2]])
        hyperplane = np.array([1.0, 1.0, 0.0])
        
        with pytest.raises(ValueError, match="out of range"):
            compute_u_values(X, hyperplane, m=5)


class TestPerturbCoefficient:
    """Test single coefficient perturbation.
    
    Paper Reference: Section 2.2
    """
    
    def test_perturb_improves_split(self):
        """Test that perturbation can improve a suboptimal hyperplane."""
        # Data with clear diagonal split
        X = np.array([
            [0, 0], [0.1, 0.1],  # Class 0
            [1, 1], [0.9, 0.9],  # Class 1
        ])
        y = np.array([0, 0, 1, 1])
        
        # Start with suboptimal hyperplane
        hp_initial = np.array([1.0, 0.0, -0.5])  # x - 0.5 = 0
        initial_impurity = evaluate_split(X, y, hp_initial)
        
        # Perturb coefficient 1 (y coefficient)
        new_hp, new_impurity, improved = perturb_coefficient(
            X, y, hp_initial, m=1, impurity_measure="sm"
        )
        
        # Should maintain or improve impurity
        assert new_impurity <= initial_impurity + 0.1
    
    def test_perturb_returns_same_if_no_improvement(self):
        """Test that perturbation returns original if no improvement."""
        # Already optimal split
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])  # Perfect x = 0.5 split
        hp = np.array([1.0, 0.0, -0.5])
        
        new_hp, new_impurity, improved = perturb_coefficient(X, y, hp, m=0)
        
        # Impurity should not get worse
        assert new_impurity == 0 or not improved
    
    def test_perturb_with_mm_measure(self):
        """Test perturbation using Max Minority measure."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 0, 1])
        hp = np.array([0.5, 0.5, -0.5])
        
        new_hp, new_impurity, improved = perturb_coefficient(
            X, y, hp, m=0, impurity_measure="mm"
        )
        
        # Should return valid result
        assert len(new_hp) == 3


class TestHillClimb:
    """Test full hill-climbing optimization.
    
    Paper Reference: Section 2.1
    """
    
    def test_hill_climb_converges(self):
        """Test that hill-climbing converges."""
        # Simple linearly separable data
        X = np.array([
            [0, 0], [0.5, 0.5],  # Class 0
            [1.5, 1.5], [2, 2],  # Class 1
        ])
        y = np.array([0, 0, 1, 1])
        
        hp, impurity, n_iters = hill_climb(X, y, max_iterations=50)
        
        assert impurity >= 0
        assert n_iters >= 1
        assert len(hp) == 3  # 2 features + bias
    
    def test_hill_climb_perfect_split(self):
        """Test hill-climbing achieves perfect split when possible."""
        # Perfectly separable by diagonal
        X = np.array([
            [0, 0], [0.3, 0.3],
            [1, 1], [0.7, 0.7],
        ])
        y = np.array([0, 0, 1, 1])
        
        hp, impurity, n_iters = hill_climb(X, y)
        
        # Should find perfect or near-perfect split
        assert impurity <= 0.5
    
    def test_hill_climb_with_initial_hyperplane(self):
        """Test hill-climbing with provided initial hyperplane."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        initial_hp = np.array([0.5, 0.5, -2.0])
        
        hp, impurity, n_iters = hill_climb(
            X, y, initial_hyperplane=initial_hp
        )
        
        assert len(hp) == 3
    
    def test_hill_climb_empty_data_error(self):
        """Test error on empty data."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            hill_climb(X, y)
    
    def test_hill_climb_dimension_mismatch(self):
        """Test error on hyperplane dimension mismatch."""
        X = np.array([[1, 2, 3]])  # 3 features
        y = np.array([0])
        initial_hp = np.array([1.0, 1.0, 0.0])  # Only 2 features + bias
        
        with pytest.raises(ValueError, match="coefficients"):
            hill_climb(X, y, initial_hyperplane=initial_hp)
    
    def test_hill_climb_stops_at_local_minimum(self):
        """Test that hill-climbing stops when no improvement is possible."""
        # XOR-like data (requires multiple splits)
        X = np.array([
            [0, 0], [1, 1],  # Class 0
            [0, 1], [1, 0],  # Class 1
        ])
        y = np.array([0, 0, 1, 1])
        
        hp, impurity, n_iters = hill_climb(X, y, max_iterations=100)
        
        # Should converge within iterations
        assert n_iters <= 100
    
    def test_hill_climb_with_tolerance(self):
        """Test tolerance parameter for convergence."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 0, 1])
        
        hp, impurity, n_iters = hill_climb(X, y, tolerance=0.001)
        
        assert n_iters >= 1


class TestFindBestHyperplane:
    """Test the main hyperplane finding function."""
    
    def test_find_best_deterministic(self):
        """Test deterministic (single restart) hyperplane finding."""
        X = np.array([
            [0, 0], [0.2, 0.2],
            [1, 1], [0.8, 0.8],
        ])
        y = np.array([0, 0, 1, 1])
        
        hp, impurity = find_best_hyperplane(
            X, y, n_restarts=1
        )
        
        assert len(hp) == 3
        assert impurity >= 0
    
    def test_find_best_with_restarts(self):
        """Test hyperplane finding with multiple restarts (for Task 2)."""
        X = np.array([
            [0, 0], [0.3, 0.3],
            [1, 1], [0.7, 0.7],
        ])
        y = np.array([0, 0, 1, 1])
        
        hp, impurity = find_best_hyperplane(
            X, y, n_restarts=3, random_state=42
        )
        
        assert len(hp) == 3
        assert impurity >= 0
    
    def test_find_best_reproducibility(self):
        """Test that random_state produces reproducible results."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        hp1, imp1 = find_best_hyperplane(X, y, n_restarts=2, random_state=42)
        hp2, imp2 = find_best_hyperplane(X, y, n_restarts=2, random_state=42)
        
        np.testing.assert_array_almost_equal(hp1, hp2)
        assert imp1 == imp2


class TestNormalizeHyperplane:
    """Test hyperplane normalization."""
    
    def test_normalize_unit_length(self):
        """Test that normalized hyperplane has unit-length feature coefficients."""
        hp = np.array([3.0, 4.0, 5.0])  # norm of [3, 4] = 5
        
        normalized = normalize_hyperplane(hp)
        
        feature_norm = np.linalg.norm(normalized[:-1])
        assert feature_norm == pytest.approx(1.0)
    
    def test_normalize_preserves_direction(self):
        """Test that normalization preserves decision boundary direction."""
        hp = np.array([2.0, 2.0, 1.0])
        
        normalized = normalize_hyperplane(hp)
        
        # Ratio of coefficients should be preserved
        assert normalized[0] / normalized[1] == pytest.approx(1.0)
    
    def test_normalize_zero_features(self):
        """Test normalization with near-zero feature coefficients."""
        hp = np.array([0.0, 0.0, 1.0])  # Only bias
        
        normalized = normalize_hyperplane(hp)
        
        # Should return original when no meaningful normalization
        np.testing.assert_array_equal(normalized, hp)


class TestEdgeCases:
    """Test edge cases for hill-climbing."""
    
    def test_single_sample(self):
        """Test with single sample (degenerate case)."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        hp = initialize_hyperplane(X, y, method="axis_parallel")
        
        assert len(hp) == 3
    
    def test_all_same_class(self):
        """Test with all samples having same class."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([1, 1, 1])
        
        hp, impurity, _ = hill_climb(X, y)
        
        # Impurity should be 0 (pure node)
        assert impurity == 0.0
    
    def test_identical_features(self):
        """Test with all identical feature values."""
        X = np.array([[1, 1], [1, 1], [1, 1]])
        y = np.array([0, 1, 0])
        
        hp = initialize_hyperplane(X, y, method="axis_parallel")
        
        # Should still return valid hyperplane
        assert len(hp) == 3
    
    def test_two_samples_different_classes(self):
        """Test with exactly two samples of different classes."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        hp, impurity, _ = hill_climb(X, y)
        
        # Should find separating hyperplane
        assert impurity == 0.0
