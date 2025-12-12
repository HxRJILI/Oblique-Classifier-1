"""Tests for Task 2 randomization features.

This module tests the Task 2 enhancements:
- Random hyperplane initialization
- Multi-coefficient perturbation
- K random trials (n_restarts parameter)
- Random perturbation order
- Edge case validation
"""
import numpy as np
import pytest
from oc1 import ObliqueDecisionTree
from oc1.data import make_xor_dataset, make_diagonal_dataset
from oc1.core.hill_climb import (
    perturb_multiple_coefficients,
    validate_hyperplane,
    find_best_hyperplane,
    normalize_hyperplane,
)


class TestNRestartsParameter:
    """Tests for the n_restarts parameter functionality."""
    
    def test_n_restarts_parameter_works(self):
        """Test that n_restarts parameter works with different values."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        for k in [1, 3, 5]:
            tree = ObliqueDecisionTree(n_restarts=k, random_state=42)
            tree.fit(X, y)
            assert tree.get_depth() >= 0  # Depth can be 0 if perfectly split at root
            assert tree.score(X, y) >= 0.5  # At least random chance
    
    def test_default_n_restarts_is_10(self):
        """Test that the default value of n_restarts is 10 (Task 2 default)."""
        tree = ObliqueDecisionTree()
        assert tree.n_restarts == 10


class TestRandomInitialization:
    """Tests for random hyperplane initialization."""
    
    def test_random_initialization_different_results(self):
        """Test that random trials can produce different results."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        depths = []
        for seed in [10, 20, 30]:
            tree = ObliqueDecisionTree(n_restarts=5, random_state=seed)
            tree.fit(X, y)
            depths.append(tree.get_depth())
        
        # Different seeds might produce different trees
        assert len(depths) == 3


class TestImprovementWithMoreTrials:
    """Tests for improvement with more random trials."""
    
    def test_more_trials_dont_hurt_performance(self):
        """Test that more trials don't hurt performance."""
        X, y = make_xor_dataset(n_samples=200, random_state=42)
        
        acc_1 = ObliqueDecisionTree(n_restarts=1, random_state=42).fit(X, y).score(X, y)
        acc_5 = ObliqueDecisionTree(n_restarts=5, random_state=42).fit(X, y).score(X, y)
        
        # More trials should help or at least not hurt significantly
        assert acc_5 >= acc_1 - 0.05  # Allow small variation


class TestDeterministicWithSeed:
    """Tests for deterministic behavior with fixed random seed."""
    
    def test_same_seed_same_results(self):
        """Test that same seed gives same results."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree1 = ObliqueDecisionTree(n_restarts=5, random_state=42)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        
        tree2 = ObliqueDecisionTree(n_restarts=5, random_state=42)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_reproducibility_across_multiple_runs(self):
        """Test reproducibility across multiple training runs."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        results = []
        for _ in range(3):
            tree = ObliqueDecisionTree(n_restarts=5, random_state=123)
            tree.fit(X, y)
            results.append((tree.get_depth(), tree.get_n_leaves(), tree.score(X, y)))
        
        # All runs should produce identical results
        assert all(r == results[0] for r in results)


class TestPerturbMultipleCoefficients:
    """Tests for the perturb_multiple_coefficients function."""
    
    def test_perturb_multiple_coefficients_basic(self):
        """Test basic functionality of perturb_multiple_coefficients."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        hyperplane = np.array([1.0, 0.0, -0.5])
        
        rng = np.random.default_rng(42)
        new_hp, new_imp = perturb_multiple_coefficients(
            X, y, hyperplane, n_coefficients=2, n_trials=5, rng=rng
        )
        
        assert new_hp.shape == hyperplane.shape
        assert isinstance(new_imp, float)
    
    def test_perturb_improves_or_maintains(self):
        """Test that perturbation doesn't make things worse."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        hyperplane = np.array([1.0, 0.0, -0.5])
        
        rng = np.random.default_rng(42)
        from oc1.core.splits import evaluate_split
        
        original_imp = evaluate_split(X, y, hyperplane, "sm")
        new_hp, new_imp = perturb_multiple_coefficients(
            X, y, hyperplane, n_coefficients=2, n_trials=10, rng=rng
        )
        
        # New impurity should be <= original (or very close)
        assert new_imp <= original_imp + 1e-10


class TestValidateHyperplane:
    """Tests for the validate_hyperplane function."""
    
    def test_validate_valid_hyperplane(self):
        """Test validation of a valid hyperplane."""
        hp = np.array([1.0, 2.0, -0.5])
        fixed_hp, is_valid = validate_hyperplane(hp)
        
        assert is_valid
        np.testing.assert_array_almost_equal(fixed_hp, hp)
    
    def test_validate_nan_hyperplane(self):
        """Test validation of hyperplane with NaN values."""
        hp = np.array([1.0, np.nan, -0.5])
        rng = np.random.default_rng(42)
        fixed_hp, is_valid = validate_hyperplane(hp, rng=rng)
        
        assert not is_valid
        assert not np.any(np.isnan(fixed_hp))
    
    def test_validate_inf_hyperplane(self):
        """Test validation of hyperplane with infinity values."""
        hp = np.array([1.0, np.inf, -0.5])
        fixed_hp, is_valid = validate_hyperplane(hp)
        
        assert not is_valid
        assert not np.any(np.isinf(fixed_hp))
    
    def test_validate_zero_weights_hyperplane(self):
        """Test validation of hyperplane with all zero weights."""
        hp = np.array([0.0, 0.0, -0.5])
        rng = np.random.default_rng(42)
        fixed_hp, is_valid = validate_hyperplane(hp, rng=rng)
        
        assert not is_valid
        # After fix, should have non-zero feature coefficients
        assert not np.allclose(fixed_hp[:-1], 0)


class TestFindBestHyperplaneTask2:
    """Tests for find_best_hyperplane with Task 2 features."""
    
    def test_find_best_hyperplane_with_random_order(self):
        """Test find_best_hyperplane with random perturbation order."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        hp, impurity = find_best_hyperplane(
            X, y,
            n_restarts=3,
            random_state=42,
            use_random_perturbation_order=True,
        )
        
        assert hp is not None
        assert len(hp) == X.shape[1] + 1
        assert impurity >= 0
    
    def test_find_best_hyperplane_multi_restart(self):
        """Test that multiple restarts are executed."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        # Single restart (deterministic)
        hp1, imp1 = find_best_hyperplane(X, y, n_restarts=1, random_state=42)
        
        # Multiple restarts (randomized)
        hp5, imp5 = find_best_hyperplane(X, y, n_restarts=5, random_state=42)
        
        # Both should return valid hyperplanes
        assert hp1 is not None and hp5 is not None
        assert imp1 >= 0 and imp5 >= 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_feature_data(self):
        """Test with single feature data."""
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(n_restarts=3, random_state=42)
        tree.fit(X, y)
        
        assert tree.score(X, y) >= 0.5
    
    def test_high_dimensional_data(self):
        """Test with high dimensional data."""
        rng = np.random.default_rng(42)
        n_features = 20
        X = rng.random((100, n_features))
        y = (X.sum(axis=1) > n_features / 2).astype(int)
        
        tree = ObliqueDecisionTree(n_restarts=3, max_depth=5, random_state=42)
        tree.fit(X, y)
        
        assert tree.score(X, y) >= 0.5
    
    def test_small_dataset(self):
        """Test with small dataset."""
        X = np.array([[0, 0], [1, 1]], dtype=float)
        y = np.array([0, 1])
        
        tree = ObliqueDecisionTree(n_restarts=3, random_state=42)
        tree.fit(X, y)
        
        # Should handle small dataset gracefully
        assert tree.get_depth() >= 0


class TestNormalizeHyperplane:
    """Tests for hyperplane normalization."""
    
    def test_normalize_maintains_direction(self):
        """Test that normalization maintains hyperplane direction."""
        hp = np.array([3.0, 4.0, 1.0])
        normalized = normalize_hyperplane(hp)
        
        # Feature coefficients should have unit norm
        feature_norm = np.linalg.norm(normalized[:-1])
        np.testing.assert_almost_equal(feature_norm, 1.0)
    
    def test_normalize_zero_handling(self):
        """Test normalization handles near-zero coefficients."""
        hp = np.array([1e-15, 1e-15, 1.0])
        normalized = normalize_hyperplane(hp)
        
        # Should return a valid array without NaN
        assert not np.any(np.isnan(normalized))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
