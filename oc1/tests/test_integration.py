"""
Integration tests for OC1 implementation.

These tests verify that all components work together correctly
and that the implementation matches the OC1 paper specifications.

Paper Reference: Murthy et al., "OC1: A randomized algorithm for
building oblique decision trees", AAAI-1992
"""

import pytest
import numpy as np
from oc1.core.tree import ObliqueDecisionTree
from oc1.core.node import ObliqueTreeNode
from oc1.core.splits import (
    partition_data,
    calculate_impurity,
    compute_class_counts,
    evaluate_hyperplane,
)
from oc1.core.hill_climb import (
    hill_climb,
    perturb_coefficient,
    compute_u_values,
)
from oc1.data.datasets import (
    make_diagonal_dataset,
    make_xor_dataset,
    make_multiclass_oblique,
    make_oblique_classification,
    make_3d_oblique,
    get_test_datasets,
)


class TestPaperFidelity:
    """Tests to verify implementation matches OC1 paper specifications."""
    
    def test_hyperplane_partition_rule(self):
        """Test partition rule: V > 0 → left, V ≤ 0 → right (Section 2)."""
        X = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        y = np.array([0, 1, 0])
        
        # Hyperplane: x - y = 0 → points where x > y go left
        hyperplane = np.array([1.0, -1.0, 0.0])
        
        X_left, y_left, X_right, y_right, V = partition_data(X, y, hyperplane)
        
        # (1, 0): V = 1 - 0 = 1 > 0 → left
        assert any(np.allclose(x, [1, 0]) for x in X_left)
        
        # (0, 1): V = 0 - 1 = -1 ≤ 0 → right
        assert any(np.allclose(x, [0, 1]) for x in X_right)
        
        # (0.5, 0.5): V = 0.5 - 0.5 = 0 ≤ 0 → right (tie goes right!)
        assert any(np.allclose(x, [0.5, 0.5]) for x in X_right)
    
    def test_sum_minority_formula(self):
        """Test SM = minority_L + minority_R (Section 2.4)."""
        # Left partition: {0: 8, 1: 2} → minority = 2
        # Right partition: {0: 3, 1: 7} → minority = 3
        left_counts = {0: 8, 1: 2}
        right_counts = {0: 3, 1: 7}
        
        sm, mm = calculate_impurity(left_counts, right_counts)
        
        # SM = 2 + 3 = 5
        assert sm == 5.0
    
    def test_max_minority_formula(self):
        """Test MM = max(minority_L, minority_R) (Section 2.4)."""
        left_counts = {0: 8, 1: 2}  # minority = 2
        right_counts = {0: 3, 1: 7}  # minority = 3
        
        sm, mm = calculate_impurity(left_counts, right_counts)
        
        # MM = max(2, 3) = 3
        assert mm == 3.0
    
    def test_equation_1_u_values(self):
        """Test U_j computation from Equation 1 (Section 2.2).
        
        U_j = a_m * x_j^m - V_j / x_j^m
        """
        # Simple test case
        X = np.array([[2.0, 1.0]])
        hyperplane = np.array([1.0, 1.0, 0.0])  # x + y = 0
        
        # V_j = 1*2 + 1*1 + 0 = 3
        V = evaluate_hyperplane(X, hyperplane)
        assert V[0] == pytest.approx(3.0)
        
        # For m=0 (first coefficient):
        # U_j = a_0 * x_j^0 - V_j / x_j^0
        # U_j = 1.0 * 2.0 - 3.0 / 2.0 = 2.0 - 1.5 = 0.5
        U, valid_mask = compute_u_values(X, hyperplane, m=0)
        assert U[0] == pytest.approx(0.5)
    
    def test_hill_climb_sequential_perturbation(self):
        """Test sequential coefficient perturbation (Section 2.1)."""
        X = np.array([
            [0, 0], [0.2, 0.1],
            [1, 1], [0.8, 0.9],
        ])
        y = np.array([0, 0, 1, 1])
        
        # Start with suboptimal hyperplane
        initial_hp = np.array([1.0, 0.0, -0.5])
        
        # Run hill-climbing
        final_hp, final_impurity, n_iters = hill_climb(
            X, y, initial_hyperplane=initial_hp
        )
        
        # Should converge
        assert n_iters >= 1
        # Final impurity should be reasonable
        assert final_impurity <= 2  # Original impurity might be higher
    
    def test_pure_node_has_zero_impurity(self):
        """Test that pure nodes have impurity = 0 (Section 2.4)."""
        # All same class
        left_counts = {0: 10}
        right_counts = {1: 10}
        
        sm, mm = calculate_impurity(left_counts, right_counts)
        
        # Both partitions are pure
        assert sm == 0.0
        assert mm == 0.0
    
    def test_multiclass_minority_calculation(self):
        """Test minority calculation with 3+ classes."""
        # 3-class partition
        # Left: {A: 10, B: 3, C: 5} → minority = 3
        # Right: {A: 2, B: 8, C: 4} → minority = 2
        left_counts = {'A': 10, 'B': 3, 'C': 5}
        right_counts = {'A': 2, 'B': 8, 'C': 4}
        
        sm, mm = calculate_impurity(left_counts, right_counts)
        
        assert sm == 5.0  # 3 + 2
        assert mm == 3.0  # max(3, 2)


class TestFullPipeline:
    """Test complete tree construction pipeline."""
    
    def test_fit_predict_pipeline(self):
        """Test full fit-predict pipeline."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5, random_state=42)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert tree.score(X, y) >= 0.6
    
    def test_tree_structure_valid(self):
        """Test that built tree has valid structure."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        # Check root is valid
        assert tree.root is not None
        
        # Check structure
        n_nodes = tree.get_n_nodes()
        n_leaves = tree.get_n_leaves()
        depth = tree.get_depth()
        
        assert n_nodes >= 1
        assert n_leaves >= 1
        assert depth <= 3
        
        # Internal nodes = total - leaves
        n_internal = n_nodes - n_leaves
        assert n_internal >= 0
    
    def test_hyperplane_dimensions(self):
        """Test that hyperplanes have correct dimensions."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        hyperplanes = tree.get_hyperplanes()
        
        for hp, depth in hyperplanes:
            # 2D data → 3 coefficients (2 features + bias)
            assert len(hp) == 3


class TestAllDatasets:
    """Test on all synthetic datasets."""
    
    def test_all_datasets_run(self):
        """Test that tree can be fit on all datasets."""
        datasets = get_test_datasets()
        
        for name, (X, y) in datasets.items():
            tree = ObliqueDecisionTree(max_depth=5, random_state=42)
            tree.fit(X, y)
            
            predictions = tree.predict(X)
            score = tree.score(X, y)
            
            assert len(predictions) == len(y), f"Failed on {name}"
            assert score >= 0, f"Invalid score on {name}"
    
    def test_3d_dataset(self):
        """Test on 3D dataset."""
        X, y = make_3d_oblique(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        # Should have 4-coefficient hyperplanes (3 features + bias)
        hyperplanes = tree.get_hyperplanes()
        if hyperplanes:
            assert len(hyperplanes[0][0]) == 4
        
        score = tree.score(X, y)
        assert score >= 0.6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_sample_per_class(self):
        """Test with minimum samples per class."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        # Should still work
        predictions = tree.predict(X)
        assert len(predictions) == 2
    
    def test_highly_imbalanced_classes(self):
        """Test with imbalanced class distribution."""
        X = np.vstack([
            np.random.randn(90, 2) - 1,  # Class 0: 90 samples
            np.random.randn(10, 2) + 1,  # Class 1: 10 samples
        ])
        y = np.array([0] * 90 + [1] * 10)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_many_features(self):
        """Test with higher dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)  # 10 features
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        assert tree.n_features_ == 10
        
        hyperplanes = tree.get_hyperplanes()
        if hyperplanes:
            assert len(hyperplanes[0][0]) == 11  # 10 features + bias
    
    def test_degenerate_all_same_features(self):
        """Test with all identical feature values."""
        X = np.ones((10, 2))  # All same
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        # Should still produce valid tree (likely just a leaf)
        assert tree._is_fitted


class TestCompatibilityForFutureTasks:
    """Test features needed for Task 2 (randomization) and Task 3 (pruning)."""
    
    def test_n_restarts_parameter(self):
        """Test n_restarts parameter for Task 2 randomization."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        # Should support multiple restarts
        tree = ObliqueDecisionTree(n_restarts=3, random_state=42)
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_impurity_threshold_parameter(self):
        """Test impurity_threshold for Task 3 pruning integration."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(impurity_threshold=0.1)
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_node_impurity_stored(self):
        """Test that nodes store impurity values for pruning."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        # Root node should have impurity stored
        assert hasattr(tree.root, 'impurity')
    
    def test_class_distribution_stored(self):
        """Test that nodes store class distributions."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        # Check root has class distribution
        assert hasattr(tree.root, 'class_distribution')
        assert tree.root.class_distribution is not None
    
    def test_find_best_hyperplane_accessible(self):
        """Test that hyperplane finding is modular (for Task 2)."""
        from oc1.core.hill_climb import find_best_hyperplane
        
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        hp, impurity = find_best_hyperplane(X, y, n_restarts=1)
        
        assert len(hp) == 3  # 2D data
        assert impurity >= 0


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_small_values(self):
        """Test with very small feature values."""
        X = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10]])
        y = np.array([0, 1, 0])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_large_values(self):
        """Test with large feature values."""
        X = np.array([[1e10, 2e10], [3e10, 4e10], [5e10, 6e10]])
        y = np.array([0, 1, 0])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_mixed_scale_features(self):
        """Test with features on different scales."""
        X = np.array([
            [1e-5, 1e5],
            [2e-5, 2e5],
            [3e-5, 3e5],
            [4e-5, 4e5],
        ])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        assert tree._is_fitted
