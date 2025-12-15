"""
Task 2 and Task 3 Compatibility Tests

These tests verify that Task 1 implementation provides all necessary
interfaces and hooks for Task 2 (Randomization) and Task 3 (Pruning).

This ensures a smooth handoff to team members implementing subsequent tasks.
"""

import pytest
import numpy as np

from oc1.core.tree import ObliqueDecisionTree
from oc1.core.node import ObliqueTreeNode
from oc1.core.hill_climb import (
    find_best_hyperplane,
    initialize_hyperplane,
    hill_climb,
    perturb_random_direction,
    normalize_hyperplane,
    compute_u_values,
)
from oc1.core.splits import (
    evaluate_split,
    is_pure,
    get_majority_class,
    partition_data,
)
from oc1.data.datasets import make_diagonal_dataset, make_xor_dataset


class TestTask2RandomizationInterfaces:
    """Verify all Task 2 randomization interfaces are available and working."""
    
    def test_random_initialization_produces_varied_hyperplanes(self):
        """Random initialization produces different hyperplanes with different seeds."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        hp1 = initialize_hyperplane(X, y, method="random", random_state=1)
        hp2 = initialize_hyperplane(X, y, method="random", random_state=2)
        hp3 = initialize_hyperplane(X, y, method="random", random_state=1)
        
        # Different seeds should produce different hyperplanes
        assert not np.allclose(hp1, hp2), "Different seeds should give different hyperplanes"
        
        # Same seed should produce same hyperplane
        np.testing.assert_array_almost_equal(hp1, hp3)
    
    def test_n_restarts_parameter_available(self):
        """Tree accepts n_restarts parameter for multiple random restarts."""
        tree = ObliqueDecisionTree(n_restarts=5)
        assert tree.n_restarts == 5
        
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree.fit(X, y)
        
        # Should complete without error
        assert tree._is_fitted
    
    def test_n_restarts_improves_or_maintains_quality(self):
        """More restarts should not worsen results."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        _, imp1 = find_best_hyperplane(X, y, n_restarts=1, random_state=42)
        _, imp3 = find_best_hyperplane(X, y, n_restarts=3, random_state=42)
        
        # More restarts should find equal or better solution
        assert imp3 <= imp1 + 1e-10, "More restarts should not increase impurity"
    
    def test_perturb_random_direction_interface(self):
        """Verify multi-coefficient perturbation function is available."""
        X = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
        y = np.array([0, 1, 0, 1])
        hp = np.array([1.0, -1.0, 0.0])
        rng = np.random.default_rng(42)
        
        new_hp, new_imp = perturb_random_direction(X, y, hp, rng=rng)
        
        assert len(new_hp) == 3, "Hyperplane should have correct dimensions"
        assert new_imp >= 0, "Impurity should be non-negative"
        assert isinstance(new_hp, np.ndarray)
    
    def test_hill_climb_accepts_perturbation_order(self):
        """Hill climb accepts custom perturbation order for Task 2."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        initial_hp = initialize_hyperplane(X, y, method="axis_parallel")
        
        # Custom perturbation order (reversed)
        custom_order = [2, 1, 0]  # For 2D features + bias
        
        hp, imp, iters = hill_climb(
            X, y,
            initial_hyperplane=initial_hp,
            perturbation_order=custom_order,
        )
        
        assert len(hp) == 3
        assert imp >= 0
    
    def test_hill_climb_accepts_rng(self):
        """Hill climb accepts RNG for reproducible randomization."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        initial_hp = initialize_hyperplane(X, y, method="axis_parallel")
        rng = np.random.default_rng(123)
        
        hp, imp, iters = hill_climb(
            X, y,
            initial_hyperplane=initial_hp,
            rng=rng,
        )
        
        assert len(hp) == 3
        assert imp >= 0
    
    def test_find_best_hyperplane_random_perturbation_order(self):
        """find_best_hyperplane supports randomized perturbation order."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        hp, imp = find_best_hyperplane(
            X, y,
            use_random_perturbation_order=True,
            random_state=42,
        )
        
        assert len(hp) == 3
        assert imp >= 0
    
    def test_normalize_hyperplane_available(self):
        """Normalize hyperplane function is exported and working."""
        hp = np.array([3.0, 4.0, 5.0])
        normalized = normalize_hyperplane(hp)
        
        # Feature coefficients should have unit norm
        feature_norm = np.linalg.norm(normalized[:-1])
        assert np.isclose(feature_norm, 1.0)
    
    def test_compute_u_values_available(self):
        """U-value computation for coefficient perturbation is available."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        hp = np.array([1.0, 1.0, -3.0])
        
        U, valid_mask = compute_u_values(X, hp, m=0)
        
        assert len(U) == 2
        assert len(valid_mask) == 2


class TestTask3PruningInterfaces:
    """Verify all Task 3 pruning interfaces are available and working."""
    
    def test_node_has_parent_reference(self):
        """Nodes have parent references for pruning traversal."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        # Root has no parent
        assert tree.root.parent is None
        
        # Children have parent references
        if tree.root.left_child:
            assert tree.root.left_child.parent is tree.root
        if tree.root.right_child:
            assert tree.root.right_child.parent is tree.root
    
    def test_deep_parent_references(self):
        """Parent references are correct at all levels."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=4)
        tree.fit(X, y)
        
        # Traverse and verify all parent references
        def verify_parents(node, expected_parent):
            assert node.parent is expected_parent
            if node.left_child:
                verify_parents(node.left_child, node)
            if node.right_child:
                verify_parents(node.right_child, node)
        
        verify_parents(tree.root, None)
    
    def test_prune_placeholder_exists(self):
        """Prune method exists and can be called with validation data (Task 3 implemented)."""
        tree = ObliqueDecisionTree()
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree.fit(X, y)
        
        # Task 3 is now implemented - prune() works with validation data
        # Test that it accepts method parameter and validation data
        assert hasattr(tree, 'prune')
        
        # Test impurity method (doesn't require validation data)
        tree.prune(method='impurity', impurity_threshold=0.5)
        
        # Test that REP method requires validation data
        with pytest.raises(ValueError) as exc_info:
            tree.prune(method='rep')
        assert "X_val" in str(exc_info.value) or "required" in str(exc_info.value).lower()
    
    def test_nodes_store_all_metadata(self):
        """Nodes store all metadata needed for pruning decisions."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        # Verify root has all required attributes
        assert hasattr(tree.root, 'impurity')
        assert hasattr(tree.root, 'class_distribution')
        assert hasattr(tree.root, 'n_samples')
        assert hasattr(tree.root, 'parent')
        assert hasattr(tree.root, 'depth')
        assert hasattr(tree.root, 'is_leaf')
        assert hasattr(tree.root, 'predicted_class')
        
        # Verify values are set correctly
        assert tree.root.n_samples == 50
        assert tree.root.depth == 0
        assert len(tree.root.class_distribution) > 0
    
    def test_get_all_nodes_available(self):
        """get_all_nodes method returns all tree nodes."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        nodes = tree.get_all_nodes()
        
        # Should return correct number of nodes
        assert len(nodes) == tree.get_n_nodes()
        
        # First node should be root
        assert nodes[0] is tree.root
        
        # All nodes should be ObliqueTreeNode instances
        for node in nodes:
            assert isinstance(node, ObliqueTreeNode)
    
    def test_get_all_nodes_breadth_first_order(self):
        """get_all_nodes returns nodes in breadth-first order."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        nodes = tree.get_all_nodes()
        
        # Verify BFS order: depth should be non-decreasing
        for i in range(1, len(nodes)):
            assert nodes[i].depth >= nodes[i-1].depth or nodes[i].depth == nodes[i-1].depth
    
    def test_impurity_threshold_parameter(self):
        """Tree accepts impurity_threshold for pre-pruning."""
        tree = ObliqueDecisionTree(impurity_threshold=0.1)
        assert tree.impurity_threshold == 0.1
        
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree.fit(X, y)
        
        # Should complete without error
        assert tree._is_fitted
    
    def test_stopping_criteria_all_available(self):
        """All stopping criteria parameters are available."""
        tree = ObliqueDecisionTree(
            max_depth=5,
            min_samples_leaf=3,
            min_samples_split=10,
            impurity_threshold=0.05,
        )
        
        assert tree.max_depth == 5
        assert tree.min_samples_leaf == 3
        assert tree.min_samples_split == 10
        assert tree.impurity_threshold == 0.05


class TestSplitFunctionsExported:
    """Verify all necessary split functions are exported."""
    
    def test_evaluate_split_exported(self):
        """evaluate_split function is exported and working."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 0, 1, 1])
        hp = np.array([0.0, 1.0, -0.5])  # Split on y = 0.5
        
        impurity = evaluate_split(X, y, hp, impurity_measure="sm")
        
        assert impurity == 0  # Perfect split
    
    def test_is_pure_exported(self):
        """is_pure function is exported and working."""
        pure = np.array([0, 0, 0])
        not_pure = np.array([0, 1, 0])
        
        assert is_pure(pure) == True
        assert is_pure(not_pure) == False
    
    def test_get_majority_class_exported(self):
        """get_majority_class function is exported and working."""
        y = np.array([0, 0, 1, 0, 1])
        
        majority = get_majority_class(y)
        
        assert majority == 0


class TestAPIStability:
    """Verify API is stable and complete for Task 2/3 integration."""
    
    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree()
        
        result = tree.fit(X, y)
        
        assert result is tree
    
    def test_all_public_methods_available(self):
        """All public methods are available on ObliqueDecisionTree."""
        tree = ObliqueDecisionTree()
        
        # Training methods
        assert hasattr(tree, 'fit')
        assert hasattr(tree, 'predict')
        assert hasattr(tree, 'predict_proba')
        assert hasattr(tree, 'score')
        
        # Inspection methods
        assert hasattr(tree, 'get_depth')
        assert hasattr(tree, 'get_n_leaves')
        assert hasattr(tree, 'get_n_nodes')
        assert hasattr(tree, 'get_all_nodes')
        assert hasattr(tree, 'get_hyperplanes')
        assert hasattr(tree, 'print_tree')
        
        # Task 3 placeholder
        assert hasattr(tree, 'prune')
    
    def test_tree_stores_training_metadata(self):
        """Tree stores training metadata needed for Task 2/3."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        assert tree.n_features_ == 2
        assert len(tree.classes_) == 2
        assert tree.n_classes_ == 2
        assert tree._is_fitted == True


class TestEdgeCasesForTask2And3:
    """Edge cases that Task 2 and Task 3 implementations must handle."""
    
    def test_single_class_data(self):
        """Tree handles single-class data (for pruning edge cases)."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 0])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        # Should create a single leaf
        assert tree.root.is_leaf
        assert tree.get_n_leaves() == 1
    
    def test_two_samples_different_classes(self):
        """Minimum viable split scenario."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        tree = ObliqueDecisionTree(min_samples_leaf=1)
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_hyperplane_at_all_internal_nodes(self):
        """All internal nodes have valid hyperplanes."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=4)
        tree.fit(X, y)
        
        for node in tree.get_all_nodes():
            if not node.is_leaf:
                assert node.hyperplane is not None
                assert len(node.hyperplane) == tree.n_features_ + 1
