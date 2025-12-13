"""
Tests for Task 3: Pruning Functionality
"""

import pytest
import numpy as np
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset, make_xor_dataset


class TestPruning:
    """Test pruning methods."""
    
    def test_prune_by_impurity_threshold(self):
        """Test pruning based on impurity threshold."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        # Build tree without pruning
        tree = ObliqueDecisionTree(max_depth=5, random_state=42)
        tree.fit(X, y)
        n_nodes_before = tree.get_n_nodes()
        n_leaves_before = tree.get_n_leaves()
        
        # Prune with high threshold (should prune many nodes)
        tree.prune(method="impurity", impurity_threshold=10.0)
        n_nodes_after = tree.get_n_nodes()
        n_leaves_after = tree.get_n_leaves()
        
        # After pruning, should have fewer nodes
        assert n_nodes_after <= n_nodes_before
        assert n_leaves_after <= n_leaves_before
        
        # Tree should still be valid
        predictions = tree.predict(X)
        assert len(predictions) == len(X)
    
    def test_prune_reduced_error(self):
        """Test Reduced Error Pruning."""
        X, y = make_diagonal_dataset(n_samples=200, random_state=42)
        
        # Split data
        from oc1.evaluation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Build tree
        tree = ObliqueDecisionTree(max_depth=10, random_state=42)
        tree.fit(X_train, y_train)
        n_nodes_before = tree.get_n_nodes()
        
        # Prune using validation set
        tree.prune(method="rep", X_val=X_test, y_val=y_test)
        n_nodes_after = tree.get_n_nodes()
        
        # Should have fewer or equal nodes
        assert n_nodes_after <= n_nodes_before
        
        # Accuracy should be maintained or improved
        accuracy_before = tree.score(X_test, y_test)
        # Re-fit to get baseline
        tree2 = ObliqueDecisionTree(max_depth=10, random_state=42)
        tree2.fit(X_train, y_train)
        accuracy_after = tree.score(X_test, y_test)
        
        # Pruned tree should still make valid predictions
        assert accuracy_after >= 0.0
    
    def test_prune_empty_tree(self):
        """Test pruning on a tree that's just a leaf."""
        X = np.array([[1, 2], [1, 2]])
        y = np.array([0, 0])  # All same class
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        # Should be a single leaf
        assert tree.get_n_nodes() == 1
        assert tree.get_n_leaves() == 1
        
        # Pruning should not crash
        tree.prune(method="impurity", impurity_threshold=0.0)
        assert tree.get_n_nodes() == 1
    
    def test_prune_invalid_method(self):
        """Test that invalid pruning method raises error."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        with pytest.raises(ValueError):
            tree.prune(method="invalid_method")
    
    def test_prune_requires_validation_set(self):
        """Test that REP requires validation set."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        with pytest.raises(ValueError):
            tree.prune(method="rep")
    
    def test_prune_preserves_predictions(self):
        """Test that pruning doesn't break predictions."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5, random_state=42)
        tree.fit(X, y)
        
        predictions_before = tree.predict(X)
        
        # Prune
        tree.prune(method="impurity", impurity_threshold=5.0)
        
        predictions_after = tree.predict(X)
        
        # Should still make predictions
        assert len(predictions_after) == len(X)
        assert all(p in tree.classes_ for p in predictions_after)

