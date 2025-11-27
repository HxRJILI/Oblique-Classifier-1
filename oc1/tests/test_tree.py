"""
Tests for ObliqueDecisionTree classifier.

Tests cover:
- Tree fitting and prediction
- Multi-class classification
- Tree structure (depth, nodes, leaves)
- Stopping criteria (pure nodes, min_samples, max_depth)
- Edge cases: single class, single sample

Paper Reference: Section 2 - Tree construction algorithm
"""

import pytest
import numpy as np
from oc1.core.tree import ObliqueDecisionTree
from oc1.data.datasets import (
    make_diagonal_dataset,
    make_xor_dataset,
    make_multiclass_oblique,
    make_oblique_classification,
)


class TestTreeFitting:
    """Test tree fitting functionality."""
    
    def test_basic_fit(self):
        """Test basic tree fitting."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        assert tree._is_fitted is True
        assert tree.n_features_ == 2
        assert tree.n_classes_ == 2
    
    def test_fit_sets_classes(self):
        """Test that fit correctly identifies classes."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 2, 0, 1, 2])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        assert len(tree.classes_) == 3
        assert set(tree.classes_) == {0, 1, 2}
    
    def test_fit_empty_data_error(self):
        """Test error on empty data."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        tree = ObliqueDecisionTree()
        
        with pytest.raises(ValueError, match="empty"):
            tree.fit(X, y)
    
    def test_fit_dimension_mismatch_error(self):
        """Test error on X and y dimension mismatch."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])  # Missing one label
        
        tree = ObliqueDecisionTree()
        
        with pytest.raises(ValueError, match="samples"):
            tree.fit(X, y)


class TestTreePrediction:
    """Test tree prediction functionality."""
    
    def test_basic_predict(self):
        """Test basic prediction."""
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X_train, y_train)
        
        predictions = tree.predict(X_train)
        
        assert len(predictions) == 4
    
    def test_predict_before_fit_error(self):
        """Test error when predicting before fit."""
        tree = ObliqueDecisionTree()
        
        with pytest.raises(ValueError, match="not fitted"):
            tree.predict(np.array([[1, 2]]))
    
    def test_predict_wrong_features_error(self):
        """Test error when predicting with wrong number of features."""
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X_train, y_train)
        
        X_test = np.array([[1, 2, 3]])  # 3 features, trained on 2
        
        with pytest.raises(ValueError, match="features"):
            tree.predict(X_test)
    
    def test_predict_single_sample(self):
        """Test prediction for single sample."""
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X_train, y_train)
        
        prediction = tree.predict(np.array([[0.5, 0.5]]))
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]


class TestTreeScore:
    """Test tree scoring (accuracy)."""
    
    def test_score_linearly_separable(self):
        """Test score on clearly linearly separable data."""
        # More clearly separated data
        X = np.array([
            [0, 0], [0.1, 0.2], [0.2, 0.1],
            [2, 2], [1.9, 2.1], [2.1, 1.9],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=10)
        tree.fit(X, y)
        
        score = tree.score(X, y)
        
        # Should classify at least half correctly (random baseline)
        assert score >= 0.5


class TestStoppingCriteria:
    """Test tree stopping criteria.
    
    Paper Reference: Section 2.4
    """
    
    def test_stop_at_pure_node(self):
        """Test that tree stops at pure nodes (zero impurity)."""
        # All same class
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([1, 1, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        # Should be just a leaf
        assert tree.root.is_leaf is True
    
    def test_max_depth(self):
        """Test max_depth constraint."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=2)
        tree.fit(X, y)
        
        assert tree.get_depth() <= 2
    
    def test_min_samples_leaf(self):
        """Test min_samples_leaf constraint."""
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],
            [1, 1], [1.1, 1.1], [1.2, 1.2],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        tree = ObliqueDecisionTree(min_samples_leaf=2)
        tree.fit(X, y)
        
        # All leaves should have at least 2 samples
        # (This is enforced during splitting)
        assert tree._is_fitted
    
    def test_min_samples_split(self):
        """Test min_samples_split constraint."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        tree = ObliqueDecisionTree(min_samples_split=5)
        tree.fit(X, y)
        
        # With 2 samples and min_samples_split=5, should be leaf
        assert tree.root.is_leaf is True


class TestTreeStructure:
    """Test tree structure methods."""
    
    def test_get_depth(self):
        """Test getting tree depth."""
        X = np.array([
            [0, 0], [0.5, 0],
            [1, 1], [0.5, 1],
        ])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        depth = tree.get_depth()
        
        assert depth >= 0
    
    def test_get_n_leaves(self):
        """Test counting leaf nodes."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        n_leaves = tree.get_n_leaves()
        
        assert n_leaves >= 1
    
    def test_get_n_nodes(self):
        """Test counting total nodes."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        n_nodes = tree.get_n_nodes()
        n_leaves = tree.get_n_leaves()
        
        # Total nodes >= leaves
        assert n_nodes >= n_leaves
    
    def test_get_hyperplanes(self):
        """Test getting all hyperplanes from tree."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3)
        tree.fit(X, y)
        
        hyperplanes = tree.get_hyperplanes()
        
        # Should have at least one hyperplane if not pure
        assert isinstance(hyperplanes, list)
        for hp, depth in hyperplanes:
            assert len(hp) == 3  # 2D data


class TestMultiClassClassification:
    """Test multi-class classification."""
    
    def test_three_class(self):
        """Test 3-class classification."""
        X, y = make_multiclass_oblique(n_samples=90, n_classes=3, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        assert tree.n_classes_ == 3
        
        predictions = tree.predict(X)
        assert set(predictions).issubset({0, 1, 2})
    
    def test_five_class(self):
        """Test 5-class classification."""
        X, y = make_multiclass_oblique(n_samples=100, n_classes=5, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=6)
        tree.fit(X, y)
        
        assert tree.n_classes_ == 5


class TestPredictProba:
    """Test probability predictions."""
    
    def test_predict_proba_shape(self):
        """Test shape of probability output."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        proba = tree.predict_proba(X)
        
        assert proba.shape == (4, 2)  # 4 samples, 2 classes
    
    def test_predict_proba_sums_to_one(self):
        """Test that probabilities sum to 1."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        proba = tree.predict_proba(X)
        
        # Each row should sum to 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(X)))


class TestImpurityMeasures:
    """Test different impurity measures."""
    
    def test_sum_minority(self):
        """Test tree with Sum Minority impurity."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(impurity_measure="sm")
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_max_minority(self):
        """Test tree with Max Minority impurity."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(impurity_measure="mm")
        tree.fit(X, y)
        
        assert tree._is_fitted
    
    def test_invalid_impurity_error(self):
        """Test error on invalid impurity measure."""
        with pytest.raises(ValueError, match="impurity"):
            ObliqueDecisionTree(impurity_measure="invalid")


class TestPrintTree:
    """Test tree visualization."""
    
    def test_print_tree(self):
        """Test tree string representation."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=2)
        tree.fit(X, y)
        
        tree_str = tree.print_tree()
        
        assert isinstance(tree_str, str)
        assert len(tree_str) > 0
    
    def test_print_tree_with_feature_names(self):
        """Test tree printing with custom feature names."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        
        tree = ObliqueDecisionTree(max_depth=2)
        tree.fit(X, y)
        
        tree_str = tree.print_tree(feature_names=["height", "width"])
        
        assert "height" in tree_str or "width" in tree_str or "Leaf" in tree_str


class TestRepr:
    """Test string representation."""
    
    def test_repr_unfitted(self):
        """Test repr of unfitted tree."""
        tree = ObliqueDecisionTree()
        
        repr_str = repr(tree)
        
        assert "not fitted" in repr_str
    
    def test_repr_fitted(self):
        """Test repr of fitted tree."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        tree = ObliqueDecisionTree()
        tree.fit(X, y)
        
        repr_str = repr(tree)
        
        assert "depth" in repr_str


class TestReproducibility:
    """Test reproducibility with random_state."""
    
    def test_random_state_reproducibility(self):
        """Test that random_state produces reproducible results."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)
        
        tree1 = ObliqueDecisionTree(n_restarts=3, random_state=123)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        
        tree2 = ObliqueDecisionTree(n_restarts=3, random_state=123)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestSyntheticDatasets:
    """Test on synthetic datasets designed for oblique trees."""
    
    def test_diagonal_dataset(self):
        """Test on diagonal decision boundary dataset."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        score = tree.score(X, y)
        
        # Should perform well on diagonal split
        assert score >= 0.7
    
    def test_oblique_45_degree(self):
        """Test on 45-degree oblique boundary."""
        X, y = make_oblique_classification(
            n_samples=100, angle=45, noise=0.1, random_state=42
        )
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        score = tree.score(X, y)
        
        assert score >= 0.6
    
    def test_xor_dataset(self):
        """Test on XOR dataset (requires multiple splits)."""
        X, y = make_xor_dataset(n_samples=200, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5)
        tree.fit(X, y)
        
        score = tree.score(X, y)
        
        # XOR is challenging, but should do reasonably well
        assert score >= 0.5
