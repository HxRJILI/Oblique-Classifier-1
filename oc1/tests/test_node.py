"""
Tests for ObliqueTreeNode class.

Tests cover:
- Node initialization and attributes
- Hyperplane evaluation (V_j calculation)
- Partition rule (V_j > 0 → left, V_j ≤ 0 → right)
- Majority class and minority count calculations
- Edge cases: empty nodes, pure nodes, single samples

Paper Reference: Section 2 - Node structure and hyperplane partitioning
"""

import pytest
import numpy as np
from oc1.core.node import ObliqueTreeNode


class TestObliqueTreeNodeInit:
    """Test node initialization."""
    
    def test_default_initialization(self):
        """Test node with default values."""
        node = ObliqueTreeNode()
        
        assert node.hyperplane is None
        assert node.class_distribution == {}
        assert node.left_child is None
        assert node.right_child is None
        assert node.is_leaf is False
        assert node.predicted_class is None
        assert node.depth == 0
        assert node.n_samples == 0
        assert node.impurity == 0.0
    
    def test_initialization_with_hyperplane(self):
        """Test node with hyperplane coefficients."""
        hyperplane = np.array([1.0, -1.0, 0.5])
        node = ObliqueTreeNode(hyperplane=hyperplane)
        
        assert node.hyperplane is not None
        np.testing.assert_array_equal(node.hyperplane, hyperplane)
    
    def test_initialization_with_class_distribution(self):
        """Test node with class distribution."""
        dist = {0: 10, 1: 5, 2: 3}
        node = ObliqueTreeNode(class_distribution=dist)
        
        assert node.class_distribution == dist
    
    def test_leaf_node_initialization(self):
        """Test leaf node initialization."""
        node = ObliqueTreeNode(
            is_leaf=True,
            predicted_class=1,
            n_samples=20,
            class_distribution={0: 5, 1: 15},
        )
        
        assert node.is_leaf is True
        assert node.predicted_class == 1
        assert node.n_samples == 20


class TestHyperplaneEvaluation:
    """Test hyperplane evaluation (V_j computation).
    
    Paper Reference: Section 2
    V_j = ∑(a_i * x_j^i) + a_{d+1}
    """
    
    def test_evaluate_simple_hyperplane(self):
        """Test evaluation of x - y = 0 hyperplane."""
        # Hyperplane: x - y = 0 → [1, -1, 0]
        node = ObliqueTreeNode(hyperplane=np.array([1.0, -1.0, 0.0]))
        
        X = np.array([
            [1, 0],   # V = 1 - 0 + 0 = 1 > 0 → left
            [0, 1],   # V = 0 - 1 + 0 = -1 ≤ 0 → right
            [1, 1],   # V = 1 - 1 + 0 = 0 ≤ 0 → right (tie goes right)
        ])
        
        V = node.evaluate(X)
        
        np.testing.assert_array_almost_equal(V, [1.0, -1.0, 0.0])
    
    def test_evaluate_with_bias(self):
        """Test evaluation of hyperplane with bias term."""
        # Hyperplane: x + y - 1 = 0 → [1, 1, -1]
        node = ObliqueTreeNode(hyperplane=np.array([1.0, 1.0, -1.0]))
        
        X = np.array([
            [0, 0],   # V = 0 + 0 - 1 = -1
            [1, 1],   # V = 1 + 1 - 1 = 1
            [0.5, 0.5],  # V = 0.5 + 0.5 - 1 = 0
        ])
        
        V = node.evaluate(X)
        
        np.testing.assert_array_almost_equal(V, [-1.0, 1.0, 0.0])
    
    def test_evaluate_single_sample(self):
        """Test evaluation of a single sample."""
        node = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, -0.5]))
        
        x = np.array([1.0, 0.0])
        V = node.evaluate(x)
        
        assert V.shape == (1,)
        assert V[0] == pytest.approx(0.5)
    
    def test_evaluate_leaf_raises_error(self):
        """Test that evaluating a leaf node raises error."""
        node = ObliqueTreeNode(is_leaf=True, predicted_class=0)
        
        with pytest.raises(ValueError, match="no hyperplane"):
            node.evaluate(np.array([[1, 2]]))
    
    def test_evaluate_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        # 2D hyperplane
        node = ObliqueTreeNode(hyperplane=np.array([1.0, 1.0, 0.0]))
        
        # 3D data
        X = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="features"):
            node.evaluate(X)


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_single_leaf(self):
        """Test prediction for leaf node."""
        node = ObliqueTreeNode(is_leaf=True, predicted_class=2)
        
        x = np.array([1.0, 2.0])
        prediction = node.predict_single(x)
        
        assert prediction == 2
    
    def test_predict_single_with_children(self):
        """Test prediction traversing to children."""
        # Root: x > 0.5 → left, else right
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, -0.5]))
        root.left_child = ObliqueTreeNode(is_leaf=True, predicted_class=1)
        root.right_child = ObliqueTreeNode(is_leaf=True, predicted_class=0)
        
        # x = 1.0: V = 1.0 - 0.5 = 0.5 > 0 → left → class 1
        assert root.predict_single(np.array([1.0, 0.0])) == 1
        
        # x = 0.0: V = 0.0 - 0.5 = -0.5 ≤ 0 → right → class 0
        assert root.predict_single(np.array([0.0, 0.0])) == 0
    
    def test_predict_deep_tree(self):
        """Test prediction in a deeper tree."""
        # Build a 2-level tree
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, -0.5]))
        
        left = ObliqueTreeNode(hyperplane=np.array([0.0, 1.0, -0.5]))
        left.left_child = ObliqueTreeNode(is_leaf=True, predicted_class="A")
        left.right_child = ObliqueTreeNode(is_leaf=True, predicted_class="B")
        
        right = ObliqueTreeNode(is_leaf=True, predicted_class="C")
        
        root.left_child = left
        root.right_child = right
        
        # (0.8, 0.8): x > 0.5 → left, y > 0.5 → left → "A"
        assert root.predict_single(np.array([0.8, 0.8])) == "A"
        
        # (0.8, 0.2): x > 0.5 → left, y ≤ 0.5 → right → "B"
        assert root.predict_single(np.array([0.8, 0.2])) == "B"
        
        # (0.2, 0.8): x ≤ 0.5 → right → "C"
        assert root.predict_single(np.array([0.2, 0.8])) == "C"


class TestClassDistribution:
    """Test class distribution related methods."""
    
    def test_get_majority_class(self):
        """Test getting majority class."""
        node = ObliqueTreeNode(class_distribution={0: 10, 1: 25, 2: 5})
        
        assert node.get_majority_class() == 1
    
    def test_get_majority_class_empty(self):
        """Test majority class for empty node."""
        node = ObliqueTreeNode(class_distribution={})
        
        assert node.get_majority_class() is None
    
    def test_get_minority_count(self):
        """Test getting minority count.
        
        Paper Reference: Section 2.4 - Minority definition
        """
        node = ObliqueTreeNode(class_distribution={0: 10, 1: 3, 2: 7})
        
        # Minority is min(10, 3, 7) = 3
        assert node.get_minority_count() == 3
    
    def test_get_minority_count_pure(self):
        """Test minority count for pure node (single class)."""
        node = ObliqueTreeNode(class_distribution={0: 20})
        
        # Pure node has minority = 0
        assert node.get_minority_count() == 0
    
    def test_get_minority_count_empty(self):
        """Test minority count for empty node."""
        node = ObliqueTreeNode(class_distribution={})
        
        assert node.get_minority_count() == 0
    
    def test_is_pure_single_class(self):
        """Test purity check for single-class node."""
        node = ObliqueTreeNode(class_distribution={1: 50})
        
        assert node.is_pure() is True
    
    def test_is_pure_multiple_classes(self):
        """Test purity check for multi-class node."""
        node = ObliqueTreeNode(class_distribution={0: 30, 1: 20})
        
        assert node.is_pure() is False
    
    def test_is_pure_empty(self):
        """Test purity check for empty node."""
        node = ObliqueTreeNode(class_distribution={})
        
        assert node.is_pure() is True


class TestNodeCopy:
    """Test node copying."""
    
    def test_copy_creates_independent_node(self):
        """Test that copy creates an independent node."""
        original = ObliqueTreeNode(
            hyperplane=np.array([1.0, 2.0, 3.0]),
            class_distribution={0: 5, 1: 10},
            depth=2,
            n_samples=15,
        )
        
        copied = original.copy()
        
        # Modify original
        original.hyperplane[0] = 999
        original.class_distribution[0] = 999
        
        # Copied should be unchanged
        assert copied.hyperplane[0] == 1.0
        assert copied.class_distribution[0] == 5


class TestTreeStatistics:
    """Test tree structure statistics."""
    
    def test_get_tree_depth_leaf(self):
        """Test depth of leaf node."""
        leaf = ObliqueTreeNode(is_leaf=True)
        
        assert leaf.get_tree_depth() == 0
    
    def test_get_tree_depth_balanced(self):
        """Test depth of balanced tree."""
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, 0.0]))
        root.left_child = ObliqueTreeNode(is_leaf=True)
        root.right_child = ObliqueTreeNode(is_leaf=True)
        
        assert root.get_tree_depth() == 1
    
    def test_get_tree_depth_unbalanced(self):
        """Test depth of unbalanced tree."""
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, 0.0]))
        left = ObliqueTreeNode(hyperplane=np.array([0.0, 1.0, 0.0]))
        left.left_child = ObliqueTreeNode(is_leaf=True)
        left.right_child = ObliqueTreeNode(is_leaf=True)
        root.left_child = left
        root.right_child = ObliqueTreeNode(is_leaf=True)
        
        assert root.get_tree_depth() == 2
    
    def test_count_nodes(self):
        """Test counting all nodes."""
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, 0.0]))
        root.left_child = ObliqueTreeNode(is_leaf=True)
        root.right_child = ObliqueTreeNode(is_leaf=True)
        
        assert root.count_nodes() == 3
    
    def test_count_leaves(self):
        """Test counting leaf nodes."""
        root = ObliqueTreeNode(hyperplane=np.array([1.0, 0.0, 0.0]))
        left = ObliqueTreeNode(hyperplane=np.array([0.0, 1.0, 0.0]))
        left.left_child = ObliqueTreeNode(is_leaf=True)
        left.right_child = ObliqueTreeNode(is_leaf=True)
        root.left_child = left
        root.right_child = ObliqueTreeNode(is_leaf=True)
        
        assert root.count_leaves() == 3


class TestNodeRepr:
    """Test string representations."""
    
    def test_repr_leaf(self):
        """Test string representation of leaf node."""
        node = ObliqueTreeNode(
            is_leaf=True,
            predicted_class=1,
            n_samples=10,
            depth=2,
        )
        
        repr_str = repr(node)
        
        assert "leaf" in repr_str
        assert "class=1" in repr_str
    
    def test_repr_internal_node(self):
        """Test string representation of internal node."""
        node = ObliqueTreeNode(
            hyperplane=np.array([1.0, -0.5, 0.25]),
            n_samples=50,
            depth=1,
        )
        
        repr_str = repr(node)
        
        assert "split" in repr_str
        assert "hyperplane" in repr_str
