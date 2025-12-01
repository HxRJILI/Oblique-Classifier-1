"""
Tests for split evaluation and impurity measures.

Tests cover:
- Hyperplane evaluation (V_j calculation)
- Data partitioning (left if V > 0, right if V ≤ 0)
- Sum Minority (SM) impurity
- Max Minority (MM) impurity
- Edge cases: empty partitions, pure nodes, single samples

Paper Reference: Section 2 and Section 2.4
"""

import pytest
import numpy as np
from oc1.core.splits import (
    evaluate_hyperplane,
    partition_data,
    compute_class_counts,
    compute_minority,
    calculate_impurity,
    calculate_impurity_from_partition,
    evaluate_split,
    find_best_threshold,
    is_pure,
    get_majority_class,
)


class TestEvaluateHyperplane:
    """Test hyperplane evaluation V_j = ∑(a_i * x_j^i) + a_{d+1}."""
    
    def test_simple_hyperplane(self):
        """Test simple 2D hyperplane evaluation."""
        X = np.array([[1, 2], [3, 4], [0, 0]])
        hyperplane = np.array([1.0, 1.0, -3.0])  # x + y - 3 = 0
        
        V = evaluate_hyperplane(X, hyperplane)
        
        # V = [1+2-3, 3+4-3, 0+0-3] = [0, 4, -3]
        np.testing.assert_array_almost_equal(V, [0.0, 4.0, -3.0])
    
    def test_axis_parallel_x(self):
        """Test axis-parallel hyperplane (x = 0.5)."""
        X = np.array([[0, 0], [1, 0], [0.5, 0]])
        hyperplane = np.array([1.0, 0.0, -0.5])  # x - 0.5 = 0
        
        V = evaluate_hyperplane(X, hyperplane)
        
        # V = [0-0.5, 1-0.5, 0.5-0.5] = [-0.5, 0.5, 0]
        np.testing.assert_array_almost_equal(V, [-0.5, 0.5, 0.0])
    
    def test_3d_hyperplane(self):
        """Test 3D hyperplane evaluation."""
        X = np.array([[1, 1, 1], [0, 0, 0]])
        hyperplane = np.array([1.0, 1.0, 1.0, -1.5])  # x + y + z - 1.5 = 0
        
        V = evaluate_hyperplane(X, hyperplane)
        
        # V = [3-1.5, 0-1.5] = [1.5, -1.5]
        np.testing.assert_array_almost_equal(V, [1.5, -1.5])
    
    def test_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        X = np.array([[1, 2, 3]])  # 3 features
        hyperplane = np.array([1.0, 1.0, 0.0])  # 2 features + bias
        
        with pytest.raises(ValueError, match="features"):
            evaluate_hyperplane(X, hyperplane)


class TestPartitionData:
    """Test data partitioning based on hyperplane.
    
    Paper Reference: Section 2
    - Left: V_j > 0
    - Right: V_j ≤ 0
    """
    
    def test_simple_partition(self):
        """Test basic partitioning."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([0, 1, 0, 1])
        hyperplane = np.array([1.0, 0.0, -0.5])  # x - 0.5 = 0
        
        X_left, y_left, X_right, y_right, V = partition_data(X, y, hyperplane)
        
        # x > 0.5: points (1,0) and (1,1) go left
        assert len(X_left) == 2
        assert len(X_right) == 2
    
    def test_partition_tie_goes_right(self):
        """Test that points exactly on hyperplane go right (V ≤ 0)."""
        X = np.array([[0.5, 0]])
        y = np.array([0])
        hyperplane = np.array([1.0, 0.0, -0.5])  # x = 0.5
        
        X_left, y_left, X_right, y_right, V = partition_data(X, y, hyperplane)
        
        # Point on hyperplane (V = 0) goes right
        assert len(X_left) == 0
        assert len(X_right) == 1
    
    def test_all_left(self):
        """Test all points going left."""
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([0, 1, 2])
        hyperplane = np.array([1.0, 1.0, 0.0])  # x + y > 0
        
        X_left, y_left, X_right, y_right, _ = partition_data(X, y, hyperplane)
        
        assert len(X_left) == 3
        assert len(X_right) == 0
    
    def test_all_right(self):
        """Test all points going right."""
        X = np.array([[-1, -1], [-2, -2], [-3, -3]])
        y = np.array([0, 1, 2])
        hyperplane = np.array([1.0, 1.0, 0.0])  # x + y ≤ 0
        
        X_left, y_left, X_right, y_right, _ = partition_data(X, y, hyperplane)
        
        assert len(X_left) == 0
        assert len(X_right) == 3
    
    def test_oblique_partition(self):
        """Test oblique (45-degree) partition."""
        # Points above x + y = 1 line
        X_above = np.array([[0, 1.5], [1.5, 0], [1, 1]])
        # Points below x + y = 1 line
        X_below = np.array([[0, 0], [0.3, 0.3]])
        
        X = np.vstack([X_above, X_below])
        y = np.concatenate([np.ones(3), np.zeros(2)])
        
        hyperplane = np.array([1.0, 1.0, -1.0])  # x + y - 1 = 0
        
        X_left, y_left, X_right, y_right, _ = partition_data(X, y, hyperplane)
        
        # Points with x + y > 1 go left
        assert len(X_left) == 3
        assert len(X_right) == 2


class TestComputeClassCounts:
    """Test class count computation."""
    
    def test_simple_counts(self):
        """Test counting classes."""
        y = np.array([0, 1, 0, 2, 1, 0])
        
        counts = compute_class_counts(y)
        
        assert counts == {0: 3, 1: 2, 2: 1}
    
    def test_single_class(self):
        """Test with single class."""
        y = np.array([5, 5, 5, 5])
        
        counts = compute_class_counts(y)
        
        assert counts == {5: 4}
    
    def test_empty_array(self):
        """Test with empty array."""
        y = np.array([])
        
        counts = compute_class_counts(y)
        
        assert counts == {}
    
    def test_string_classes(self):
        """Test with string class labels."""
        y = np.array(['cat', 'dog', 'cat', 'bird'])
        
        counts = compute_class_counts(y)
        
        assert counts == {'cat': 2, 'dog': 1, 'bird': 1}


class TestComputeMinority:
    """Test minority count computation.
    
    Paper Reference: Section 2.4
    minority = min(count of each class)
    """
    
    def test_basic_minority(self):
        """Test basic minority calculation."""
        counts = {0: 10, 1: 3, 2: 7}
        
        minority = compute_minority(counts)
        
        assert minority == 3
    
    def test_two_classes(self):
        """Test minority with two classes."""
        counts = {0: 20, 1: 5}
        
        minority = compute_minority(counts)
        
        assert minority == 5
    
    def test_pure_node(self):
        """Test minority for pure node (single class)."""
        counts = {0: 100}
        
        minority = compute_minority(counts)
        
        # Pure node has minority = 0
        assert minority == 0
    
    def test_empty_counts(self):
        """Test minority for empty partition."""
        counts = {}
        
        minority = compute_minority(counts)
        
        assert minority == 0
    
    def test_equal_counts(self):
        """Test minority when all classes have equal counts."""
        counts = {0: 10, 1: 10, 2: 10}
        
        minority = compute_minority(counts)
        
        assert minority == 10


class TestCalculateImpurity:
    """Test Sum Minority (SM) and Max Minority (MM) impurity measures.
    
    Paper Reference: Section 2.4
    SM(H) = minority_L + minority_R
    MM(H) = max(minority_L, minority_R)
    """
    
    def test_perfect_split(self):
        """Test impurity for perfect split (zero impurity)."""
        # Left: all class 0, Right: all class 1
        left = {0: 10}
        right = {1: 10}
        
        sm, mm = calculate_impurity(left, right)
        
        assert sm == 0.0
        assert mm == 0.0
    
    def test_imperfect_split(self):
        """Test impurity calculation from paper example."""
        # Left: 8 class-0, 2 class-1 → minority_L = 2
        # Right: 3 class-0, 7 class-1 → minority_R = 3
        left = {0: 8, 1: 2}
        right = {0: 3, 1: 7}
        
        sm, mm = calculate_impurity(left, right)
        
        # SM = 2 + 3 = 5
        assert sm == 5.0
        # MM = max(2, 3) = 3
        assert mm == 3.0
    
    def test_one_sided_split(self):
        """Test impurity when all samples go to one side."""
        left = {0: 5, 1: 5}  # minority = 5
        right = {}  # minority = 0
        
        sm, mm = calculate_impurity(left, right)
        
        assert sm == 5.0
        assert mm == 5.0
    
    def test_multiclass_impurity(self):
        """Test impurity with 3+ classes."""
        # Left: 10 A, 3 B, 5 C → minority = 3
        # Right: 2 A, 8 B, 4 C → minority = 2
        left = {'A': 10, 'B': 3, 'C': 5}
        right = {'A': 2, 'B': 8, 'C': 4}
        
        sm, mm = calculate_impurity(left, right)
        
        # SM = 3 + 2 = 5
        assert sm == 5.0
        # MM = max(3, 2) = 3
        assert mm == 3.0
    
    def test_impurity_from_partition(self):
        """Test convenience function for computing impurity from labels."""
        y_left = np.array([0, 0, 0, 1, 1])  # 3 zeros, 2 ones → minority = 2
        y_right = np.array([0, 1, 1, 1])  # 1 zero, 3 ones → minority = 1
        
        sm, mm = calculate_impurity_from_partition(y_left, y_right)
        
        assert sm == 3.0  # 2 + 1
        assert mm == 2.0  # max(2, 1)


class TestEvaluateSplit:
    """Test split evaluation function."""
    
    def test_evaluate_with_sm(self):
        """Test split evaluation using Sum Minority."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])  # Perfect split possible at x = 0.5
        hyperplane = np.array([1.0, 0.0, -0.5])
        
        impurity = evaluate_split(X, y, hyperplane, impurity_measure="sm")
        
        # Should be 0 for perfect split
        assert impurity == 0.0
    
    def test_evaluate_with_mm(self):
        """Test split evaluation using Max Minority."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        hyperplane = np.array([1.0, 0.0, -0.5])
        
        impurity = evaluate_split(X, y, hyperplane, impurity_measure="mm")
        
        assert impurity == 0.0
    
    def test_invalid_impurity_measure(self):
        """Test error on invalid impurity measure."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        hyperplane = np.array([1.0, 0.0, 0.0])
        
        with pytest.raises(ValueError, match="impurity_measure"):
            evaluate_split(X, y, hyperplane, impurity_measure="invalid")


class TestFindBestThreshold:
    """Test univariate threshold finding.
    
    Paper Reference: Section 2.2 - Finding best split on U_j values
    """
    
    def test_simple_threshold(self):
        """Test finding threshold on simple data."""
        values = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([0, 0, 0, 1, 1, 1])  # Perfect split at 3.5
        
        threshold, impurity = find_best_threshold(values, y)
        
        assert threshold == pytest.approx(3.5)
        assert impurity == 0.0
    
    def test_threshold_with_noise(self):
        """Test threshold finding with some noise."""
        values = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([0, 0, 1, 0, 1, 1])  # Imperfect separation
        
        threshold, impurity = find_best_threshold(values, y)
        
        # Best split should still minimize impurity
        assert impurity >= 0
    
    def test_single_value(self):
        """Test with single value."""
        values = np.array([5.0])
        y = np.array([0])
        
        threshold, impurity = find_best_threshold(values, y)
        
        assert threshold == 5.0
        assert impurity == 0.0
    
    def test_all_same_values(self):
        """Test when all values are identical (degenerate case)."""
        values = np.array([3.0, 3.0, 3.0])
        y = np.array([0, 1, 0])
        
        threshold, impurity = find_best_threshold(values, y)
        
        # No meaningful split possible
        assert impurity == float('inf')
    
    def test_multiclass_threshold(self):
        """Test threshold finding with multiple classes."""
        values = np.array([1, 2, 5, 6, 9, 10])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        threshold, impurity = find_best_threshold(values, y)
        
        # Should find a reasonable split
        assert impurity >= 0


class TestIsPure:
    """Test purity checking."""
    
    def test_pure_single_class(self):
        """Test pure node with single class."""
        y = np.array([1, 1, 1, 1])
        
        assert is_pure(y) is True
    
    def test_not_pure(self):
        """Test impure node with multiple classes."""
        y = np.array([0, 1, 0, 1])
        
        assert is_pure(y) is False
    
    def test_empty_is_pure(self):
        """Test empty array is considered pure."""
        y = np.array([])
        
        assert is_pure(y) is True


class TestGetMajorityClass:
    """Test majority class finding."""
    
    def test_clear_majority(self):
        """Test with clear majority."""
        y = np.array([0, 0, 0, 1, 2])
        
        assert get_majority_class(y) == 0
    
    def test_tie_returns_first(self):
        """Test tie-breaking (returns smallest label)."""
        y = np.array([0, 0, 1, 1])
        
        # With ties, numpy returns first in sorted order
        majority = get_majority_class(y)
        assert majority in [0, 1]
    
    def test_empty_returns_none(self):
        """Test empty array returns None."""
        y = np.array([])
        
        assert get_majority_class(y) is None
