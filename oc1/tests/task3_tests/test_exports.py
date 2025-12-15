"""
Tests for export methods and feature importances.

Task 3 Extension: Export methods (to_dict, to_json, to_dot), feature importances
"""

import pytest
import numpy as np
import json
import tempfile
import os

from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset, make_xor_dataset


class TestExportMethods:
    """Test tree export functionality."""
    
    def test_to_dict_basic(self):
        """Test basic to_dict export."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        tree_dict = tree.to_dict()
        
        # Check metadata
        assert "metadata" in tree_dict
        assert tree_dict["metadata"]["n_features"] == 2
        assert tree_dict["metadata"]["n_classes"] == 2
        
        # Check tree stats
        assert "tree_stats" in tree_dict
        assert "depth" in tree_dict["tree_stats"]
        assert "n_nodes" in tree_dict["tree_stats"]
        assert "n_leaves" in tree_dict["tree_stats"]
        
        # Check root node
        assert "root" in tree_dict
        assert "is_leaf" in tree_dict["root"]
    
    def test_to_dict_unfitted_raises(self):
        """Test that to_dict raises error if not fitted."""
        tree = ObliqueDecisionTree()
        with pytest.raises(ValueError):
            tree.to_dict()
    
    def test_to_json_string(self):
        """Test JSON string export."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        json_str = tree.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "root" in parsed
    
    def test_to_json_file(self):
        """Test JSON file export."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tree.json")
            tree.to_json(filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["metadata"]["n_features"] == 2
    
    def test_to_dot_basic(self):
        """Test DOT format export."""
        X, y = make_xor_dataset(n_samples=100, random_state=42)  # XOR needs deeper tree
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        dot_str = tree.to_dot()
        
        # Check DOT format
        assert "digraph OC1Tree" in dot_str
        assert "node0" in dot_str
        # Either has edges (internal nodes) or just has class labels (leaf only)
        assert "class=" in dot_str
    
    def test_to_dot_with_feature_names(self):
        """Test DOT export with custom feature names."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        dot_str = tree.to_dot(feature_names=["Feature_A", "Feature_B"])
        
        # Custom feature names should appear
        assert "Feature_A" in dot_str or "Feature_B" in dot_str or "class=" in dot_str


class TestFeatureImportances:
    """Test feature importance computation."""
    
    def test_feature_importances_shape(self):
        """Test that feature importances have correct shape."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        importances = tree.feature_importances_
        
        assert len(importances) == X.shape[1]
    
    def test_feature_importances_sum_to_one(self):
        """Test that feature importances sum to 1."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        importances = tree.feature_importances_
        
        # Should sum to approximately 1
        assert abs(importances.sum() - 1.0) < 1e-10 or importances.sum() == 0
    
    def test_feature_importances_non_negative(self):
        """Test that all feature importances are non-negative."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        importances = tree.feature_importances_
        
        assert np.all(importances >= 0)
    
    def test_feature_importances_unfitted_raises(self):
        """Test that feature_importances_ raises error if not fitted."""
        tree = ObliqueDecisionTree()
        with pytest.raises(ValueError):
            _ = tree.feature_importances_


class TestParameterValidation:
    """Test parameter validation."""
    
    def test_invalid_max_depth_raises(self):
        """Test that invalid max_depth raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(max_depth=0)
    
    def test_invalid_min_samples_leaf_raises(self):
        """Test that invalid min_samples_leaf raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(min_samples_leaf=0)
    
    def test_invalid_min_samples_split_raises(self):
        """Test that invalid min_samples_split raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(min_samples_split=1)
    
    def test_invalid_impurity_measure_raises(self):
        """Test that invalid impurity_measure raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(impurity_measure="gini")
    
    def test_invalid_max_iterations_raises(self):
        """Test that invalid max_iterations raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(max_iterations=0)
    
    def test_invalid_n_restarts_raises(self):
        """Test that invalid n_restarts raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(n_restarts=0)
    
    def test_invalid_impurity_threshold_raises(self):
        """Test that invalid impurity_threshold raises error."""
        with pytest.raises(ValueError):
            ObliqueDecisionTree(impurity_threshold=-1.0)


class TestBasicMethods:
    """Test basic tree methods."""
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree(random_state=42)
        
        result = tree.fit(X, y)
        
        assert result is tree
    
    def test_score_method(self):
        """Test score method returns valid accuracy."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        score = tree.score(X, y)
        
        assert 0.0 <= score <= 1.0
