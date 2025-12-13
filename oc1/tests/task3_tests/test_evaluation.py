"""
Tests for Task 3: Evaluation Tools
"""

import pytest
import numpy as np
from oc1 import ObliqueDecisionTree
from oc1.evaluation import (
    train_test_split,
    confusion_matrix,
    classification_report,
    cross_validate,
)
from oc1.data import make_diagonal_dataset, make_xor_dataset


class TestTrainTestSplit:
    """Test train_test_split function."""
    
    def test_basic_split(self):
        """Test basic train/test split."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 1, 1, 1])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_stratified_split(self):
        """Test stratified split maintains class distribution."""
        X = np.random.randn(100, 2)
        y = np.array([0] * 50 + [1] * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=True, random_state=42
        )
        
        # Check class distribution is roughly maintained
        train_ratio = np.sum(y_train == 0) / len(y_train)
        test_ratio = np.sum(y_test == 0) / len(y_test)
        
        # Should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1
    
    def test_invalid_test_size(self):
        """Test that invalid test_size raises error."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.5)
        
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.0)


class TestConfusionMatrix:
    """Test confusion_matrix function."""
    
    def test_binary_classification(self):
        """Test confusion matrix for binary classification."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # True negatives
        assert cm[1, 1] == 2  # True positives
        assert cm[0, 1] == 1  # False positives
        assert cm[1, 0] == 1  # False negatives
    
    def test_multiclass(self):
        """Test confusion matrix for multiclass."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # Class 0 correctly predicted
        assert cm[1, 1] == 2  # Class 1 correctly predicted
        assert cm[2, 2] == 1  # Class 2 correctly predicted


class TestClassificationReport:
    """Test classification_report function."""
    
    def test_basic_report(self):
        """Test that classification report generates correctly."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        report = classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert "Precision" in report
        assert "Recall" in report
        assert "F1-Score" in report


class TestCrossValidate:
    """Test cross_validate function."""
    
    def test_basic_cross_validation(self):
        """Test basic cross-validation."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        
        results = cross_validate(tree, X, y, cv=5, random_state=42)
        
        assert 'test_score' in results
        assert 'fit_time' in results
        assert 'score_time' in results
        assert len(results['test_score']) == 5
        assert all(0 <= score <= 1 for score in results['test_score'])
    
    def test_cross_validation_with_train_score(self):
        """Test cross-validation with training scores."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        
        results = cross_validate(
            tree, X, y, cv=3, return_train_score=True, random_state=42
        )
        
        assert 'train_score' in results
        assert len(results['train_score']) == 3
        # Training score should generally be >= test score
        assert results['train_score'].mean() >= 0.0
    
    def test_invalid_cv(self):
        """Test that invalid cv raises error."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        tree = ObliqueDecisionTree()
        
        with pytest.raises(ValueError):
            cross_validate(tree, X, y, cv=1)
    
    def test_different_scoring_metrics(self):
        """Test different scoring metrics."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        tree = ObliqueDecisionTree(max_depth=3, random_state=42)
        
        for metric in ["accuracy", "precision", "recall", "f1"]:
            results = cross_validate(
                tree, X, y, cv=3, scoring=metric, random_state=42
            )
            assert len(results['test_score']) == 3
            assert all(0 <= score <= 1 for score in results['test_score'])

