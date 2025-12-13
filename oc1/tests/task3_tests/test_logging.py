"""
Tests for Task 3: Logging Functionality
"""

import pytest
import numpy as np
import tempfile
import os
from oc1 import ObliqueDecisionTree
from oc1.core.logging import TreeConstructionLogger
from oc1.data import make_diagonal_dataset


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_initialization(self):
        """Test logger can be initialized."""
        logger = TreeConstructionLogger(verbose=False)
        assert logger is not None
        assert logger.construction_log == []
    
    def test_log_tree_construction(self):
        """Test logging during tree construction."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3, verbose=True, random_state=42)
        tree.fit(X, y)
        
        # Logger should be initialized
        assert tree.logger is not None
        
        # Should have log entries
        log_summary = tree.logger.get_log_summary()
        assert log_summary['total_events'] > 0
        assert log_summary['nodes_created'] > 0
    
    def test_log_to_file(self):
        """Test logging to file."""
        # Create temporary file with a unique name
        log_file = tempfile.mktemp(suffix='.log')
        
        tree = None
        try:
            X, y = make_diagonal_dataset(n_samples=50, random_state=42)
            
            tree = ObliqueDecisionTree(
                max_depth=3,
                log_file=log_file,
                verbose=False,
                random_state=42
            )
            tree.fit(X, y)
            
            # Close all handlers to release file handle
            if tree.logger:
                tree.logger.close()
            
            # Check file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert "Tree Construction" in content
        finally:
            # Ensure file is closed before deletion
            if tree and tree.logger:
                tree.logger.close()
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except PermissionError:
                    # On Windows, sometimes need to wait a bit
                    import time
                    time.sleep(0.1)
                    try:
                        os.remove(log_file)
                    except PermissionError:
                        pass  # Skip if still can't delete
    
    def test_log_summary(self):
        """Test log summary generation."""
        X, y = make_diagonal_dataset(n_samples=50, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=3, verbose=False, random_state=42)
        tree.fit(X, y)
        
        summary = tree.logger.get_log_summary()
        
        assert 'total_events' in summary
        assert 'nodes_created' in summary
        assert 'hyperplanes_found' in summary
        assert 'splits' in summary
        assert 'construction_log' in summary
        assert isinstance(summary['construction_log'], list)
    
    def test_log_pruning(self):
        """Test logging pruning operations."""
        X, y = make_diagonal_dataset(n_samples=100, random_state=42)
        
        tree = ObliqueDecisionTree(max_depth=5, verbose=False, random_state=42)
        tree.fit(X, y)
        
        n_leaves_before = tree.get_n_leaves()
        
        tree.prune(method="impurity", impurity_threshold=5.0)
        
        # Check that pruning was logged
        log_summary = tree.logger.get_log_summary()
        pruning_events = [
            e for e in log_summary['construction_log']
            if e.get('event') == 'pruning'
        ]
        assert len(pruning_events) > 0

