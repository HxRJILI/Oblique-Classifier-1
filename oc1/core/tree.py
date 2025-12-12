"""
OC1 Oblique Decision Tree Classifier

This module implements the main ObliqueDecisionTree class as specified in the OC1 paper
"OC1: A randomized algorithm for building oblique decision trees" by Murthy et al. (1992).

Paper Reference:
- Section 2: Recursive tree construction algorithm
- Section 2.1: Hill-climbing for hyperplane optimization
- Section 2.4: Stopping criteria (zero impurity)

Key Features:
- Recursive tree construction with oblique hyperplane splits
- Deterministic hill-climbing optimization (Task 1)
- Multi-class classification support
- Compatible with Task 2 (randomization) and Task 3 (pruning) extensions
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from oc1.core.node import ObliqueTreeNode
from oc1.core.splits import (
    partition_data,
    calculate_impurity_from_partition,
    compute_class_counts,
    is_pure,
    get_majority_class,
)
from oc1.core.hill_climb import (
    hill_climb,
    find_best_hyperplane,
    initialize_hyperplane,
)


class ObliqueDecisionTree:
    """
    Oblique Decision Tree Classifier implementing the OC1 algorithm.
    
    This classifier builds a decision tree where each internal node contains
    an oblique hyperplane that splits the feature space. Unlike axis-parallel
    decision trees (like CART or C4.5), oblique trees can represent linear
    decision boundaries at any angle.
    
    Hyperplane equation at each node:
        ∑_{i=1}^{d} (a_i * x_i) + a_{d+1} = 0
    
    Partitioning rule (Section 2):
        - Left child: V_j > 0
        - Right child: V_j ≤ 0
        where V_j = ∑(a_i * x_j^i) + a_{d+1}
    
    Attributes:
        root: Root node of the tree (ObliqueTreeNode)
        n_features: Number of features in training data
        classes_: Unique class labels
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples required in a leaf
        min_samples_split: Minimum samples required to split a node
        impurity_measure: "sm" (Sum Minority) or "mm" (Max Minority)
        max_iterations: Maximum hill-climbing iterations
        n_restarts: Number of random restarts (1 for deterministic)
        random_state: Random seed for reproducibility
    
    Paper Reference: Murthy et al., AAAI-1992
    
    Example:
        >>> from oc1.core.tree import ObliqueDecisionTree
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> y = np.array([0, 1, 1, 0])
        >>> tree = ObliqueDecisionTree(max_depth=3)
        >>> tree.fit(X, y)
        >>> tree.predict(X)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        impurity_measure: str = "sm",
        max_iterations: int = 100,
        n_restarts: int = 10,
        random_state: Optional[int] = None,
        impurity_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the Oblique Decision Tree classifier.
        
        Args:
            max_depth: Maximum depth of the tree. None for unlimited.
            min_samples_leaf: Minimum number of samples required in a leaf node.
            min_samples_split: Minimum number of samples required to split a node.
            impurity_measure: "sm" for Sum Minority or "mm" for Max Minority.
            max_iterations: Maximum number of hill-climbing iterations per node.
            n_restarts: Number of random restarts for hill-climbing.
                       Use 1 for deterministic (Task 1), >1 for randomized (Task 2).
            random_state: Random seed for reproducibility.
            impurity_threshold: Stop splitting if impurity falls below this.
                               Prepared for Task 3 pruning integration.
        
        Paper Reference: Section 2.4 - Stopping criteria
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.impurity_measure = impurity_measure.lower()
        self.max_iterations = max_iterations
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.impurity_threshold = impurity_threshold
        
        # Validate impurity measure
        if self.impurity_measure not in ("sm", "mm"):
            raise ValueError(
                f"impurity_measure must be 'sm' or 'mm', got {impurity_measure}"
            )
        
        # Tree state (set after fit)
        self.root: Optional[ObliqueTreeNode] = None
        self.n_features_: int = 0
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0
        self._is_fitted: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ObliqueDecisionTree':
        """
        Build the oblique decision tree from training data.
        
        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
        
        Returns:
            self: The fitted classifier.
        
        Raises:
            ValueError: If X and y have inconsistent shapes.
        
        Paper Reference: Section 2 - Recursive tree construction
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        # Set random state using modern RNG
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)
        else:
            self._rng = np.random.default_rng()
        
        # Store dataset info
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0)
        self._is_fitted = True
        
        return self
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        parent: Optional[ObliqueTreeNode] = None,
    ) -> ObliqueTreeNode:
        """
        Recursively build the decision tree.
        
        This implements the recursive tree construction algorithm:
        1. Check stopping criteria (pure node, min samples, max depth)
        2. Find best hyperplane using hill-climbing
        3. Partition data and recursively build children
        
        Args:
            X: Feature matrix for current node.
            y: Labels for current node.
            depth: Current depth in the tree.
            parent: Parent node reference (for Task 3 pruning support).
        
        Returns:
            ObliqueTreeNode: The constructed node (may be leaf or internal).
        
        Paper Reference: Section 2 - Recursive algorithm
        """
        n_samples = len(y)
        class_counts = compute_class_counts(y)
        majority_class = get_majority_class(y)
        
        # Create node with parent reference
        node = ObliqueTreeNode(
            class_distribution=class_counts,
            depth=depth,
            n_samples=n_samples,
            predicted_class=majority_class,
            parent=parent,
        )
        
        # Check stopping criteria (Section 2.4)
        should_stop = (
            is_pure(y) or  # Zero impurity
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            (self.max_depth is not None and depth >= self.max_depth)
        )
        
        if should_stop:
            node.is_leaf = True
            return node
        
        # Find best hyperplane using hill-climbing (Section 2.1)
        # Generate a seed from tree's RNG for reproducibility
        node_seed = int(self._rng.integers(0, 2**31 - 1))
        try:
            best_hyperplane, best_impurity = find_best_hyperplane(
                X, y,
                impurity_measure=self.impurity_measure,
                n_restarts=self.n_restarts,
                max_iterations=self.max_iterations,
                random_state=node_seed,
                use_random_perturbation_order=True,  # Task 2: Enable random perturbation order
            )
        except Exception:
            # Fall back to leaf if hyperplane finding fails
            node.is_leaf = True
            return node
        
        # Check if split is useful (impurity threshold for Task 3)
        if best_impurity <= self.impurity_threshold:
            node.is_leaf = True
            node.impurity = best_impurity
            return node
        
        # Partition data
        X_left, y_left, X_right, y_right, _ = partition_data(X, y, best_hyperplane)
        
        # Check if partition is valid
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node.is_leaf = True
            return node
        
        # Check for degenerate split (all to one side)
        if len(y_left) == 0 or len(y_right) == 0:
            node.is_leaf = True
            return node
        
        # Set hyperplane and create internal node
        node.hyperplane = best_hyperplane
        node.impurity = best_impurity
        node.is_leaf = False
        
        # Recursively build children with parent reference
        node.left_child = self._build_tree(X_left, y_left, depth + 1, parent=node)
        node.right_child = self._build_tree(X_right, y_right, depth + 1, parent=node)
        
        return node
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        
        Raises:
            ValueError: If the tree has not been fitted.
        """
        self._check_is_fitted()
        
        X = np.atleast_2d(X)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features but tree was trained with "
                f"{self.n_features_} features"
            )
        
        predictions = np.array([
            self.root.predict_single(x) for x in X
        ])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Probabilities are based on the class distribution in the leaf node.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        probabilities = np.zeros((n_samples, self.n_classes_))
        
        for i, x in enumerate(X):
            leaf = self._get_leaf(x)
            total = sum(leaf.class_distribution.values())
            
            for j, cls in enumerate(self.classes_):
                count = leaf.class_distribution.get(cls, 0)
                probabilities[i, j] = count / total if total > 0 else 0
        
        return probabilities
    
    def _get_leaf(self, x: np.ndarray) -> ObliqueTreeNode:
        """
        Get the leaf node for a single sample.
        
        Args:
            x: Feature vector of shape (n_features,).
        
        Returns:
            ObliqueTreeNode: The leaf node reached by the sample.
        """
        node = self.root
        
        while not node.is_leaf:
            x = np.atleast_1d(x)
            V = node.evaluate(x.reshape(1, -1))[0]
            
            if V > 0:
                node = node.left_child if node.left_child else node
            else:
                node = node.right_child if node.right_child else node
        
        return node
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True class labels of shape (n_samples,).
        
        Returns:
            float: Accuracy score (proportion of correct predictions).
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_depth(self) -> int:
        """
        Get the maximum depth of the tree.
        
        Returns:
            int: Maximum depth (0 if tree is just a leaf).
        """
        self._check_is_fitted()
        return self.root.get_tree_depth()
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaf nodes in the tree.
        
        Returns:
            int: Number of leaves.
        """
        self._check_is_fitted()
        return self.root.count_leaves()
    
    def get_n_nodes(self) -> int:
        """
        Get the total number of nodes in the tree.
        
        Returns:
            int: Total number of nodes.
        """
        self._check_is_fitted()
        return self.root.count_nodes()
    
    def get_all_nodes(self) -> List[ObliqueTreeNode]:
        """
        Get all nodes in the tree in breadth-first order.
        
        This method is useful for Task 3 pruning operations that need
        to traverse or inspect all nodes in the tree.
        
        Returns:
            List[ObliqueTreeNode]: All nodes in breadth-first order.
        
        Example:
            >>> tree.fit(X, y)
            >>> nodes = tree.get_all_nodes()
            >>> leaves = [n for n in nodes if n.is_leaf]
        """
        self._check_is_fitted()
        
        nodes = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            
            if node.left_child:
                queue.append(node.left_child)
            if node.right_child:
                queue.append(node.right_child)
        
        return nodes
    
    def prune(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        method: str = "rep",
    ) -> 'ObliqueDecisionTree':
        """
        Prune the tree to reduce overfitting.
        
        This is a placeholder for Task 3 pruning implementation.
        
        Planned pruning methods:
        - "rep": Reduced Error Pruning - prune subtrees that don't improve
                 validation accuracy
        - "mep": Minimum Error Pruning - prune based on expected error
        - "cost_complexity": Cost-complexity pruning (CART-style)
        
        Args:
            X_val: Validation feature matrix for pruning decisions.
            y_val: Validation labels.
            method: Pruning method to use ("rep", "mep", "cost_complexity").
        
        Returns:
            self: The pruned tree.
        
        Raises:
            NotImplementedError: This method will be implemented in Task 3.
        
        Paper Reference: Section 4 - Pruning (to be implemented in Task 3)
        """
        raise NotImplementedError(
            "prune() will be implemented in Task 3. "
            "This placeholder ensures API compatibility for pruning extensions."
        )
    
    def _check_is_fitted(self) -> None:
        """Check if the tree has been fitted."""
        if not self._is_fitted or self.root is None:
            raise ValueError(
                "This ObliqueDecisionTree instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
    
    def get_hyperplanes(self) -> List[Tuple[np.ndarray, int]]:
        """
        Get all hyperplanes in the tree with their depths.
        
        Returns:
            List of (hyperplane, depth) tuples for all internal nodes.
        """
        self._check_is_fitted()
        
        hyperplanes = []
        self._collect_hyperplanes(self.root, hyperplanes)
        return hyperplanes
    
    def _collect_hyperplanes(
        self,
        node: ObliqueTreeNode,
        hyperplanes: List[Tuple[np.ndarray, int]],
    ) -> None:
        """Recursively collect hyperplanes from the tree."""
        if node is None or node.is_leaf:
            return
        
        if node.hyperplane is not None:
            hyperplanes.append((node.hyperplane.copy(), node.depth))
        
        self._collect_hyperplanes(node.left_child, hyperplanes)
        self._collect_hyperplanes(node.right_child, hyperplanes)
    
    def print_tree(self, feature_names: Optional[List[str]] = None) -> str:
        """
        Generate a string representation of the tree structure.
        
        Args:
            feature_names: Optional list of feature names for readable output.
        
        Returns:
            str: Formatted tree structure.
        """
        self._check_is_fitted()
        
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_)]
        
        lines = []
        self._print_node(self.root, "", True, feature_names, lines)
        return "\n".join(lines)
    
    def _print_node(
        self,
        node: ObliqueTreeNode,
        prefix: str,
        is_last: bool,
        feature_names: List[str],
        lines: List[str],
    ) -> None:
        """Recursively print node information."""
        connector = "└── " if is_last else "├── "
        
        if node.is_leaf:
            lines.append(
                f"{prefix}{connector}Leaf: class={node.predicted_class}, "
                f"samples={node.n_samples}, dist={node.class_distribution}"
            )
        else:
            # Format hyperplane equation
            terms = []
            for i, coef in enumerate(node.hyperplane[:-1]):
                if abs(coef) > 1e-10:
                    sign = "+" if coef > 0 else "-"
                    terms.append(f"{sign}{abs(coef):.3f}*{feature_names[i]}")
            
            bias = node.hyperplane[-1]
            bias_str = f"{'+' if bias >= 0 else '-'}{abs(bias):.3f}"
            equation = "".join(terms) + bias_str
            
            lines.append(
                f"{prefix}{connector}Split: {equation} = 0, "
                f"samples={node.n_samples}, impurity={node.impurity:.3f}"
            )
            
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            if node.left_child:
                self._print_node(
                    node.left_child, new_prefix, node.right_child is None,
                    feature_names, lines
                )
            if node.right_child:
                self._print_node(
                    node.right_child, new_prefix, True,
                    feature_names, lines
                )
    
    def __repr__(self) -> str:
        """String representation of the tree."""
        if self._is_fitted:
            return (
                f"ObliqueDecisionTree(depth={self.get_depth()}, "
                f"n_leaves={self.get_n_leaves()}, "
                f"impurity='{self.impurity_measure}')"
            )
        else:
            return "ObliqueDecisionTree(not fitted)"
