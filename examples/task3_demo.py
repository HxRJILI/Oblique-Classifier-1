"""
Task 3 Demo: Pruning, Evaluation, Logging, and Visualization

This script demonstrates the Task 3 features:
- Pruning (impurity-based and Reduced Error Pruning)
- Cross-validation and evaluation metrics
- Detailed logging
- Visualization (if matplotlib is available)
"""

import numpy as np
from oc1 import ObliqueDecisionTree
from oc1.evaluation import (
    train_test_split,
    cross_validate,
    confusion_matrix,
    classification_report,
)
from oc1.data import make_diagonal_dataset, make_xor_dataset

try:
    from oc1.visualization import plot_decision_boundary_2d, plot_hyperplanes_2d
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization not available (matplotlib not installed)")


def demo_pruning():
    """Demonstrate pruning functionality."""
    print("=" * 60)
    print("Demo: Pruning")
    print("=" * 60)
    
    # Generate data
    X, y = make_diagonal_dataset(n_samples=200, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Build tree without pruning
    print("\n1. Building tree without pruning...")
    tree = ObliqueDecisionTree(max_depth=10, random_state=42, verbose=False)
    tree.fit(X_train, y_train)
    
    print(f"   Nodes before pruning: {tree.get_n_nodes()}")
    print(f"   Leaves before pruning: {tree.get_n_leaves()}")
    print(f"   Depth: {tree.get_depth()}")
    print(f"   Validation accuracy: {tree.score(X_val, y_val):.3f}")
    
    # Prune using Reduced Error Pruning
    print("\n2. Pruning with Reduced Error Pruning...")
    tree.prune(method="rep", X_val=X_val, y_val=y_val)
    
    print(f"   Nodes after REP: {tree.get_n_nodes()}")
    print(f"   Leaves after REP: {tree.get_n_leaves()}")
    print(f"   Validation accuracy: {tree.score(X_val, y_val):.3f}")
    
    # Prune using impurity threshold
    print("\n3. Pruning with impurity threshold...")
    tree2 = ObliqueDecisionTree(max_depth=10, random_state=42, verbose=False)
    tree2.fit(X_train, y_train)
    
    print(f"   Nodes before: {tree2.get_n_nodes()}")
    tree2.prune(method="impurity", impurity_threshold=2.0)
    print(f"   Nodes after impurity pruning: {tree2.get_n_nodes()}")
    print(f"   Validation accuracy: {tree2.score(X_val, y_val):.3f}")


def demo_evaluation():
    """Demonstrate evaluation tools."""
    print("\n" + "=" * 60)
    print("Demo: Evaluation Tools")
    print("=" * 60)
    
    # Generate data
    X, y = make_diagonal_dataset(n_samples=200, random_state=42)
    
    # Cross-validation
    print("\n1. Cross-Validation:")
    tree = ObliqueDecisionTree(max_depth=5, random_state=42)
    results = cross_validate(tree, X, y, cv=5, random_state=42)
    
    print(f"   Mean accuracy: {results['test_score'].mean():.3f} ± {results['test_score'].std():.3f}")
    print(f"   Mean fit time: {results['fit_time'].mean():.3f}s")
    
    # Train/test split and metrics
    print("\n2. Train/Test Split and Metrics:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:\n{cm}")
    
    # Classification report
    print("\n3. Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)


def demo_logging():
    """Demonstrate logging functionality."""
    print("\n" + "=" * 60)
    print("Demo: Detailed Logging")
    print("=" * 60)
    
    # Generate data
    X, y = make_diagonal_dataset(n_samples=100, random_state=42)
    
    # Build tree with verbose logging
    print("\n1. Building tree with verbose logging:")
    print("-" * 60)
    tree = ObliqueDecisionTree(
        max_depth=4,
        verbose=True,
        random_state=42
    )
    tree.fit(X, y)
    print("-" * 60)
    
    # Get log summary
    print("\n2. Log Summary:")
    summary = tree.logger.get_log_summary()
    print(f"   Total events: {summary['total_events']}")
    print(f"   Nodes created: {summary['nodes_created']}")
    print(f"   Hyperplanes found: {summary['hyperplanes_found']}")
    print(f"   Splits: {summary['splits']}")
    
    # Log to file
    print("\n3. Logging to file...")
    tree2 = ObliqueDecisionTree(
        max_depth=4,
        log_file="tree_construction.log",
        verbose=False,
        random_state=42
    )
    tree2.fit(X, y)
    print("   Log written to tree_construction.log")


def demo_visualization():
    """Demonstrate visualization (if available)."""
    if not VISUALIZATION_AVAILABLE:
        print("\n" + "=" * 60)
        print("Demo: Visualization (skipped - matplotlib not available)")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("Demo: Visualization")
    print("=" * 60)
    
    # Generate data
    X, y = make_diagonal_dataset(n_samples=100, random_state=42)
    
    # Build tree
    tree = ObliqueDecisionTree(max_depth=5, random_state=42)
    tree.fit(X, y)
    
    print("\n1. Plotting decision boundary...")
    try:
        import matplotlib.pyplot as plt
        plot_decision_boundary_2d(tree, X, y)
        plt.savefig("decision_boundary.png", dpi=150, bbox_inches='tight')
        print("   Saved to decision_boundary.png")
        plt.close()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Plotting hyperplanes...")
    try:
        import matplotlib.pyplot as plt
        plot_hyperplanes_2d(tree, X)
        plt.savefig("hyperplanes.png", dpi=150, bbox_inches='tight')
        print("   Saved to hyperplanes.png")
        plt.close()
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("OC1 Task 3 Feature Demonstrations")
    print("=" * 60)
    
    demo_pruning()
    demo_evaluation()
    demo_logging()
    demo_visualization()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

