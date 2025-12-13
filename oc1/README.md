# OC1: Oblique Classifier 1 - Python Implementation

A Python implementation of the OC1 oblique decision tree algorithm as described in:

> **"OC1: A randomized algorithm for building oblique decision trees"**  
> by Sreerama K. Murthy, Simon Kasif, Steven Salzberg, and Richard Beigel  
> AAAI-1992

## Task 1: Core Tree Construction (Deterministic)

This implementation covers the core deterministic components of OC1:

1. **Data Structures** (Section 2)
   - `ObliqueTreeNode`: Node with hyperplane coefficients, class distribution, children
   - `ObliqueDecisionTree`: Main classifier class

2. **Oblique Split Evaluation** (Section 2)
   - Hyperplane: `∑(a_i * x_i) + a_{d+1} = 0`
   - Partition rule: Left if `V_j > 0`, Right if `V_j ≤ 0`

3. **Impurity Measures** (Section 2.4)
   - Sum Minority (SM): `SM = minority_L + minority_R`
   - Max Minority (MM): `MM = max(minority_L, minority_R)`

4. **Coefficient Perturbation** (Section 2.2, Equation 1)
   - `U_j = a_m * x_j^m - V_j / x_j^m`
   - Find best univariate threshold on U values

5. **Deterministic Hill-Climbing** (Section 2.1)
   - Sequential coefficient perturbation
   - Accept if impurity improves
   - Continue until local minimum
  
## Compatibility with Future Tasks

This implementation is designed to integrate with:

- **Task 2 (Randomization)**: `n_restarts` parameter, modular `find_best_hyperplane()`
- **Task 3 (Pruning)**: `impurity_threshold` parameter, stored node impurities

## Task 2: Randomization and Enhanced Search

This implementation includes Task 2 enhancements for improved hyperplane search:

1. **Random Hyperplane Initialization** (Section 2.3)
   - First trial: deterministic axis-parallel split
   - Subsequent trials: random direction initialization
   - Proper RNG seed propagation for reproducibility

2. **Multi-Coefficient Perturbation**
   - Perturb 2-5 coefficients simultaneously
   - Explore diagonal directions in coefficient space
   - Escape local minima more effectively

3. **K Random Trials** (n_restarts parameter)
   - Try multiple random starting points
   - Select best result across all trials
   - Default: k=10 for good exploration

4. **Random Perturbation Order**
   - Randomize order of coefficient optimization
   - Reduce bias from sequential ordering

## Task 3: Auxiliary Features, Pruning, and Usability

This implementation includes Task 3 enhancements for production-ready usage:

1. **Pruning** (Section 2.4)
   - **Impurity-based pruning**: Prune subtrees where impurity is below threshold
   - **Reduced Error Pruning (REP)**: Prune based on validation set performance
   - Automatic subtree removal to reduce overfitting

2. **Explicit Stopping Criteria**
   - Zero impurity (pure nodes)
   - Minimum node size (`min_samples_leaf`, `min_samples_split`)
   - Maximum tree depth (`max_depth`)
   - Impurity threshold (`impurity_threshold`)

3. **Evaluation Tools**
   - **Cross-validation**: K-fold cross-validation with multiple metrics
   - **Train/test split**: Stratified data splitting
   - **Confusion matrix**: Classification performance analysis
   - **Classification report**: Detailed precision, recall, F1-score metrics

4. **Detailed Logging**
   - Log all tree construction steps
   - Record hyperplane coefficients and random seeds at each node
   - Track impurity values and stopping criteria
   - Optional file logging for analysis

5. **Visualization** (optional, requires matplotlib)
   - **Decision boundary plots**: Visualize 2D decision regions
   - **Hyperplane visualization**: Plot all splitting hyperplanes
   - **Tree structure**: Graph representation of tree topology

6. **Enhanced API**
   - Improved parameter documentation
   - Verbose mode for construction monitoring
   - Log file support for detailed analysis
   - Comprehensive error handling

### Using Task 2 Features

```python
# Deterministic (Task 1 only)
tree_det = ObliqueDecisionTree(n_restarts=1, random_state=42)

# Randomized (Task 2) - Default
tree_rand = ObliqueDecisionTree(n_restarts=5, random_state=42)

# Aggressive randomization
tree_aggressive = ObliqueDecisionTree(n_restarts=10, random_state=42)
```

### Performance Comparison

```python
from oc1.data import make_xor_dataset

X, y = make_xor_dataset(n_samples=200, random_state=42)

for k in [1, 3, 5, 10]:
    tree = ObliqueDecisionTree(n_restarts=k, random_state=42)
    tree.fit(X, y)
    print(f"k={k}: accuracy={tree.score(X, y):.3f}, depth={tree.get_depth()}")
```

## Installation

```bash
# From the project root
pip install -e .

# Or install dependencies directly
pip install numpy pytest
```

## Quick Start

```python
import numpy as np
from oc1 import ObliqueDecisionTree

# Create sample data
X = np.array([
    [0, 0], [0.2, 0.1], [0.1, 0.2],  # Class 0
    [1, 1], [0.8, 0.9], [0.9, 0.8],  # Class 1
])
y = np.array([0, 0, 0, 1, 1, 1])

# Train oblique decision tree
tree = ObliqueDecisionTree(
    max_depth=5,
    impurity_measure="sm",  # Sum Minority
    min_samples_leaf=1,
)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print(f"Predictions: {predictions}")

# Get accuracy
accuracy = tree.score(X, y)
print(f"Accuracy: {accuracy:.2%}")

# View tree structure
print(tree.print_tree())
```

## Module Structure

```
oc1/
├── __init__.py           # Package exports
├── core/
│   ├── node.py           # ObliqueTreeNode class
│   ├── tree.py           # ObliqueDecisionTree classifier
│   ├── splits.py         # Impurity measures, partitioning
│   └── hill_climb.py     # Coefficient perturbation, optimization
├── data/
│   └── datasets.py       # Synthetic test datasets
└── tests/
    ├── task1_tests/      # Task 1: Core tree construction tests
    │   ├── test_node.py
    │   ├── test_splits.py
    │   ├── test_hill_climb.py
    │   ├── test_tree.py
    │   ├── test_integration.py
    │   └── test_task_compatibility.py
    └── task2_tests/      # Task 2: Randomization tests
        └── test_task2.py
```

## API Reference

### ObliqueDecisionTree

```python
ObliqueDecisionTree(
    max_depth=None,           # Maximum tree depth
    min_samples_leaf=1,       # Minimum samples per leaf
    min_samples_split=2,      # Minimum samples to split
    impurity_measure="sm",    # "sm" (Sum Minority) or "mm" (Max Minority)
    max_iterations=100,       # Hill-climbing iterations per node
    n_restarts=5,             # Random restarts (5 = Task 2 default, 1 = deterministic)
    random_state=None,        # Random seed
    impurity_threshold=0.0,   # Stop splitting threshold (Task 3)
    verbose=False,            # Enable verbose logging (Task 3)
    log_file=None,            # Optional log file path (Task 3)
)
```

**Methods:**
- `fit(X, y)`: Train the tree
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `score(X, y)`: Calculate accuracy
- `get_depth()`: Get tree depth
- `get_n_leaves()`: Count leaf nodes
- `get_hyperplanes()`: Get all splitting hyperplanes
- `print_tree()`: Get tree visualization
- `prune(X_val, y_val, method, impurity_threshold)`: Prune tree (Task 3)
- `get_all_nodes()`: Get all nodes in breadth-first order

### Synthetic Datasets

```python
from oc1.data import (
    make_diagonal_dataset,      # 45° diagonal boundary
    make_xor_dataset,           # XOR problem
    make_oblique_classification, # Custom angle boundary
    make_multiclass_oblique,    # Multi-class sectors
    make_3d_oblique,            # 3D oblique plane
)

# Example
X, y = make_diagonal_dataset(n_samples=100, random_state=42)
```

## Task 3 Usage Examples

### Pruning

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import train_test_split
from oc1.data import make_diagonal_dataset

# Generate data
X, y = make_diagonal_dataset(n_samples=200, random_state=42)

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Build and prune tree
tree = ObliqueDecisionTree(max_depth=10, random_state=42)
tree.fit(X_train, y_train)

# Prune using validation set
tree.prune(method="rep", X_val=X_val, y_val=y_val)

# Or prune by impurity threshold
tree.prune(method="impurity", impurity_threshold=2.0)
```

### Cross-Validation

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import cross_validate
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=200, random_state=42)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)

# 5-fold cross-validation
results = cross_validate(tree, X, y, cv=5, random_state=42)

print(f"Mean accuracy: {results['test_score'].mean():.3f}")
print(f"Std accuracy: {results['test_score'].std():.3f}")
```

### Logging

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

# Enable verbose logging
tree = ObliqueDecisionTree(max_depth=5, verbose=True, random_state=42)
tree.fit(X, y)

# Or log to file
tree = ObliqueDecisionTree(
    max_depth=5,
    log_file="tree_construction.log",
    verbose=False,
    random_state=42
)
tree.fit(X, y)

# Get log summary
summary = tree.logger.get_log_summary()
print(f"Nodes created: {summary['nodes_created']}")
print(f"Hyperplanes found: {summary['hyperplanes_found']}")
```

### Visualization

```python
from oc1 import ObliqueDecisionTree
from oc1.visualization import plot_decision_boundary_2d, plot_hyperplanes_2d
from oc1.data import make_diagonal_dataset
import matplotlib.pyplot as plt

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X, y)

# Plot decision boundary
plot_decision_boundary_2d(tree, X, y)
plt.show()

# Plot hyperplanes
plot_hyperplanes_2d(tree, X)
plt.show()
```

## Running Tests

```bash
# Run all tests
pytest oc1/tests/ -v

# Run Task 1 tests only
pytest oc1/tests/task1_tests/ -v

# Run Task 2 tests only
pytest oc1/tests/task2_tests/ -v

# Run Task 3 tests only
pytest oc1/tests/task3_tests/ -v

# Run with coverage
pytest oc1/tests/ --cov=oc1 --cov-report=html

# Run specific test file
pytest oc1/tests/task1_tests/test_tree.py -v
```

## Paper Fidelity

This implementation follows the OC1 paper exactly:

### Task 1: Core Tree Construction

| Feature | Paper Section | Implementation |
|---------|--------------|----------------|
| Hyperplane equation | Section 2 | `evaluate_hyperplane()` |
| Partition rule (V>0 → left) | Section 2 | `partition_data()` |
| Sum Minority impurity | Section 2.4 | `calculate_impurity()` |
| Max Minority impurity | Section 2.4 | `calculate_impurity()` |
| U_j formula (Eq. 1) | Section 2.2 | `compute_u_values()` |
| Sequential perturbation | Section 2.1 | `hill_climb()` |

### Task 2: Randomization Enhancements

| Feature | Paper Section | Implementation |
|---------|--------------|----------------|
| Random hyperplane init | Section 2.3 | `initialize_hyperplane(method="random")` |
| K random trials | Section 2.3 | `n_restarts` parameter |
| Multi-coefficient perturbation | Section 2 | `perturb_multiple_coefficients()` |
| Random perturbation order | Section 2 | `use_random_perturbation_order` |
| Degenerate hyperplane handling | Section 2.4 | `validate_hyperplane()` |

### Task 3: Pruning and Evaluation

| Feature | Paper Section | Implementation |
|---------|--------------|----------------|
| Impurity-based pruning | Section 2.4 | `prune(method="impurity")` |
| Reduced Error Pruning | Section 2.4 | `prune(method="rep")` |
| Explicit stopping criteria | Section 2.4 | `max_depth`, `min_samples_leaf`, `impurity_threshold` |
| Cross-validation | Standard ML | `cross_validate()` |
| Evaluation metrics | Standard ML | `confusion_matrix()`, `classification_report()` |
| Detailed logging | Task 3 | `TreeConstructionLogger` |
| Visualization | Task 3 | `plot_decision_boundary_2d()`, `plot_hyperplanes_2d()` |



## Example: Comparing with Axis-Parallel Trees

```python
import numpy as np
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

# Generate diagonal boundary data
X, y = make_diagonal_dataset(n_samples=200, random_state=42)

# Train oblique tree
tree = ObliqueDecisionTree(max_depth=5)
tree.fit(X, y)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
print(f"Accuracy: {tree.score(X, y):.2%}")

# An axis-parallel tree would need more splits to approximate
# the diagonal boundary!
```

## License

See LICENSE file in the repository root.

## References

```bibtex
@inproceedings{murthy1992oc1,
  title={OC1: A randomized algorithm for building oblique decision trees},
  author={Murthy, Sreerama K and Kasif, Simon and Salzberg, Steven and Beigel, Richard},
  booktitle={Proceedings of the National Conference on Artificial Intelligence (AAAI)},
  pages={322--327},
  year={1992}
}
```
