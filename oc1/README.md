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
    ├── test_node.py
    ├── test_splits.py
    ├── test_hill_climb.py
    ├── test_tree.py
    └── test_integration.py
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
    n_restarts=1,             # Random restarts (1 = deterministic)
    random_state=None,        # Random seed
    impurity_threshold=0.0,   # Stop splitting threshold
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

## Running Tests

```bash
# Run all tests
pytest oc1/tests/ -v

# Run with coverage
pytest oc1/tests/ --cov=oc1 --cov-report=html

# Run specific test file
pytest oc1/tests/test_tree.py -v
```

## Paper Fidelity

This implementation follows the OC1 paper exactly:

| Feature | Paper Section | Implementation |
|---------|--------------|----------------|
| Hyperplane equation | Section 2 | `evaluate_hyperplane()` |
| Partition rule (V>0 → left) | Section 2 | `partition_data()` |
| Sum Minority impurity | Section 2.4 | `calculate_impurity()` |
| Max Minority impurity | Section 2.4 | `calculate_impurity()` |
| U_j formula (Eq. 1) | Section 2.2 | `compute_u_values()` |
| Sequential perturbation | Section 2.1 | `hill_climb()` |

## Compatibility with Future Tasks

This implementation is designed to integrate with:

- **Task 2 (Randomization)**: `n_restarts` parameter, modular `find_best_hyperplane()`
- **Task 3 (Pruning)**: `impurity_threshold` parameter, stored node impurities

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
