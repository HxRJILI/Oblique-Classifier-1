# OC1: Oblique Classifier 1

[![PyPI version](https://img.shields.io/pypi/v/oblique-classifier-1.svg)](https://pypi.org/project/oblique-classifier-1/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-236%20passed-green.svg)]()


Project Presentation: https://www.canva.com/design/DAG7t5nKUvA/BfUUuEFykRJ90x7hd12dAQ/edit?utm_content=DAG7t5nKUvA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

A Python implementation of the **OC1 oblique decision tree algorithm** as described in:

> **"OC1: A randomized algorithm for building oblique decision trees"**  
> Sreerama K. Murthy, Simon Kasif, Steven Salzberg, and Richard Beigel  
> *AAAI-1992*

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install oblique-classifier-1
```

### With Visualization Support

```bash
pip install oblique-classifier-1[viz]
```

### From Source (Development)

```bash
git clone https://github.com/HxRJILI/Oblique-Classifier-1.git
cd Oblique-Classifier-1
pip install -e .
```

---

## 🚀 Quick Start

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
tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print(f"Predictions: {predictions}")
print(f"Accuracy: {tree.score(X, y):.2%}")
```

---

## 📖 Overview

OC1 (Oblique Classifier 1) builds decision trees using **oblique hyperplanes** (linear combinations of attributes) rather than axis-parallel splits. This often results in **smaller and more accurate trees** than traditional methods like C4.5 or CART.

### Why Oblique Trees?

```
Axis-Parallel Tree (many splits)     Oblique Tree (one split)
        |                                    \
   -----+-----                                \
        |                                      \
   -----+-----           vs                     \
        |                                        \
   -----+-----                                    \
```

**Benefits:**
- ✅ **Smaller trees** (fewer nodes)
- ✅ **Better accuracy** on diagonal decision boundaries
- ✅ **More interpretable** for linear relationships

### Key Features

- **Oblique Splits**: Uses hyperplanes of the form `∑(aᵢxᵢ) + a_{d+1} = 0`
- **Hill-Climbing Optimization**: Sequential coefficient perturbation with local search
- **Randomization**: Multiple random restarts to escape local minima
- **Impurity Measures**: Sum Minority (SM) and Max Minority (MM)
- **Pruning**: Impurity-based and Reduced Error Pruning (REP)
- **Evaluation Tools**: Cross-validation, confusion matrix, classification report
- **Visualization**: Decision boundary and hyperplane plotting (requires matplotlib)
- **Export Methods**: JSON, dictionary, and DOT format for tree visualization

---

## 👥 Team Members

| GitHub Username | Name |
|-----------------|------|
| **HxRJILI** | RJILI Houssam |
| **Kim8x-srscb** | Fatima-Ezzahrae AKEBLI |
| **Yasseriads** | Yasser |

---

# 📚 API Documentation

## ObliqueDecisionTree

The main classifier class implementing the OC1 algorithm.

### Constructor

```python
from oc1 import ObliqueDecisionTree

tree = ObliqueDecisionTree(
    max_depth=None,           # Maximum tree depth (None = unlimited)
    min_samples_leaf=1,       # Minimum samples required in a leaf node
    min_samples_split=2,      # Minimum samples required to split a node
    impurity_measure="sm",    # Impurity measure: "sm" or "mm"
    max_iterations=100,       # Maximum hill-climbing iterations per node
    n_restarts=10,            # Number of random restarts (1 = deterministic)
    random_state=None,        # Random seed for reproducibility
    impurity_threshold=0.0,   # Stop splitting if impurity below this
    verbose=False,            # Enable verbose logging
    log_file=None,            # Optional log file path
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int` or `None` | `None` | Maximum depth of the tree. `None` for unlimited depth. |
| `min_samples_leaf` | `int` | `1` | Minimum number of samples required to be at a leaf node. |
| `min_samples_split` | `int` | `2` | Minimum number of samples required to split an internal node. |
| `impurity_measure` | `str` | `"sm"` | Impurity measure: `"sm"` (Sum Minority) or `"mm"` (Max Minority). |
| `max_iterations` | `int` | `100` | Maximum number of hill-climbing iterations per node. |
| `n_restarts` | `int` | `10` | Number of random restarts for hyperplane optimization. Use `1` for deterministic. |
| `random_state` | `int` or `None` | `None` | Random seed for reproducibility. |
| `impurity_threshold` | `float` | `0.0` | Stop splitting if node impurity falls below this threshold. |
| `verbose` | `bool` | `False` | Enable verbose logging during tree construction. |
| `log_file` | `str` or `None` | `None` | Path to write detailed construction logs. |

---

### Methods

#### `fit(X, y)`

Build the oblique decision tree from training data.

```python
tree.fit(X, y)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Training feature matrix of shape `(n_samples, n_features)` |
| `y` | `np.ndarray` | Training labels of shape `(n_samples,)` |

**Returns:** `self` (the fitted classifier)

---

#### `predict(X)`

Predict class labels for samples.

```python
predictions = tree.predict(X)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Feature matrix of shape `(n_samples, n_features)` |

**Returns:** `np.ndarray` of predicted class labels

---

#### `predict_proba(X)`

Predict class probabilities for samples.

```python
probabilities = tree.predict_proba(X)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Feature matrix of shape `(n_samples, n_features)` |

**Returns:** `np.ndarray` of shape `(n_samples, n_classes)` with class probabilities

---

#### `score(X, y)`

Calculate the classification accuracy.

```python
accuracy = tree.score(X, y)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Feature matrix |
| `y` | `np.ndarray` | True class labels |

**Returns:** `float` accuracy score (0.0 to 1.0)

---

#### `prune(X_val, y_val, method, impurity_threshold)`

Prune the tree to reduce overfitting.

```python
# Reduced Error Pruning (requires validation set)
tree.prune(X_val=X_val, y_val=y_val, method="rep")

# Impurity-based pruning
tree.prune(method="impurity", impurity_threshold=2.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_val` | `np.ndarray` | `None` | Validation features (required for `"rep"`) |
| `y_val` | `np.ndarray` | `None` | Validation labels (required for `"rep"`) |
| `method` | `str` | `"rep"` | Pruning method: `"rep"` or `"impurity"` |
| `impurity_threshold` | `float` | `None` | Threshold for impurity-based pruning |

**Returns:** `self` (the pruned tree)

---

#### `get_depth()`

Get the maximum depth of the tree.

```python
depth = tree.get_depth()
```

**Returns:** `int` maximum depth (0 if tree is just a leaf)

---

#### `get_n_leaves()`

Get the number of leaf nodes.

```python
n_leaves = tree.get_n_leaves()
```

**Returns:** `int` number of leaf nodes

---

#### `get_n_nodes()`

Get the total number of nodes.

```python
n_nodes = tree.get_n_nodes()
```

**Returns:** `int` total number of nodes

---

#### `get_hyperplanes()`

Get all splitting hyperplanes in the tree.

```python
hyperplanes = tree.get_hyperplanes()
```

**Returns:** `List[np.ndarray]` list of hyperplane coefficient arrays

---

#### `get_all_nodes()`

Get all nodes in breadth-first order.

```python
nodes = tree.get_all_nodes()
leaves = [n for n in nodes if n.is_leaf]
```

**Returns:** `List[ObliqueTreeNode]` all nodes

---

#### `print_tree()`

Get a string representation of the tree structure.

```python
print(tree.print_tree())
```

**Returns:** `str` tree visualization

---

#### `to_dict()`

Export tree structure as a dictionary.

```python
tree_dict = tree.to_dict()
```

**Returns:** `Dict` containing tree parameters, statistics, and structure

---

#### `to_json(filepath, indent)`

Export tree structure to JSON.

```python
# Get JSON string
json_str = tree.to_json()

# Save to file
tree.to_json(filepath="model.json", indent=2)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | `None` | Optional path to save JSON file |
| `indent` | `int` | `2` | JSON indentation level |

**Returns:** `str` JSON representation

---

#### `to_dot(feature_names)`

Export tree structure to DOT format for Graphviz visualization.

```python
dot_str = tree.to_dot(feature_names=["age", "income"])

# Save and render with Graphviz
with open("tree.dot", "w") as f:
    f.write(dot_str)
# Then run: dot -Tpng tree.dot -o tree.png
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_names` | `List[str]` | `None` | Optional feature names for labels |

**Returns:** `str` DOT format string

---

### Properties

#### `feature_importances_`

Compute feature importances based on hyperplane coefficients.

```python
importances = tree.feature_importances_

for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.3f}")
```

**Returns:** `np.ndarray` normalized feature importance scores (sum to 1.0)

---

## Evaluation Functions

### `train_test_split(X, y, test_size, random_state, stratify)`

Split data into training and test sets.

```python
from oc1.evaluation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | - | Feature matrix |
| `y` | `np.ndarray` | - | Labels |
| `test_size` | `float` | `0.2` | Proportion for testing (0.0 to 1.0) |
| `random_state` | `int` | `None` | Random seed |
| `stratify` | `bool` | `True` | Maintain class distribution |

**Returns:** `Tuple[X_train, X_test, y_train, y_test]`

---

### `cross_validate(estimator, X, y, cv, scoring, random_state, return_train_score)`

Perform k-fold cross-validation.

```python
from oc1.evaluation import cross_validate

results = cross_validate(
    tree, X, y,
    cv=5,
    scoring="accuracy",
    random_state=42,
    return_train_score=True,
)

print(f"Mean accuracy: {results['test_score'].mean():.3f} ± {results['test_score'].std():.3f}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | classifier | - | Classifier with `fit()` and `score()` methods |
| `X` | `np.ndarray` | - | Feature matrix |
| `y` | `np.ndarray` | - | Labels |
| `cv` | `int` | `5` | Number of folds |
| `scoring` | `str` | `"accuracy"` | Metric: `"accuracy"`, `"precision"`, `"recall"`, `"f1"` |
| `random_state` | `int` | `None` | Random seed |
| `return_train_score` | `bool` | `False` | Include training scores |

**Returns:** `Dict` with keys `test_score`, `fit_time`, `score_time`, and optionally `train_score`

---

### `confusion_matrix(y_true, y_pred, labels)`

Compute confusion matrix.

```python
from oc1.evaluation import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
# cm[i, j] = samples with true class i predicted as class j
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `np.ndarray` | - | True class labels |
| `y_pred` | `np.ndarray` | - | Predicted class labels |
| `labels` | `np.ndarray` | `None` | Class labels to include |

**Returns:** `np.ndarray` of shape `(n_classes, n_classes)`

---

### `classification_report(y_true, y_pred, labels, target_names)`

Generate a detailed classification report.

```python
from oc1.evaluation import classification_report

report = classification_report(y_true, y_pred)
print(report)
```

**Output:**
```
Classification Report
==================================================

Class           Precision    Recall       F1-Score     Support   
-----------------------------------------------------------------
0               0.9500       0.9000       0.9244       100       
1               0.9100       0.9600       0.9344       100       
-----------------------------------------------------------------
Macro Avg       0.9300       0.9300       0.9294       200       
Weighted Avg    0.9300       0.9300       0.9294       200       
```

---

## Synthetic Datasets

```python
from oc1.data import (
    make_diagonal_dataset,       # 45° diagonal boundary
    make_xor_dataset,            # XOR problem (requires oblique splits)
    make_oblique_classification, # Custom angle boundary
    make_multiclass_oblique,     # Multi-class sectors
    make_3d_oblique,             # 3D oblique plane
)
```

### `make_diagonal_dataset(n_samples, noise, random_state)`

Create dataset with 45° diagonal decision boundary.

```python
X, y = make_diagonal_dataset(n_samples=200, noise=0.1, random_state=42)
```

### `make_xor_dataset(n_samples, noise, random_state)`

Create XOR dataset (classic non-linearly-separable problem).

```python
X, y = make_xor_dataset(n_samples=200, random_state=42)
```

### `make_oblique_classification(n_samples, angle, random_state)`

Create dataset with custom angle boundary.

```python
X, y = make_oblique_classification(n_samples=200, angle=30, random_state=42)
```

---

## Visualization (requires matplotlib)

```python
from oc1.visualization import plot_decision_boundary_2d, plot_hyperplanes_2d
import matplotlib.pyplot as plt

# Plot decision boundary
plot_decision_boundary_2d(tree, X, y)
plt.title("Decision Boundary")
plt.show()

# Plot hyperplanes
plot_hyperplanes_2d(tree, X)
plt.title("Hyperplanes")
plt.show()
```

---

# 📋 Usage Examples

## Basic Classification

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset
from oc1.evaluation import train_test_split

# Generate data
X, y = make_diagonal_dataset(n_samples=500, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train tree
tree = ObliqueDecisionTree(max_depth=5, n_restarts=10, random_state=42)
tree.fit(X_train, y_train)

# Evaluate
print(f"Train accuracy: {tree.score(X_train, y_train):.2%}")
print(f"Test accuracy: {tree.score(X_test, y_test):.2%}")
print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
```

## Cross-Validation

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import cross_validate
from oc1.data import make_xor_dataset

X, y = make_xor_dataset(n_samples=300, random_state=42)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
results = cross_validate(tree, X, y, cv=5, random_state=42)

print(f"CV Accuracy: {results['test_score'].mean():.3f} ± {results['test_score'].std():.3f}")
```

## Pruning

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import train_test_split
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=500, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# Build full tree
tree = ObliqueDecisionTree(max_depth=15, random_state=42)
tree.fit(X_train, y_train)
print(f"Before pruning: {tree.get_n_leaves()} leaves, {tree.score(X_val, y_val):.2%} accuracy")

# Prune with REP
tree.prune(X_val=X_val, y_val=y_val, method="rep")
print(f"After pruning: {tree.get_n_leaves()} leaves, {tree.score(X_val, y_val):.2%} accuracy")
```

## Export & Save Model

```python
from oc1 import ObliqueDecisionTree

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X, y)

# Export to JSON
tree.to_json("model.json")

# Export to DOT (for Graphviz)
dot_str = tree.to_dot(feature_names=["feature_1", "feature_2"])
with open("tree.dot", "w") as f:
    f.write(dot_str)

# Export as dictionary
model_dict = tree.to_dict()
```

## Feature Importances

```python
from oc1 import ObliqueDecisionTree

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X, y)

importances = tree.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

## Verbose Logging

```python
from oc1 import ObliqueDecisionTree

# Console logging
tree = ObliqueDecisionTree(max_depth=5, verbose=True, random_state=42)
tree.fit(X, y)

# File logging
tree = ObliqueDecisionTree(max_depth=5, log_file="training.log", random_state=42)
tree.fit(X, y)

# Access log summary
summary = tree.logger.get_log_summary()
print(f"Nodes created: {summary['nodes_created']}")
```

---

# 🔬 Algorithm Details

## Hyperplane Representation

Each internal node contains a hyperplane defined by:

$$\sum_{i=1}^{d} a_i x_i + a_{d+1} = 0$$

Where:
- $a_i$ are the hyperplane coefficients
- $x_i$ are the feature values
- $a_{d+1}$ is the bias term

## Partitioning Rule

For a sample $x_j$, compute:

$$V_j = \sum_{i=1}^{d} a_i x_j^i + a_{d+1}$$

- **Left child**: $V_j > 0$
- **Right child**: $V_j \leq 0$

## Impurity Measures

### Sum Minority (SM)
$$SM = minority_L + minority_R$$

### Max Minority (MM)
$$MM = \max(minority_L, minority_R)$$

Where $minority$ is the count of samples not belonging to the majority class.

## Hill-Climbing Optimization

The algorithm optimizes hyperplane coefficients using **sequential perturbation** (Equation 1 from the paper):

$$U_j = a_m x_j^m - \frac{V_j}{x_j^m}$$

This transforms the multi-dimensional optimization into a series of 1D threshold searches.

---

# 📁 Project Structure

```
oblique-classifier-1/
├── oc1/
│   ├── __init__.py              # Package exports
│   ├── core/
│   │   ├── node.py              # ObliqueTreeNode class
│   │   ├── tree.py              # ObliqueDecisionTree classifier
│   │   ├── splits.py            # Impurity measures, partitioning
│   │   ├── hill_climb.py        # Coefficient perturbation algorithm
│   │   └── logging.py           # TreeConstructionLogger
│   ├── data/
│   │   └── datasets.py          # Synthetic test datasets
│   ├── evaluation/
│   │   └── __init__.py          # Evaluation tools
│   └── visualization/
│       └── __init__.py          # Plotting utilities
├── examples/
│   ├── task2_demo.py            # Randomization demo
│   └── task3_demo.py            # Pruning & evaluation demo
└── tests/                       # 236 unit tests
from oc1.visualization import (
    plot_decision_boundary_2d,  # Plot 2D decision regions
    plot_hyperplanes_2d,        # Plot all hyperplanes
)
```

---

## Usage Examples

### Pruning

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import train_test_split
from oc1.data import make_diagonal_dataset

# Generate data
X, y = make_diagonal_dataset(n_samples=200, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Build tree
tree = ObliqueDecisionTree(max_depth=10, random_state=42)
tree.fit(X_train, y_train)

# Prune using Reduced Error Pruning
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
    random_state=42
)
tree.fit(X, y)

# Get log summary
summary = tree.logger.get_log_summary()
print(f"Nodes created: {summary['nodes_created']}")
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

### Export Tree

```python
from oc1 import ObliqueDecisionTree
import json

tree = ObliqueDecisionTree(max_depth=3, random_state=42)
tree.fit(X, y)

# Export as dictionary
tree_dict = tree.to_dict()

# Export as JSON
tree_json = tree.to_json(indent=2)
with open("tree.json", "w") as f:
    f.write(tree_json)

# Export as DOT (for Graphviz)
dot_string = tree.to_dot()
with open("tree.dot", "w") as f:
    f.write(dot_string)
```

---

## Project Structure

```
Oblique-Classifier-1/
├── oc1/
│   ├── __init__.py              # Package exports
│   ├── core/
│   │   ├── node.py              # ObliqueTreeNode class
│   │   ├── tree.py              # ObliqueDecisionTree classifier
│   │   ├── splits.py            # Impurity measures, partitioning
│   │   ├── hill_climb.py        # Coefficient perturbation
│   │   └── logging.py           # TreeConstructionLogger
│   ├── data/
│   │   └── datasets.py          # Synthetic test datasets
│   ├── evaluation/
│   │   └── __init__.py          # Evaluation tools
│   ├── visualization/
│   │   └── __init__.py          # Plotting utilities
│   └── tests/
│       ├── task1_tests/         # Core tree construction tests
│       ├── task2_tests/         # Randomization tests
│       └── task3_tests/         # Pruning & evaluation tests
├── examples/
│   ├── task2_demo.py            # Task 2 demonstration
│   └── task3_demo.py            # Task 3 demonstration
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.py                     # Package installation
└── TESTING_GUIDE.md             # Testing documentation
```

---

## Running Tests

```bash
# Run all tests (236 tests)
pytest oc1/tests/ -v

# Run specific task tests
pytest oc1/tests/task1_tests/ -v  # Core tree construction
pytest oc1/tests/task2_tests/ -v  # Randomization
pytest oc1/tests/task3_tests/ -v  # Pruning & evaluation

# Run with coverage report
pytest oc1/tests/ --cov=oc1 --cov-report=html

# Run specific test file
pytest oc1/tests/task1_tests/test_tree.py -v
```

---

# 📄 Paper Reference

This implementation follows the original OC1 paper:

```bibtex
@inproceedings{murthy1992oc1,
  title={OC1: A randomized algorithm for building oblique decision trees},
  author={Murthy, Sreerama K and Kasif, Simon and Salzberg, Steven and Beigel, Richard},
  booktitle={Proceedings of the National Conference on Artificial Intelligence (AAAI)},
  pages={322--327},
  year={1992}
}
```

---

# 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# 🔗 Links

- **PyPI**: https://pypi.org/project/oblique-classifier-1/
- **GitHub**: https://github.com/HxRJILI/Oblique-Classifier-1
- **Issues**: https://github.com/HxRJILI/Oblique-Classifier-1/issues
