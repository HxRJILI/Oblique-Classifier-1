"""
OC1: Oblique Classifier 1 - Implementation of Murthy et al. (1992)

A Python implementation of the OC1 oblique decision tree algorithm as described in:
"OC1: A randomized algorithm for building oblique decision trees"
by Sreerama K. Murthy, Simon Kasif, Steven Salzberg, and Richard Beigel.
AAAI-1992.

This package provides:
- ObliqueTreeNode: Node representation with hyperplane coefficients
- ObliqueDecisionTree: Main classifier implementing the OC1 algorithm
- Impurity measures: Sum Minority (SM) and Max Minority (MM)
- Coefficient perturbation and hill-climbing optimization

Task 1 Implementation - Core Tree Construction (Deterministic)
"""

from oc1.core.node import ObliqueTreeNode
from oc1.core.tree import ObliqueDecisionTree
from oc1.core.splits import (
    partition_data,
    calculate_impurity,
    compute_class_counts,
    evaluate_hyperplane,
)
from oc1.core.hill_climb import (
    perturb_coefficient,
    hill_climb,
    initialize_hyperplane,
)

__version__ = "0.1.0"
__author__ = "OC1 Implementation Team"
__paper__ = "Murthy et al., OC1: A randomized algorithm for building oblique decision trees, AAAI-1992"

__all__ = [
    "ObliqueTreeNode",
    "ObliqueDecisionTree",
    "partition_data",
    "calculate_impurity",
    "compute_class_counts",
    "evaluate_hyperplane",
    "perturb_coefficient",
    "hill_climb",
    "initialize_hyperplane",
]
