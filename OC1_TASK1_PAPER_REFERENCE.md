OC1 Paper Reference Guide for Task 1 Implementation
📄 "OC1: A randomized algorithm for building oblique decision trees"
Authors: Sreerama K. Murthy, Simon Kasif, Steven Salzberg (Johns Hopkins University), Richard Beigel (Yale University)
Conference: AAAI-1992​

🎯 TASK 1 CORE COMPONENTS - EXACT PAPER REFERENCES
1. HYPERPLANE DEFINITION (Section 2)
text
Equation of hyperplane H at each non-leaf node:
∑_{i=1 to d} (a_i * X_i) + a_{d+1} = 0

For example Pⱼ = (xⱼ¹, xⱼ², ..., xⱼᵈ):
Vⱼ = ∑_{i=1 to d} (a_i * xⱼⁱ) + a_{d+1}

Partition rule:
- Left: Vⱼ > 0  
- Right: Vⱼ ≤ 0
2. IMPURITY MEASURES (Section 2.4 - EXACT DEFINITIONS)
text
For partitions L and R:
- minority_L = min(count of each class in L)
- minority_R = min(count of each class in R)

Sum Minority (SM): SM(H) = minority_L + minority_R
Max Minority (MM): MM(H) = max(minority_L, minority_R)

LOWER impurity = BETTER split
Stop splitting when impurity = 0 (pure node)
3. COEFFICIENT PERTURBATION (Section 2.2 - Equation 1)
text
To optimize coefficient aₘ (fix others as constants):

For each example j:
Uⱼ = aₘ * xⱼᵐ - Vⱼ / xⱼᵐ     [EXACT Equation 1 from paper]

1. Compute {Uⱼ} for all examples
2. Find BEST univariate threshold on Uⱼ values
3. Set new aₘ = optimal_threshold
4. DETERMINISTIC HILL-CLIMBING (Section 2.1)
text
Algorithm at each node:
1. Start with initial hyperplane H
2. WHILE improvement possible:
   FOR m = 1 to d+1 (cycle through coefficients):
      a. Perturb aₘ using Equation 1 method
      b. Evaluate new impurity
      c. IF new_impurity < current_impurity:
         Accept new aₘ
3. Stop at local minimum (no improvements)
5. SEARCH STRATEGIES (Section 2.1)
text
Three deterministic perturbation cycles available:
- Seq: Sequential (one coefficient at a time)
- Best: Best coefficient first  
- R-50: Random order of 50 coefficients

OC1 uses these in combination with randomization (Task 2)
6. TREE CONSTRUCTION (Section 2)
text
Recursive algorithm:
- At each node with impure samples:
  1. Find best hyperplane H using hill-climbing
  2. Partition data into L and R using H
  3. Recursively build left/right subtrees
- STOP SPLITTING when:
  - Zero impurity (all samples same class) [Sec 2.4]
  - OR minimum samples threshold reached

Pruning: Cut subtrees where impurity < threshold [Sec 1]
7. COMPLEXITY & RESTRICTIONS
text
- OC1 recognizes only O(nᵈ) distinct hyperplanes
  (cannot distinguish hyperplanes with identical partitions)
- Polynomial time per node guaranteed
- Numeric attributes only (symbolic handled separately)
🔧 IMPLEMENTATION NOTES FOR TASK 1
Data Structures Required:
python
class ObliqueTreeNode:
    hyperplane: np.ndarray  # [a1, a2, ..., ad, a_{d+1}]
    class_distribution: dict
    left_child: ObliqueTreeNode | None
    right_child: ObliqueTreeNode | None
    is_leaf: bool
    predicted_class: Any
Key Functions (modular for Tasks 2+3):
python
def partition_data(X: np.ndarray, y: np.ndarray, hyperplane: np.ndarray) -> tuple
def calculate_impurity(class_counts_L: dict, class_counts_R: dict) -> tuple[float, float]
def perturb_coefficient(X: np.ndarray, y: np.ndarray, hyperplane: np.ndarray, m: int) -> np.ndarray
def hill_climb(X: np.ndarray, y: np.ndarray, hyperplane: np.ndarray, max_iters: int = 10) -> np.ndarray
Test Cases (validate paper fidelity):
text
1. 2D synthetic data requiring 45° oblique split
2. Verify Equation 1: Uⱼ calculation matches paper
3. Hill-climbing converges to local minimum
4. SM/MM impurities match manual calculations
5. Multi-class (3+ classes) minority calculations
6. Degenerate cases: all points on hyperplane
📊 EXAMPLE FROM PAPER (Section 3 Experiments)
text
OC1 produces smaller trees than C4.5 on real datasets:
- Glass: 15 nodes vs 27 nodes
- Hepatitis: 8 nodes vs 19 nodes
- Ionosphere: 12 nodes vs 31 nodes
This document contains ALL Task 1 specifications from the OC1 paper. Reference sections 2.1, 2.2, 2.4 for implementation fidelity.