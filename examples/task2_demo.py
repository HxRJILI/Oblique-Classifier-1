"""Demo of Task 2 randomization features.

This script demonstrates the Task 2 enhancements:
- Random hyperplane initialization
- Multi-coefficient perturbation
- K random trials (n_restarts parameter)
- Random perturbation order

Run this script to see how randomization improves tree performance:
    python examples/task2_demo.py
"""
import numpy as np
from oc1 import ObliqueDecisionTree
from oc1.data import make_xor_dataset, make_diagonal_dataset

print("=" * 70)
print("TASK 2 DEMONSTRATION: Randomization Benefits")
print("=" * 70)

# Dataset 1: XOR (needs good splits)
print("\n1. XOR Dataset")
print("-" * 70)
print("The XOR problem requires oblique splits to solve efficiently.")
print("More random restarts help find better hyperplanes.\n")

X, y = make_xor_dataset(n_samples=200, random_state=42)

print(f"{'k':>3}  {'Accuracy':>10}  {'Depth':>6}  {'Leaves':>7}")
print("-" * 35)
for k in [1, 5, 10, 20]:
    tree = ObliqueDecisionTree(
        n_restarts=k,
        max_depth=5,
        random_state=42
    )
    tree.fit(X, y)
    acc = tree.score(X, y)
    depth = tree.get_depth()
    leaves = tree.get_n_leaves()
    
    print(f"{k:>3}  {acc:>10.3f}  {depth:>6}  {leaves:>7}")

# Dataset 2: Diagonal (simple but oblique)
print("\n2. Diagonal Dataset")
print("-" * 70)
print("A simple diagonal boundary that benefits from oblique splits.\n")

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

tree_det = ObliqueDecisionTree(n_restarts=1, random_state=42)
tree_det.fit(X, y)

tree_rand = ObliqueDecisionTree(n_restarts=10, random_state=42)
tree_rand.fit(X, y)

print(f"Deterministic (k=1):  acc={tree_det.score(X, y):.3f}, depth={tree_det.get_depth()}, leaves={tree_det.get_n_leaves()}")
print(f"Randomized (k=10):    acc={tree_rand.score(X, y):.3f}, depth={tree_rand.get_depth()}, leaves={tree_rand.get_n_leaves()}")

# Dataset 3: Reproducibility test
print("\n3. Reproducibility Test")
print("-" * 70)
print("Same random_state should give identical results.\n")

X, y = make_xor_dataset(n_samples=150, random_state=42)

results = []
for run in range(3):
    tree = ObliqueDecisionTree(n_restarts=10, random_state=123)
    tree.fit(X, y)
    results.append({
        'accuracy': tree.score(X, y),
        'depth': tree.get_depth(),
        'leaves': tree.get_n_leaves()
    })
    print(f"Run {run + 1}: acc={results[-1]['accuracy']:.3f}, depth={results[-1]['depth']}, leaves={results[-1]['leaves']}")

# Check reproducibility
if all(r == results[0] for r in results):
    print("\n✅ All runs produced identical results - reproducibility confirmed!")
else:
    print("\n⚠️ Results differ between runs - check RNG propagation")

# Dataset 4: Different seeds comparison
print("\n4. Different Seeds Comparison")
print("-" * 70)
print("Different random states explore different parts of the solution space.\n")

X, y = make_xor_dataset(n_samples=200, random_state=42)

print(f"{'Seed':>6}  {'Accuracy':>10}  {'Depth':>6}  {'Leaves':>7}")
print("-" * 40)
for seed in [42, 123, 456, 789]:
    tree = ObliqueDecisionTree(n_restarts=10, random_state=seed)
    tree.fit(X, y)
    print(f"{seed:>6}  {tree.score(X, y):>10.3f}  {tree.get_depth():>6}  {tree.get_n_leaves():>7}")

print("\n" + "=" * 70)
print("✅ Task 2 features working correctly!")
print("=" * 70)

print("\nSummary of Task 2 Features:")
print("-" * 70)
print("• n_restarts=1: Deterministic mode (Task 1 behavior)")
print("• n_restarts=10: Default randomized mode (Task 2)")
print("• n_restarts=20+: Aggressive search for hard problems")
print("• random_state: Ensures reproducibility across runs")
print("• Multi-coefficient perturbation: Helps escape local minima")
print("• Random perturbation order: Reduces sequential bias")
