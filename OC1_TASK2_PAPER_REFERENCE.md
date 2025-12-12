# OC1 Paper Reference Guide for Task 2 Implementation

📄 **"OC1: A randomized algorithm for building oblique decision trees"**  
Authors: Sreerama K. Murthy, Simon Kasif, Steven Salzberg (Johns Hopkins University), Richard Beigel (Yale University)  
Conference: AAAI-1992

---

## 🎯 TASK 2 CORE COMPONENTS - EXACT PAPER REFERENCES

### 1. MOTIVATION FOR RANDOMIZATION (Section 2.3)
```text
Problem: Deterministic hill-climbing gets stuck in local minima.

"The deterministic algorithm described in Section 2.1 
is susceptible to local minima in the impurity function."

Solution: Add randomization to explore more of the search space.
- Random hyperplane initialization
- Multiple random restarts (k trials)
- Perturbing multiple coefficients simultaneously
```

### 2. RANDOM HYPERPLANE INITIALIZATION (Section 2.3)
```text
Instead of starting with axis-parallel or fixed hyperplane:

1. Generate random coefficients for initial hyperplane
2. Each coefficient aᵢ drawn from uniform distribution
3. Different random seeds produce different starting points

Purpose: Explore diverse regions of hyperplane space
```

### 3. K RANDOM RESTARTS (Section 2.3 - KEY ALGORITHM)
```text
Algorithm with k random trials:

FOR trial = 1 to k:
   1. Generate RANDOM initial hyperplane H_random
   2. Apply hill-climbing optimization to H_random
   3. Record final hyperplane H_trial and its impurity
   
RETURN hyperplane with LOWEST impurity across all k trials

Paper recommendation: k should be small constant
Implementation: n_restarts parameter (default = 10)
```

### 4. MULTI-COEFFICIENT PERTURBATION (Section 2.3)
```text
Enhancement to escape local minima:

Standard (Task 1): Perturb ONE coefficient at a time
Enhanced (Task 2): Perturb MULTIPLE coefficients simultaneously

"perturbing more than one coefficient at a time...
can help the search escape from local minima"

Implementation: perturb_multiple_coefficients() function
- Randomly select subset of coefficients
- Apply perturbation to all selected coefficients
- Evaluate combined effect on impurity
```

### 5. RANDOM PERTURBATION ORDER (Section 2.1/2.3)
```text
In deterministic mode: Cycle through coefficients in fixed order
   FOR m = 1 to d+1: perturb(aₘ)

In randomized mode: Shuffle coefficient order each iteration
   indices = random_permutation([1, 2, ..., d+1])
   FOR m in indices: perturb(aₘ)

Purpose: Avoid bias from always starting with same coefficient
Implementation: use_random_perturbation_order=True
```

### 6. COMPLETE RANDOMIZED ALGORITHM (Section 2.3)
```text
At each internal node:

1. INITIALIZATION:
   - Generate k random hyperplanes as starting points
   
2. OPTIMIZATION (for each of k trials):
   a. Start with random hyperplane H
   b. Randomize coefficient perturbation order
   c. Apply hill-climbing with random order
   d. Optionally perturb multiple coefficients
   e. Record best hyperplane found
   
3. SELECTION:
   - Return hyperplane with minimum impurity across k trials
   
4. PARTITION:
   - Split data using best hyperplane
   - Recurse on left/right children
```

### 7. CONVERGENCE PROPERTIES (Section 2.3)
```text
"Our algorithm maintains the guarantee that the 
impurity measure decreases monotonically."

Key properties:
- Each trial still converges to a local minimum
- More trials = higher chance of finding global minimum
- Random restarts are independent (embarrassingly parallel)
- Expected quality improves with k
```

---

## 🔧 IMPLEMENTATION NOTES FOR TASK 2

### New Parameters:
```python
n_restarts: int = 10      # Number of random trials (k)
random_state: int = None  # Seed for reproducibility
use_random_perturbation_order: bool = True
```

### Key Functions Added:
```python
def find_best_hyperplane(X, y, n_restarts=10, random_state=None):
    """Run k random restarts and return best hyperplane."""
    best_hp, best_impurity = None, float('inf')
    
    for trial in range(n_restarts):
        # Random initialization
        hp = generate_random_hyperplane(n_features)
        
        # Hill-climbing with random perturbation order
        hp = hill_climb(X, y, hp, use_random_order=True)
        
        # Track best
        impurity = calculate_impurity(X, y, hp)
        if impurity < best_impurity:
            best_hp, best_impurity = hp, impurity
    
    return best_hp, best_impurity

def perturb_multiple_coefficients(X, y, hyperplane, n_coeffs=2):
    """Perturb multiple coefficients simultaneously."""
    # Select random subset of coefficients
    # Apply combined perturbation
    # Return improved hyperplane
```

### Relationship to Task 1:
```text
- Task 1 behavior: n_restarts=1, use_random_perturbation_order=False
- Task 2 behavior: n_restarts>1, use_random_perturbation_order=True

All Task 1 code remains valid - Task 2 extends functionality.
```

---

## 📊 EXPECTED IMPROVEMENTS (Section 3)

### Empirical Results from Paper:
```text
Randomization benefits:
- Better escape from local minima
- Smaller trees on average
- Higher classification accuracy
- More robust to initialization

Trade-off: k trials = k × computation time per node
Recommendation: k=10 balances quality vs speed
```

### Test Cases for Task 2:
```text
1. XOR dataset: Accuracy should improve significantly with k>1
2. Reproducibility: Same random_state → identical results
3. k=1 vs k=10: Higher k should find equal or better splits
4. Random order: Different seeds explore different solutions
5. Multi-coefficient: Should escape some local minima
```

---

## 🔗 CONNECTIONS TO TASK 3 (PRUNING)

```text
Task 2 randomization creates trees that may need pruning:
- More exploration → potentially deeper trees initially
- Pruning (Task 3) will simplify without losing accuracy
- impurity_threshold parameter already in place for Task 3
```

---

This document contains ALL Task 2 specifications from the OC1 paper.  
Reference Section 2.3 for randomization implementation fidelity.
