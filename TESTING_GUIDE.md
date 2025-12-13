# Guide de Test pour la Tâche 3

Ce guide explique comment tester toutes les fonctionnalités de la Tâche 3.

## Tests Rapides

### 1. Test de base (recommandé pour commencer)

```bash
python test_task3.py
```

Ce script teste rapidement toutes les fonctionnalités principales :
- Imports des modules
- Construction d'arbre
- Prédictions
- Élagage (impureté et REP)
- Validation croisée
- Métriques d'évaluation
- Journalisation

### 2. Tests unitaires complets

```bash
# Tous les tests de la Tâche 3
pytest oc1/tests/task3_tests/ -v

# Tests spécifiques
pytest oc1/tests/task3_tests/test_pruning.py -v
pytest oc1/tests/task3_tests/test_evaluation.py -v
pytest oc1/tests/task3_tests/test_logging.py -v

# Tous les tests (Task 1, 2, 3)
pytest oc1/tests/ -v

# Avec couverture de code
pytest oc1/tests/ --cov=oc1 --cov-report=html
```

## Tests Manuels par Fonctionnalité

### Test 1: Élagage par Impureté

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

# Générer des données
X, y = make_diagonal_dataset(n_samples=200, random_state=42)

# Construire un arbre
tree = ObliqueDecisionTree(max_depth=10, random_state=42)
tree.fit(X, y)

print(f"Avant élagage: {tree.get_n_nodes()} nœuds")

# Élaguer
tree.prune(method="impurity", impurity_threshold=2.0)

print(f"Après élagage: {tree.get_n_nodes()} nœuds")
print(f"Précision: {tree.score(X, y):.3f}")
```

### Test 2: Élagage REP (Reduced Error Pruning)

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import train_test_split
from oc1.data import make_diagonal_dataset

# Générer et diviser les données
X, y = make_diagonal_dataset(n_samples=200, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Construire un arbre
tree = ObliqueDecisionTree(max_depth=10, random_state=42)
tree.fit(X_train, y_train)

print(f"Avant élagage: {tree.get_n_nodes()} nœuds")
print(f"Précision validation: {tree.score(X_val, y_val):.3f}")

# Élaguer avec REP
tree.prune(method="rep", X_val=X_val, y_val=y_val)

print(f"Après élagage: {tree.get_n_nodes()} nœuds")
print(f"Précision validation: {tree.score(X_val, y_val):.3f}")
```

### Test 3: Validation Croisée

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import cross_validate
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=200, random_state=42)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)

# 5-fold cross-validation
results = cross_validate(tree, X, y, cv=5, random_state=42)

print(f"Précision moyenne: {results['test_score'].mean():.3f}")
print(f"Écart-type: {results['test_score'].std():.3f}")
print(f"Temps d'entraînement moyen: {results['fit_time'].mean():.3f}s")
```

### Test 4: Métriques d'Évaluation

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import (
    train_test_split,
    confusion_matrix,
    classification_report
)
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=200, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion:")
print(cm)

# Rapport de classification
report = classification_report(y_test, y_pred)
print("\nRapport de classification:")
print(report)
```

### Test 5: Journalisation

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

# Journalisation verbose (console)
tree = ObliqueDecisionTree(
    max_depth=4,
    verbose=True,  # Active les logs dans la console
    random_state=42
)
tree.fit(X, y)

# Obtenir le résumé des logs
summary = tree.logger.get_log_summary()
print(f"\nRésumé:")
print(f"  Événements totaux: {summary['total_events']}")
print(f"  Nœuds créés: {summary['nodes_created']}")
print(f"  Hyperplans trouvés: {summary['hyperplanes_found']}")

# Journalisation dans un fichier
tree2 = ObliqueDecisionTree(
    max_depth=4,
    log_file="tree.log",  # Écrit dans un fichier
    verbose=False,
    random_state=42
)
tree2.fit(X, y)
print("\nLogs écrits dans tree.log")
```

### Test 6: Visualisation (nécessite matplotlib)

```python
from oc1 import ObliqueDecisionTree
from oc1.visualization import plot_decision_boundary_2d, plot_hyperplanes_2d
from oc1.data import make_diagonal_dataset
import matplotlib.pyplot as plt

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

tree = ObliqueDecisionTree(max_depth=5, random_state=42)
tree.fit(X, y)

# Visualiser la frontière de décision
plot_decision_boundary_2d(tree, X, y)
plt.title("Frontière de Décision")
plt.show()

# Visualiser les hyperplans
plot_hyperplanes_2d(tree, X)
plt.title("Hyperplans de Division")
plt.show()
```

## Démonstration Complète

Pour voir toutes les fonctionnalités en action :

```bash
python examples/task3_demo.py
```

Ce script démontre :
- Élagage (impureté et REP)
- Validation croisée
- Métriques d'évaluation
- Journalisation
- Visualisation (si matplotlib est installé)

## Vérification des Dépendances

### Dépendances de base (requises)
```bash
pip install numpy
```

### Dépendances pour les tests
```bash
pip install pytest pytest-cov
```

### Dépendances pour la visualisation (optionnel)
```bash
pip install matplotlib networkx
```

Ou installez tout d'un coup :
```bash
pip install -r requirements.txt
pip install matplotlib networkx  # Pour la visualisation
```

## Résolution de Problèmes

### Erreur d'import
Si vous obtenez `ModuleNotFoundError`:
```bash
# Installez le package en mode développement
pip install -e .
```

### Erreur de visualisation
Si matplotlib n'est pas disponible, les fonctions de visualisation ne fonctionneront pas, mais le reste du code fonctionnera normalement.

### Tests qui échouent
Vérifiez que vous avez la bonne version de Python (>= 3.8) et que toutes les dépendances sont installées :
```bash
python --version
pip list | grep -E "(numpy|pytest)"
```

## Tests de Performance

Pour tester les performances sur différents datasets :

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_xor_dataset, make_diagonal_dataset
from oc1.evaluation import cross_validate

datasets = {
    "diagonal": make_diagonal_dataset(n_samples=200, random_state=42),
    "xor": make_xor_dataset(n_samples=200, random_state=42),
}

for name, (X, y) in datasets.items():
    tree = ObliqueDecisionTree(max_depth=5, random_state=42)
    results = cross_validate(tree, X, y, cv=5, random_state=42)
    print(f"{name}: {results['test_score'].mean():.3f} ± {results['test_score'].std():.3f}")
```

## Checklist de Test

- [ ] Test rapide (`python test_task3.py`) passe
- [ ] Tests unitaires (`pytest oc1/tests/task3_tests/`) passent
- [ ] Élagage par impureté fonctionne
- [ ] Élagage REP fonctionne
- [ ] Validation croisée fonctionne
- [ ] Métriques d'évaluation fonctionnent
- [ ] Journalisation verbose fonctionne
- [ ] Journalisation fichier fonctionne
- [ ] Visualisation fonctionne (si matplotlib installé)
- [ ] Démonstration complète fonctionne

