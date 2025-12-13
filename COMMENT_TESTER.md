# Comment Tester la Tâche 3

## ✅ Test Rapide (Recommandé)

Le moyen le plus simple de vérifier que tout fonctionne :

```bash
python test_task3.py
```

Ce script teste automatiquement toutes les fonctionnalités principales. Si tous les tests passent (✓), tout fonctionne correctement !

## 📋 Méthodes de Test Disponibles

### 1. Test Automatique Simple

```bash
python test_task3.py
```

**Résultat attendu :** Tous les tests doivent afficher ✓

### 2. Démonstration Complète

```bash
python examples/task3_demo.py
```

Affiche une démonstration de toutes les fonctionnalités avec des exemples concrets.

### 3. Tests Unitaires (si pytest est installé)

```bash
# Installer pytest d'abord
pip install pytest

# Puis exécuter les tests
pytest oc1/tests/task3_tests/ -v
```

### 4. Test Manuel Interactif

Ouvrez Python et testez directement :

```python
# Test basique
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=100, random_state=42)
tree = ObliqueDecisionTree(max_depth=3, verbose=True, random_state=42)
tree.fit(X, y)

# Vérifier que ça fonctionne
print(f"Précision: {tree.score(X, y):.3f}")
print(f"Nœuds: {tree.get_n_nodes()}")
print(f"Feuilles: {tree.get_n_leaves()}")

# Tester l'élagage
tree.prune(method="impurity", impurity_threshold=2.0)
print(f"Nœuds après élagage: {tree.get_n_nodes()}")
```

## 🎯 Checklist de Vérification

Vérifiez que ces fonctionnalités fonctionnent :

- [x] **Construction d'arbre** - `tree.fit(X, y)` fonctionne
- [x] **Prédictions** - `tree.predict(X)` retourne des prédictions valides
- [x] **Élagage par impureté** - `tree.prune(method="impurity")` réduit le nombre de nœuds
- [x] **Élagage REP** - `tree.prune(method="rep", X_val=X_val, y_val=y_val)` fonctionne
- [x] **Validation croisée** - `cross_validate()` retourne des résultats
- [x] **Métriques** - `confusion_matrix()` et `classification_report()` fonctionnent
- [x] **Journalisation** - Les logs s'affichent avec `verbose=True`

## 🔍 Tests Spécifiques par Fonctionnalité

### Test de l'Élagage

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=200, random_state=42)
tree = ObliqueDecisionTree(max_depth=10, random_state=42)
tree.fit(X, y)

print(f"Avant: {tree.get_n_nodes()} nœuds")
tree.prune(method="impurity", impurity_threshold=2.0)
print(f"Après: {tree.get_n_nodes()} nœuds")
# Le nombre de nœuds devrait diminuer ou rester égal
```

### Test de la Validation Croisée

```python
from oc1 import ObliqueDecisionTree
from oc1.evaluation import cross_validate
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=200, random_state=42)
tree = ObliqueDecisionTree(max_depth=5, random_state=42)

results = cross_validate(tree, X, y, cv=5, random_state=42)
print(f"Précision: {results['test_score'].mean():.3f}")
# Devrait afficher un nombre entre 0 et 1
```

### Test de la Journalisation

```python
from oc1 import ObliqueDecisionTree
from oc1.data import make_diagonal_dataset

X, y = make_diagonal_dataset(n_samples=100, random_state=42)

# Avec verbose=True, vous devriez voir des logs
tree = ObliqueDecisionTree(max_depth=3, verbose=True, random_state=42)
tree.fit(X, y)

# Vérifier le résumé
summary = tree.logger.get_log_summary()
print(f"Événements: {summary['total_events']}")
```

## ⚠️ Résolution de Problèmes

### Si `python test_task3.py` échoue

1. **Vérifiez les imports :**
   ```python
   python -c "from oc1 import ObliqueDecisionTree; print('OK')"
   ```

2. **Vérifiez que numpy est installé :**
   ```bash
   pip install numpy
   ```

3. **Si vous avez modifié le code, réinstallez :**
   ```bash
   pip install -e .
   ```

### Si les tests unitaires échouent

1. **Installez pytest :**
   ```bash
   pip install pytest
   ```

2. **Vérifiez que vous êtes dans le bon répertoire :**
   ```bash
   cd "c:\Users\pc\Desktop\ETUDES 7\Knowledge Discovery in Databases\Nouveau dossier\Oblique-Classifier-1"
   ```

### Si la visualisation ne fonctionne pas

C'est normal si matplotlib n'est pas installé. La visualisation est optionnelle :
```bash
pip install matplotlib networkx
```

## 📊 Résultats Attendus

Quand vous exécutez `python test_task3.py`, vous devriez voir :

```
============================================================
Test de la Tâche 3 - OC1 Oblique Decision Tree
============================================================

1. Test des imports...
   ✓ Imports réussis

2. Test de construction d'arbre...
   ✓ Arbre construit: X nœuds, Y feuilles

3. Test des prédictions...
   ✓ Prédictions réussies, précision: 0.XXX

... (tous les autres tests avec ✓)

============================================================
✓ Tous les tests de base sont passés avec succès!
============================================================
```

## 🚀 Test Rapide en Une Ligne

Pour un test ultra-rapide :

```python
python -c "from oc1 import ObliqueDecisionTree; from oc1.data import make_diagonal_dataset; X,y=make_diagonal_dataset(50,42); t=ObliqueDecisionTree(3,42); t.fit(X,y); print('✓ OK' if t._is_fitted else '✗ Erreur')"
```

Si ça affiche `✓ OK`, tout fonctionne !

## 📝 Notes

- Les tests peuvent prendre quelques secondes
- Certains tests peuvent donner des résultats légèrement différents selon la graine aléatoire
- La précision peut varier selon le dataset utilisé (c'est normal)
- La visualisation nécessite matplotlib (optionnel)

