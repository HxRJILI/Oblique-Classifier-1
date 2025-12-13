"""
Script de test rapide pour vérifier que la Tâche 3 fonctionne correctement.

Usage:
    python test_task3.py
"""

import sys
import numpy as np

print("=" * 60)
print("Test de la Tâche 3 - OC1 Oblique Decision Tree")
print("=" * 60)

# Test 1: Import des modules
print("\n1. Test des imports...")
try:
    from oc1 import ObliqueDecisionTree
    from oc1.evaluation import (
        train_test_split,
        cross_validate,
        confusion_matrix,
        classification_report,
    )
    from oc1.data import make_diagonal_dataset
    from oc1.core.logging import TreeConstructionLogger
    print("   ✓ Imports réussis")
except ImportError as e:
    print(f"   ✗ Erreur d'import: {e}")
    sys.exit(1)

# Test 2: Construction d'arbre avec journalisation
print("\n2. Test de construction d'arbre avec journalisation...")
try:
    X, y = make_diagonal_dataset(n_samples=100, random_state=42)
    tree = ObliqueDecisionTree(max_depth=3, verbose=False, random_state=42)
    tree.fit(X, y)
    
    assert tree._is_fitted, "L'arbre devrait être entraîné"
    assert tree.get_n_nodes() > 0, "L'arbre devrait avoir des nœuds"
    print(f"   ✓ Arbre construit: {tree.get_n_nodes()} nœuds, {tree.get_n_leaves()} feuilles")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 3: Prédiction
print("\n3. Test des prédictions...")
try:
    predictions = tree.predict(X[:10])
    assert len(predictions) == 10, "Devrait prédire 10 échantillons"
    assert all(p in tree.classes_ for p in predictions), "Prédictions invalides"
    accuracy = tree.score(X, y)
    print(f"   ✓ Prédictions réussies, précision: {accuracy:.3f}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 4: Élagage par impureté
print("\n4. Test de l'élagage par impureté...")
try:
    tree2 = ObliqueDecisionTree(max_depth=5, random_state=42)
    tree2.fit(X, y)
    n_nodes_before = tree2.get_n_nodes()
    
    tree2.prune(method="impurity", impurity_threshold=2.0)
    n_nodes_after = tree2.get_n_nodes()
    
    assert n_nodes_after <= n_nodes_before, "L'élagage devrait réduire le nombre de nœuds"
    print(f"   ✓ Élagage réussi: {n_nodes_before} → {n_nodes_after} nœuds")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 5: Élagage REP
print("\n5. Test de l'élagage REP (Reduced Error Pruning)...")
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    tree3 = ObliqueDecisionTree(max_depth=5, random_state=42)
    tree3.fit(X_train, y_train)
    n_nodes_before = tree3.get_n_nodes()
    
    tree3.prune(method="rep", X_val=X_val, y_val=y_val)
    n_nodes_after = tree3.get_n_nodes()
    
    assert n_nodes_after <= n_nodes_before, "L'élagage REP devrait réduire le nombre de nœuds"
    print(f"   ✓ Élagage REP réussi: {n_nodes_before} → {n_nodes_after} nœuds")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 6: Validation croisée
print("\n6. Test de la validation croisée...")
try:
    tree4 = ObliqueDecisionTree(max_depth=3, random_state=42)
    results = cross_validate(tree4, X, y, cv=3, random_state=42)
    
    assert 'test_score' in results, "Résultats devraient contenir 'test_score'"
    assert len(results['test_score']) == 3, "Devrait avoir 3 scores"
    assert all(0 <= s <= 1 for s in results['test_score']), "Scores devraient être entre 0 et 1"
    print(f"   ✓ Validation croisée réussie: précision moyenne = {results['test_score'].mean():.3f}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 7: Matrice de confusion
print("\n7. Test de la matrice de confusion...")
try:
    y_pred = tree.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    assert cm.shape[0] == cm.shape[1], "Matrice devrait être carrée"
    assert cm.sum() == len(y), "Somme devrait être égale au nombre d'échantillons"
    print(f"   ✓ Matrice de confusion générée: {cm.shape}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 8: Rapport de classification
print("\n8. Test du rapport de classification...")
try:
    report = classification_report(y, y_pred)
    assert isinstance(report, str), "Le rapport devrait être une chaîne"
    assert "Precision" in report, "Le rapport devrait contenir 'Precision'"
    assert "Recall" in report, "Le rapport devrait contenir 'Recall'"
    print("   ✓ Rapport de classification généré")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 9: Journalisation
print("\n9. Test de la journalisation...")
try:
    tree5 = ObliqueDecisionTree(max_depth=3, verbose=False, random_state=42)
    tree5.fit(X, y)
    
    if tree5.logger:
        summary = tree5.logger.get_log_summary()
        assert 'total_events' in summary, "Le résumé devrait contenir 'total_events'"
        assert summary['total_events'] > 0, "Devrait y avoir des événements"
        print(f"   ✓ Journalisation active: {summary['total_events']} événements enregistrés")
    else:
        print("   ⚠ Journalisation non activée (verbose=False)")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 10: Journalisation verbose
print("\n10. Test de la journalisation verbose...")
try:
    tree6 = ObliqueDecisionTree(max_depth=2, verbose=True, random_state=42)
    print("   (Les logs devraient apparaître ci-dessus)")
    tree6.fit(X, y)
    print("   ✓ Journalisation verbose fonctionne")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 11: Visualisation (optionnel)
print("\n11. Test de la visualisation (optionnel)...")
try:
    from oc1.visualization import plot_decision_boundary_2d, plot_hyperplanes_2d
    print("   ✓ Module de visualisation disponible")
    print("   (Pour tester la visualisation, exécutez: python examples/task3_demo.py)")
except ImportError:
    print("   ⚠ Visualisation non disponible (matplotlib non installé)")
    print("   Installez avec: pip install matplotlib networkx")

print("\n" + "=" * 60)
print("✓ Tous les tests de base sont passés avec succès!")
print("=" * 60)
print("\nPour des tests plus complets, exécutez:")
print("  pytest oc1/tests/task3_tests/ -v")
print("\nPour voir une démonstration complète:")
print("  python examples/task3_demo.py")

