# Optimisation de portefeuille moyenne-variance

Projet académique de finance quantitative : implémentation et comparaison de
méthodes d'optimisation numérique pour un problème d'allocation de portefeuille
moyenne-variance.

L'objectif est de minimiser

```text
J(x) = x.T @ Sigma @ x - phi * mu.T @ x
```

sous les contraintes de portefeuille :

- les poids somment à un ;
- pas de vente à découvert dans la version contrainte ;
- un arbitrage rendement-risque contrôlé par `phi`.

## Méthodes implémentées

- réduction de la contrainte d'égalité de la dimension `n` à `n - 1` ;
- descente de gradient à pas exact (recherche linéaire compatible Wolfe) ;
- méthode de Newton avec recherche linéaire de Wolfe forte ;
- gradient conjugué linéaire sur le système d'optimalité ;
- solveurs de référence SciPy (BFGS, Newton-CG) ;
- optimisation contrainte sur le simplexe avec SLSQP ;
- tracés de convergence et des poids du portefeuille.

## Enseignements principaux

Les méthodes non contraintes convergent vers le même minimiseur à la précision
numérique près, mais à des vitesses très différentes :

- la descente de gradient demande beaucoup d'itérations (information de premier
  ordre uniquement) ;
- la méthode de Newton est très rapide sur cet objectif quadratique ;
- le gradient conjugué est bien adapté au système linéaire associé ;
- la solution contrainte sur le simplexe modifie l'allocation en imposant des
  poids positifs.

## Exécution

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer le script :

```bash
python mean_variance_portfolio.py
```

Les figures générées sont écrites dans `figures/`.
