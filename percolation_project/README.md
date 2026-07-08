# Percolation par arêtes

Projet académique en probabilités, simulation et théorie des graphes.

Le rapport étudie la percolation par arêtes sur une grille finie. Chaque arête est ouverte indépendamment avec une probabilité `p`, puis une simulation détecte l'existence d'un chemin ouvert reliant le bord gauche au bord droit.

## Contenu

- `main.tex` : source LaTeX principal, prêt pour Overleaf.
- `Percolation_FR_final.pdf` : version PDF compilée.
- `figures/` : figures utilisées dans le rapport.

## Thèmes abordés

- percolation de Bernoulli par arêtes ;
- probabilité de croisement gauche-droite ;
- simulation Monte Carlo ;
- parcours en largeur pour détecter les chemins ouverts ;
- transition autour de `p = 1/2` ;
- dualité planaire et seuil critique du réseau carré ;
- limites et extensions possibles, dont Union-Find et finite-size scaling.

## Utilisation Overleaf

Importer ce dossier ou un zip contenant ces fichiers dans Overleaf, puis compiler `main.tex` avec `pdfLaTeX`.
