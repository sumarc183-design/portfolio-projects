# Méthode du gradient conjugué

Projet académique (LaTeX/Overleaf) sur la **méthode du gradient conjugué** :
théorie, convergence finie, interprétation en optimisation quadratique, un
exemple numérique détaillé et une annexe contenant du code Python.

**📄 [Lire le rapport compilé (PDF)](Conjugate_Gradient_Method.pdf)**

## Contenu

- `main.tex` : source LaTeX principale (prête pour Overleaf).
- [`Conjugate_Gradient_Method.pdf`](Conjugate_Gradient_Method.pdf) : version PDF compilée.
- `Logo_of_the_Pantheon-Sorbonne_University_in_Paris.png` : logo de la page de titre.
- [`conjugate_gradient.py`](conjugate_gradient.py) : code Python de l'annexe, exécutable directement.

## Thèmes abordés

- résolution de systèmes linéaires symétriques définis positifs ;
- construction de directions conjuguées ;
- convergence en au plus `n` itérations (dimension du système) ;
- interprétation comme minimisation d'une forme quadratique ;
- comparaison avec la descente de gradient ;
- exemple numérique et code Python en annexe.

## Compilation

Ouvrir `main.tex` dans Overleaf, ou compiler localement :

```bash
pdflatex main.tex
```

## Exécution du code Python

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer le script :

```bash
python conjugate_gradient.py
```
