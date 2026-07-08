# Portfolio de projets Data Science, mathématiques et MLOps

Ce dépôt regroupe plusieurs projets académiques et professionnels présentés sous une forme publiable. Les travaux sensibles sont anonymisés : aucune donnée source, aucun modèle entraîné, aucun identifiant interne et aucun secret ne sont publiés.

## Projets inclus

- [Résumé public Data Science / MLOps](#objectif-du-projet) : synthèse anonymisée d'un projet de modélisation, industrialisation et monitoring.
- [Mean-Variance Portfolio Optimization](mean_variance_portfolio_optimization/) : projet académique personnel en finance quantitative, avec implémentation Python de méthodes numériques d'optimisation.
- [Conjugate Gradient Method](conjugate_gradient_method/) : projet académique LaTeX/Overleaf sur la méthode du gradient conjugué, sa convergence finie, son interprétation optimisation et un exemple numérique Python.
- [Percolation par arêtes](percolation_project/) : projet académique LaTeX/Overleaf sur la percolation, la simulation Monte Carlo, les chemins ouverts et la dualité planaire.

## Objectif du projet

L'objectif était de construire une chaîne reproductible permettant de préparer des données métier brutes, d'entraîner des modèles supervisés, d'évaluer leur stabilité dans le temps, puis de faciliter leur déploiement et leur suivi opérationnel.

Le travail a couvert plusieurs volets :

- préparation et fiabilisation des données ;
- création de variables métier interprétables ;
- séparation temporelle des jeux d'entraînement et de test ;
- comparaison de plusieurs familles de modèles supervisés ;
- analyse du surapprentissage et de la dérive de données ;
- industrialisation d'un pipeline complet de scoring ;
- suivi des expériences et des artefacts de modèle ;
- contrôle de cohérence entre scoring local et scoring via API ;
- mise en place d'indicateurs de monitoring pour vérifier la disponibilité des variables utilisées en production.

## Travaux réalisés

### Préparation des données

La préparation a été structurée pour limiter les risques de fuite d'information entre l'entraînement et l'évaluation. Les transformations ont été regroupées dans des composants réutilisables afin que le même traitement soit appliqué pendant l'entraînement, le test et le scoring.

Les travaux incluent notamment :

- nettoyage et typage de variables ;
- gestion explicite des valeurs manquantes ;
- création d'indicateurs de qualité de données ;
- transformation de variables numériques et catégorielles ;
- regroupement de modalités rares ;
- construction de classes à partir de seuils appris uniquement sur le jeu d'entraînement.

### Modélisation

Plusieurs approches supervisées ont été comparées avec une logique d'évaluation temporelle. L'objectif n'était pas seulement de maximiser une métrique, mais de trouver un compromis entre performance, stabilité, explicabilité et facilité d'exploitation.

Les analyses ont porté sur :

- modèles linéaires ;
- modèles d'ensemble ;
- modèles de gradient boosting ;
- métriques de discrimination et de classement ;
- matrices de confusion ;
- lift et priorisation opérationnelle ;
- écarts entre entraînement, validation et test.

### Sélection et stabilité des variables

Une attention particulière a été portée aux variables conservées dans le modèle. Les analyses ont cherché à identifier les signaux utiles, à limiter la redondance et à repérer les variables instables dans le temps.

Les contrôles incluent :

- classement de variables ;
- analyse de contribution ;
- détection de variables trop corrélées ;
- analyse de dérive entre périodes ;
- suivi de la stabilité des distributions.

### Industrialisation

La préparation des données et le modèle ont été intégrés dans un pipeline unique afin de réduire les écarts entre expérimentation et production.

Les travaux d'industrialisation comprennent :

- encapsulation de la préparation et du classifieur dans un pipeline unique ;
- gestion robuste des colonnes absentes ou supplémentaires ;
- alignement des noms de variables ;
- sérialisation du pipeline ;
- journalisation des paramètres, métriques et artefacts ;
- documentation du fonctionnement et des limites du modèle.

### Validation API

Une validation a été réalisée pour vérifier que le score obtenu via l'API correspondait au score du modèle rechargé localement sur les mêmes entrées.

Cette étape permettait de contrôler :

- le format des payloads ;
- la cohérence des transformations embarquées ;
- la stabilité du wrapper de scoring ;
- la correspondance entre modèle déployé et modèle évalué ;
- l'absence d'écart numérique significatif sur les scénarios testés.

### Monitoring

Un dispositif de surveillance a été conçu pour suivre la disponibilité des variables importantes utilisées par le modèle une fois exposé via API.

Le monitoring vérifie notamment :

- la présence des variables attendues ;
- la présence des clés nécessaires aux enrichissements ;
- les cas où une variable ne peut pas être enrichie ;
- les taux hebdomadaires d'anomalie ;
- les écarts par rapport à l'historique.

## Résultats anonymisés

Les artefacts MLflow analysés confirment que deux pipelines XGBoost ont été exportés au format `python_function`, avec une signature d'entrée explicite et deux probabilités de sortie. Les fichiers sources de ces artefacts ne sont pas publiés.

Les ordres de grandeur validés sont les suivants :

- premier périmètre : environ 17 000 lignes d'entraînement et 4 000 lignes de test ;
- second périmètre : environ 49 000 lignes d'entraînement et 12 000 lignes de test ;
- deux modèles sérialisés sous MLflow avec environnement Python reproductible ;
- entrées rendues optionnelles dans la signature afin de supporter un scoring robuste avec imputation et alignement de colonnes ;
- sorties standardisées sous forme de probabilités de classe ;
- validation orientée production : rechargement du modèle, scoring sur données brutes préparées par le pipeline et contrôle de cohérence avec l'exposition API.

Ces résultats montrent que le projet ne s'est pas limité à une expérimentation notebook : il a produit des artefacts industrialisables, versionnés et contrôlables. Les métriques exactes, noms de variables, identifiants de runs et chemins internes restent exclus de cette version publique.

## Compétences mobilisées

- Python
- Pandas
- Scikit-learn
- XGBoost
- MLflow
- Dataiku DSS
- APIs de scoring
- Feature engineering
- Validation temporelle
- Analyse de dérive
- Monitoring de modèles
- Documentation technique
- Industrialisation de pipelines ML

## Ce qui n'est pas publié

Pour des raisons de confidentialité, ce dépôt ne contient pas :

- données sources ou extraits de données ;
- modèles entraînés ;
- artefacts MLflow ;
- métriques détaillées confidentielles ;
- noms d'organisations, projets, endpoints ou datasets internes ;
- captures d'écran d'outils internes ;
- logs API ;
- fichiers de configuration ;
- secrets, tokens ou identifiants ;
- code métier réutilisable en production.

## Résultat

Le projet a abouti à une chaîne complète allant de la préparation des données au suivi post-déploiement. La valeur principale réside dans la reproductibilité du pipeline, la maîtrise des risques de fuite d'information, la traçabilité des expériences et la capacité à surveiller les variables réellement utilisées par le modèle en production.
