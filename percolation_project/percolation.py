"""Bond percolation simulation on a finite square grid.

Simulates edge percolation, detects left-right crossings via breadth-first
search, and estimates the crossing probability by Monte Carlo simulation.
"""

import numpy as np
from collections import deque


def generate_bond_percolation(n, p, rng=None):
    """
    Genere une configuration de percolation par aretes sur une grille n x n.

    Les sommets sont indexes par (i, j), avec i, j dans {0, ..., n}.

    horizontal[i, j] represente l'arete entre (i, j) et (i, j+1).
    vertical[i, j] represente l'arete entre (i, j) et (i+1, j).
    """
    if rng is None:
        rng = np.random.default_rng()

    horizontal = rng.random((n + 1, n)) < p
    vertical = rng.random((n, n + 1)) < p

    return horizontal, vertical


def has_left_right_crossing(horizontal, vertical):
    """
    Verifie s'il existe un chemin ouvert du bord gauche
    vers le bord droit en utilisant un parcours en largeur.
    """
    n = horizontal.shape[1]
    visited = np.zeros((n + 1, n + 1), dtype=bool)
    queue = deque()

    # Initialisation avec tous les sommets du bord gauche
    for i in range(n + 1):
        visited[i, 0] = True
        queue.append((i, 0))

    while queue:
        i, j = queue.popleft()

        # Si le bord droit est atteint, un croisement existe
        if j == n:
            return True

        # Deplacement vers la gauche
        if j > 0 and horizontal[i, j - 1] and not visited[i, j - 1]:
            visited[i, j - 1] = True
            queue.append((i, j - 1))

        # Deplacement vers la droite
        if j < n and horizontal[i, j] and not visited[i, j + 1]:
            visited[i, j + 1] = True
            queue.append((i, j + 1))

        # Deplacement vers le haut
        if i > 0 and vertical[i - 1, j] and not visited[i - 1, j]:
            visited[i - 1, j] = True
            queue.append((i - 1, j))

        # Deplacement vers le bas
        if i < n and vertical[i, j] and not visited[i + 1, j]:
            visited[i + 1, j] = True
            queue.append((i + 1, j))

    return False


def estimate_theta(n, p, simulations=1000, seed=42):
    """
    Estime theta_n(p), la probabilite d'un croisement gauche-droite,
    par simulation Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    crossings = 0

    for _ in range(simulations):
        horizontal, vertical = generate_bond_percolation(n, p, rng)
        if has_left_right_crossing(horizontal, vertical):
            crossings += 1

    return crossings / simulations


def confidence_interval(theta_hat, simulations, level=1.96):
    """
    Renvoie un intervalle de confiance approximatif a 95% pour theta_hat.
    """
    standard_error = np.sqrt(theta_hat * (1 - theta_hat) / simulations)
    lower = max(0.0, theta_hat - level * standard_error)
    upper = min(1.0, theta_hat + level * standard_error)
    return lower, upper


def main() -> None:
    sizes = [10, 30, 60]
    probabilities = [0.4, 0.5, 0.6]
    simulations = 1000

    for n in sizes:
        print(f"n = {n}")
        for p in probabilities:
            theta_hat = estimate_theta(n, p, simulations=simulations)
            ci_low, ci_high = confidence_interval(theta_hat, simulations)
            print(
                f"p = {p:.2f}, theta_hat = {theta_hat:.3f}, "
                f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]"
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
