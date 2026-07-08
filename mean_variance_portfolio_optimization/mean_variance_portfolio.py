"""Mean-variance portfolio optimization with numerical methods.

This script compares unconstrained and constrained optimization methods for
the quadratic mean-variance objective used in a portfolio allocation problem.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize


np.set_printoptions(precision=10, suppress=True)

PHI = 5.0
N_ASSETS = 12

MU = np.array(
    [3.66, 1.33, 2.51, 2.24, 2.62, 2.03, 1.91, 1.39, 2.61, 3.58, 5.23, 8.36],
    dtype=float,
)

SIGMA = np.array(
    [
        [73.73, 40.79, 39.46, 38.01, 49.40, 58.51, 46.08, 42.26, -1.22, 12.44, -2.31, 131.17],
        [40.79, 39.54, 32.15, 34.43, 34.25, 36.88, 33.11, 26.50, -3.13, 19.06, 0.53, 136.78],
        [39.46, 32.15, 54.98, 31.62, 33.80, 41.20, 26.48, 29.76, -2.90, 25.34, -3.62, 151.82],
        [38.01, 34.43, 31.62, 39.61, 35.73, 33.64, 38.12, 26.00, -0.08, 13.23, -5.61, 73.96],
        [49.40, 34.25, 33.80, 35.73, 68.10, 36.88, 37.02, 31.64, 14.42, 3.33, 1.43, -0.12],
        [58.51, 36.88, 41.20, 33.64, 36.88, 66.93, 40.89, 39.08, -13.71, 22.07, -14.96, 210.26],
        [46.08, 33.11, 26.48, 38.12, 37.02, 40.89, 75.40, 38.58, 19.01, 19.37, 30.95, 52.63],
        [42.26, 26.50, 29.76, 26.00, 31.64, 39.08, 38.58, 47.35, -3.53, 17.02, -8.28, 111.37],
        [-1.22, -3.13, -2.90, -0.08, 14.42, -13.71, 19.01, -3.53, 242.03, -90.65, 196.18, -507.15],
        [12.44, 19.06, 25.34, 13.23, 3.33, 22.07, 19.37, 17.02, -90.65, 110.87, -50.97, 360.36],
        [-2.31, 0.53, -3.62, -5.61, 1.43, -14.96, 30.95, -8.28, 196.18, -50.97, 428.23, -504.38],
        [131.17, 136.78, 151.82, 73.96, -0.12, 210.26, 52.63, 111.37, -507.15, 360.36, -504.38, 2863.07],
    ],
    dtype=float,
)


def build_reduced_problem() -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the reduced quadratic constants after eliminating the last weight."""
    cov_nn = SIGMA[N_ASSETS - 1, N_ASSETS - 1]
    cov_in = SIGMA[: N_ASSETS - 1, N_ASSETS - 1]

    a = -PHI * (MU[: N_ASSETS - 1] - MU[N_ASSETS - 1]) - 2.0 * cov_nn + 2.0 * cov_in
    b_matrix = SIGMA[: N_ASSETS - 1, : N_ASSETS - 1] - cov_in[:, None] - cov_in[None, :] + cov_nn
    constant = -PHI * MU[N_ASSETS - 1] + cov_nn
    hessian = 2.0 * b_matrix
    rhs = -a
    return constant, a, b_matrix, hessian, rhs


CONST, A_REDUCED, B_REDUCED, H_REDUCED, RHS = build_reduced_problem()


def reduced_objective(x: np.ndarray) -> float:
    """Reduced objective in R^(n-1)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(CONST + x @ A_REDUCED + x @ (B_REDUCED @ x))


def reduced_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of the reduced objective."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return A_REDUCED + 2.0 * (B_REDUCED @ x)


def full_objective(x: np.ndarray) -> float:
    """Full mean-variance objective in R^n."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ SIGMA @ x - PHI * (MU @ x))


def full_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of the full objective."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return 2.0 * SIGMA @ x - PHI * MU


def complete_portfolio(x_reduced: np.ndarray) -> np.ndarray:
    """Append the last portfolio weight implied by the equality constraint."""
    x_reduced = np.asarray(x_reduced, dtype=float).reshape(-1)
    return np.append(x_reduced, 1.0 - np.sum(x_reduced))


def steepest_descent_exact(
    x0: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 5_000_000,
    store_history: bool = False,
):
    """Steepest descent with exact line search for a quadratic objective."""
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    grad_norm_history: list[float] = []
    objective_history: list[float] = []

    for iteration in range(max_iter):
        gradient = reduced_gradient(x)
        grad_norm = float(la.norm(gradient))

        if store_history:
            grad_norm_history.append(grad_norm)
            objective_history.append(reduced_objective(x))

        if grad_norm <= tol:
            result = (x, reduced_objective(x), iteration, grad_norm)
            return result + (grad_norm_history, objective_history) if store_history else result

        denominator = float(gradient @ (H_REDUCED @ gradient))
        alpha = 1e-3 if denominator <= 0 else float((gradient @ gradient) / denominator)
        x = x - alpha * gradient

    final_grad_norm = float(la.norm(reduced_gradient(x)))
    result = (x, reduced_objective(x), max_iter, final_grad_norm)
    return result + (grad_norm_history, objective_history) if store_history else result


def strong_wolfe_line_search(
    xk: np.ndarray,
    pk: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    alpha_max: float = 10.0,
    max_iter: int = 50,
) -> float:
    """Strong Wolfe line search for phi(alpha)=f(xk + alpha * pk)."""
    xk = np.asarray(xk, dtype=float).reshape(-1)
    pk = np.asarray(pk, dtype=float).reshape(-1)

    def phi_alpha(alpha: float) -> float:
        return reduced_objective(xk + alpha * pk)

    def dphi_alpha(alpha: float) -> float:
        return float(reduced_gradient(xk + alpha * pk) @ pk)

    phi0 = phi_alpha(0.0)
    dphi0 = dphi_alpha(0.0)
    if dphi0 >= 0:
        return 0.0

    alpha_prev = 0.0
    phi_prev = phi0
    alpha = alpha0

    def zoom(alo: float, ahi: float) -> float:
        philo = phi_alpha(alo)
        for _ in range(max_iter):
            aj = 0.5 * (alo + ahi)
            phij = phi_alpha(aj)
            if (phij > phi0 + c1 * aj * dphi0) or (phij >= philo):
                ahi = aj
            else:
                dphij = dphi_alpha(aj)
                if abs(dphij) <= -c2 * dphi0:
                    return aj
                if dphij * (ahi - alo) >= 0:
                    ahi = alo
                alo = aj
                philo = phij
            if abs(ahi - alo) < 1e-14:
                return aj
        return aj

    for _ in range(max_iter):
        phia = phi_alpha(alpha)
        if (phia > phi0 + c1 * alpha * dphi0) or (phia >= phi_prev and alpha > 0):
            return zoom(alpha_prev, alpha)

        dphia = dphi_alpha(alpha)
        if abs(dphia) <= -c2 * dphi0:
            return alpha
        if dphia >= 0:
            return zoom(alpha, alpha_prev)

        alpha_prev = alpha
        phi_prev = phia
        alpha = min(2.0 * alpha, alpha_max)

    return alpha


def newton_wolfe(
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    store_history: bool = False,
):
    """Newton method with Wolfe line search for the reduced quadratic problem."""
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    grad_norm_history: list[float] = []

    for iteration in range(max_iter):
        gradient = reduced_gradient(x)
        grad_norm = float(la.norm(gradient))

        if store_history:
            grad_norm_history.append(grad_norm)

        if grad_norm <= tol:
            result = (x, reduced_objective(x), iteration)
            return result + (grad_norm_history,) if store_history else result

        direction = la.solve(H_REDUCED, -gradient)
        alpha = strong_wolfe_line_search(x, direction)
        x = x + (1.0 if alpha == 0.0 else alpha) * direction

    if store_history:
        grad_norm_history.append(float(la.norm(reduced_gradient(x))))
        return x, reduced_objective(x), max_iter, grad_norm_history
    return x, reduced_objective(x), max_iter


def conjugate_gradient(
    matrix: np.ndarray,
    rhs: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    store_history: bool = False,
):
    """Linear conjugate gradient method for Ax=b."""
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    residual = matrix @ x - rhs
    direction = -residual
    residual_dot = float(residual @ residual)
    initial_norm = float(np.sqrt(residual_dot))
    residual_history = [float(la.norm(residual))] if store_history else None

    if initial_norm == 0.0:
        result = (x, 0, 0.0)
        return result + (residual_history,) if store_history else result

    for iteration in range(max_iter):
        matrix_direction = matrix @ direction
        denominator = float(direction @ matrix_direction)
        if denominator <= 0:
            break

        alpha = residual_dot / denominator
        x = x + alpha * direction
        residual = residual + alpha * matrix_direction
        residual_dot_new = float(residual @ residual)
        residual_norm = float(np.sqrt(residual_dot_new))

        if store_history:
            residual_history.append(residual_norm)

        if residual_norm <= tol * initial_norm:
            result = (x, iteration + 1, residual_norm)
            return result + (residual_history,) if store_history else result

        beta = residual_dot_new / residual_dot
        direction = -residual + beta * direction
        residual_dot = residual_dot_new

    final_norm = float(la.norm(matrix @ x - rhs))
    result = (x, iteration + 1, final_norm)
    return result + (residual_history,) if store_history else result


def solve_unconstrained_with_scipy(x0: np.ndarray):
    """Solve the reduced unconstrained problem using SciPy BFGS and Newton-CG."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    x_reference = la.solve(H_REDUCED, -A_REDUCED)
    f_reference = reduced_objective(x_reference)

    res_bfgs = minimize(
        fun=lambda z: reduced_objective(z),
        x0=x0,
        jac=lambda z: reduced_gradient(z),
        method="BFGS",
        options={"gtol": 1e-10, "disp": False},
    )

    res_newton_cg = minimize(
        fun=lambda z: reduced_objective(z),
        x0=x0,
        jac=lambda z: reduced_gradient(z),
        hessp=lambda z, v: H_REDUCED @ v,
        method="Newton-CG",
        options={"xtol": 1e-12, "disp": False, "maxiter": 50},
    )

    return x_reference, f_reference, res_bfgs, res_newton_cg


def solve_constrained_slsqp(x0: np.ndarray | None = None, tol: float = 1e-10, max_iter: int = 10_000):
    """Solve the no-short-selling constrained portfolio problem using SLSQP."""
    if x0 is None:
        x0 = np.ones(N_ASSETS) / N_ASSETS
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    constraints = (
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1.0,
            "jac": lambda x: np.ones_like(x),
        },
    )
    bounds = [(0.0, None) for _ in range(N_ASSETS)]

    return minimize(
        full_objective,
        x0,
        jac=full_gradient,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"ftol": tol, "maxiter": max_iter, "disp": False},
    )


def save_steepest_descent_figure(grad_norm_history: list[float]) -> None:
    """Save the steepest descent convergence figure."""
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(grad_norm_history)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient norm")
    plt.title("Steepest descent convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/steepest_descent_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_newton_figure(grad_norm_history: list[float]) -> None:
    """Save the Newton convergence figure."""
    x_axis = list(range(len(grad_norm_history)))
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(x_axis, grad_norm_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient norm")
    plt.title("Newton method convergence")
    plt.xticks(x_axis)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/newton_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_conjugate_gradient_figure(residual_history: list[float]) -> None:
    """Save the conjugate gradient convergence figure."""
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(residual_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm")
    plt.title("Linear conjugate gradient convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/conjugate_gradient_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_constrained_portfolio_figure(x_constrained: np.ndarray) -> None:
    """Save the constrained optimal portfolio weights figure."""
    x_plot = np.asarray(x_constrained, dtype=float).reshape(-1)
    plt.figure(figsize=(8, 4.5))
    plt.bar(np.arange(1, len(x_plot) + 1), x_plot)
    plt.xlabel("Asset")
    plt.ylabel("Weight")
    plt.title("Constrained optimal portfolio weights")
    plt.xticks(np.arange(1, len(x_plot) + 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/constrained_portfolio.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    x0_reduced = np.ones(N_ASSETS - 1) * (1.0 / N_ASSETS)

    x_sd, f_sd, it_sd, grad_sd, grad_hist_sd, _ = steepest_descent_exact(
        x0_reduced,
        tol=1e-5,
        store_history=True,
    )
    save_steepest_descent_figure(grad_hist_sd)
    print("=== Steepest descent ===")
    print(f"Iterations: {it_sd}")
    print(f"Final gradient norm: {grad_sd:.6e}")
    print(f"Reduced objective: {f_sd:.12f}")
    print(f"Full portfolio: {complete_portfolio(x_sd)}")
    print()

    x_newton, f_newton, it_newton, grad_hist_newton = newton_wolfe(
        x0_reduced,
        tol=1e-10,
        max_iter=50,
        store_history=True,
    )
    save_newton_figure(grad_hist_newton)
    print("=== Newton + Wolfe ===")
    print(f"Iterations: {it_newton}")
    print(f"Reduced objective: {f_newton:.12f}")
    print(f"Full portfolio: {complete_portfolio(x_newton)}")
    print()

    x_cg, it_cg, res_norm_cg, residual_hist_cg = conjugate_gradient(
        H_REDUCED,
        RHS,
        x0_reduced,
        tol=1e-10,
        max_iter=5000,
        store_history=True,
    )
    save_conjugate_gradient_figure(residual_hist_cg)
    print("=== Linear conjugate gradient ===")
    print(f"Iterations: {it_cg}")
    print(f"Final residual norm: {res_norm_cg:.3e}")
    print(f"Reduced objective: {reduced_objective(x_cg):.12f}")
    print(f"Full portfolio: {complete_portfolio(x_cg)}")
    print()

    x_ref, f_ref, res_bfgs, res_ncg = solve_unconstrained_with_scipy(x0_reduced)
    print("=== SciPy reference solvers ===")
    print(f"Linear solve objective: {f_ref:.12f}")
    print(f"BFGS success: {res_bfgs.success}, iterations: {res_bfgs.nit}")
    print(f"Newton-CG success: {res_ncg.success}, iterations: {res_ncg.nit}")
    print(f"BFGS distance to reference: {la.norm(res_bfgs.x - x_ref):.3e}")
    print(f"Newton-CG distance to reference: {la.norm(res_ncg.x - x_ref):.3e}")
    print()

    constrained_result = solve_constrained_slsqp()
    x_constrained = constrained_result.x
    save_constrained_portfolio_figure(x_constrained)
    print("=== Constrained SLSQP ===")
    print(f"Success: {constrained_result.success}")
    print(f"Iterations: {constrained_result.nit}")
    print(f"Sum of weights: {np.sum(x_constrained):.12f}")
    print(f"Minimum weight: {np.min(x_constrained):.12f}")
    print(f"Objective: {full_objective(x_constrained):.12f}")
    print(f"Portfolio: {x_constrained}")
    print()

    print("Figures saved in the folder: figures/")


if __name__ == "__main__":
    main()
