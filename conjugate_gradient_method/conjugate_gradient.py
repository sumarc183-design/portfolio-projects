"""Conjugate Gradient method for symmetric positive definite systems.

Reproduces the numerical example from Chapter 4 of the report: solves
Ax = b from scratch and compares the result with NumPy's direct solver.
"""

import numpy as np


def is_symmetric_positive_definite(A, tol=1e-12):
    """
    Check whether a matrix is symmetric positive definite.

    A matrix A is symmetric positive definite if:
    1. A is symmetric: A.T = A
    2. all eigenvalues of A are strictly positive
    """
    if not np.allclose(A, A.T, atol=tol):
        return False

    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > tol)


def quadratic_function(A, b, x):
    """
    Compute the quadratic objective associated with Ax = b:

        f(x) = 1/2 x.T A x - b.T x
    """
    return 0.5 * x.T @ A @ x - b.T @ x


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None,
                       verbose=True):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.
    """
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).copy()

    if max_iter is None:
        max_iter = n

    # Residual r_k = b - A x_k.
    # Since g_k = A x_k - b = -r_k, the first descent direction is r_0.
    r = b - A @ x
    d = r.copy()

    history = [np.linalg.norm(r)]

    if verbose:
        print("Initial point:")
        print("x0 =", x)
        print("Initial residual norm =", history[-1])
        print("-" * 60)

    for k in range(max_iter):
        Ad = A @ d
        denominator = d.T @ Ad

        if abs(denominator) < 1e-15:
            raise ValueError("The denominator is too close to zero.")

        alpha = (r.T @ r) / denominator
        x_new = x + alpha * d
        r_new = r - alpha * Ad

        residual_norm = np.linalg.norm(r_new)
        history.append(residual_norm)

        if verbose:
            print(f"Iteration {k + 1}")
            print("alpha =", alpha)
            print("x =", x_new)
            print("residual =", r_new)
            print("residual norm =", residual_norm)
            print("objective value =", quadratic_function(A, b, x_new))
            print("-" * 60)

        if residual_norm < tol:
            return x_new, history

        beta = (r_new.T @ r_new) / (r.T @ r)
        d = r_new + beta * d

        x = x_new
        r = r_new

    return x, history


def print_convergence_table(residual_history):
    """Print a formatted table of iteration index vs residual norm."""
    print("Convergence table:")
    print(f"{'Iteration':>10} | {'Residual norm':>15}")
    print("-" * 30)
    for k, residual_norm in enumerate(residual_history):
        print(f"{k:>10} | {residual_norm:>15.6e}")


def main() -> None:
    # Numerical example from the report
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ])

    b = np.array([1.0, 2.0, 3.0])
    x0 = np.array([0.0, 0.0, 0.0])

    print("Is A symmetric positive definite?", is_symmetric_positive_definite(A))
    print("=" * 60)

    x_cg, residual_history = conjugate_gradient(
        A=A,
        b=b,
        x0=x0,
        tol=1e-12,
        max_iter=10,
        verbose=True
    )

    x_direct = np.linalg.solve(A, b)

    print("Final solution obtained with Conjugate Gradient:")
    print(x_cg)

    print("Direct solution using numpy.linalg.solve:")
    print(x_direct)

    print("Difference between both solutions:")
    print(np.linalg.norm(x_cg - x_direct))

    print("Residual history:")
    print(residual_history)

    print_convergence_table(residual_history)


if __name__ == "__main__":
    main()
