# Mean-Variance Portfolio Optimization

Academic quantitative finance project implementing and comparing numerical optimization methods for a mean-variance portfolio allocation problem.

The objective is to minimize

```text
J(x) = x.T @ Sigma @ x - phi * mu.T @ x
```

under the portfolio constraints:

- the weights sum to one;
- no short selling in the constrained version;
- a risk-return trade-off controlled by `phi`.

## Methods Implemented

- equality-constraint reduction from dimension `n` to `n - 1`;
- steepest descent with exact Wolfe-compliant line search;
- Newton method with strong Wolfe line search;
- linear conjugate gradient on the optimality system;
- SciPy BFGS and Newton-CG reference solvers;
- constrained optimization on the simplex with SLSQP;
- convergence and portfolio-weight plots.

## Main Takeaways

The unconstrained methods converge to the same minimizer up to numerical precision, but with very different speeds:

- steepest descent requires many iterations because it only uses first-order information;
- Newton's method is extremely fast on this quadratic objective;
- conjugate gradient is well suited to the associated linear system;
- the constrained simplex solution changes the allocation by enforcing nonnegative weights.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python mean_variance_portfolio.py
```

Generated figures are written to `figures/`.
