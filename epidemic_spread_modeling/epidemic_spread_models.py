"""Epidemic spread models from the TER report.

Implemented models:
- SIR: Susceptible, Infected, Recovered.
- CUIR: Confined, Unconfined, Infected, Recovered.
- VSIRS: Vaccinated, Susceptible, Infected, Recovered, Susceptible loop.

The numerical method is the explicit Euler scheme:
    y[n + 1] = y[n] + dt * f(y[n])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path(__file__).resolve().parent / "figures"


@dataclass(frozen=True)
class SIRParameters:
    beta: float
    gamma: float
    susceptible_0: float
    infected_0: float
    recovered_0: float = 0.0


@dataclass(frozen=True)
class CUIRParameters:
    beta_confined: float
    beta_unconfined: float
    gamma: float
    confined_0: float
    unconfined_0: float
    infected_0: float
    recovered_0: float = 0.0


@dataclass(frozen=True)
class VSIRSParameters:
    beta: float
    gamma: float
    kappa: float
    alpha: float
    susceptible_0: float
    infected_0: float
    vaccinated_0: float = 0.0
    recovered_0: float = 0.0


def explicit_euler(rhs, initial_state: np.ndarray, final_time: float, dt: float):
    """Solve y' = rhs(y) with the explicit Euler method."""
    if dt <= 0:
        raise ValueError("dt must be strictly positive")

    n_steps = int(np.ceil(final_time / dt))
    time = np.linspace(0.0, n_steps * dt, n_steps + 1)
    states = np.zeros((n_steps + 1, len(initial_state)), dtype=float)
    states[0] = initial_state

    for i in range(n_steps):
        states[i + 1] = states[i] + dt * rhs(states[i])
        states[i + 1] = np.maximum(states[i + 1], 0.0)

    return time, states


def simulate_sir(params: SIRParameters, final_time: float, dt: float):
    """Simulate the SIR model from the report."""
    y0 = np.array([params.susceptible_0, params.infected_0, params.recovered_0])

    def rhs(y):
        susceptible, infected, recovered = y
        return np.array(
            [
                -params.beta * susceptible * infected,
                params.beta * susceptible * infected - params.gamma * infected,
                params.gamma * infected,
            ]
        )

    return explicit_euler(rhs, y0, final_time, dt)


def simulate_cuir(params: CUIRParameters, final_time: float, dt: float):
    """Simulate the CUIR model from the report."""
    y0 = np.array(
        [
            params.confined_0,
            params.unconfined_0,
            params.infected_0,
            params.recovered_0,
        ]
    )

    def rhs(y):
        confined, unconfined, infected, recovered = y
        return np.array(
            [
                -params.beta_confined * infected * confined,
                -params.beta_unconfined * infected * unconfined,
                params.beta_confined * infected * confined
                + params.beta_unconfined * infected * unconfined
                - params.gamma * infected,
                params.gamma * infected,
            ]
        )

    return explicit_euler(rhs, y0, final_time, dt)


def simulate_vsirs(params: VSIRSParameters, final_time: float, dt: float):
    """Simulate the VSIRS model from the report."""
    y0 = np.array(
        [
            params.vaccinated_0,
            params.susceptible_0,
            params.infected_0,
            params.recovered_0,
        ]
    )

    def rhs(y):
        vaccinated, susceptible, infected, recovered = y
        return np.array(
            [
                params.kappa * susceptible,
                -params.kappa * susceptible
                - params.beta * infected * susceptible
                + params.alpha * recovered,
                params.beta * infected * susceptible - params.gamma * infected,
                params.gamma * infected - params.alpha * recovered,
            ]
        )

    return explicit_euler(rhs, y0, final_time, dt)


def basic_reproduction_number(beta: float, susceptible_0: float, gamma: float) -> float:
    """Return R0 = beta * S0 / gamma."""
    return beta * susceptible_0 / gamma


def herd_immunity_threshold(population: float, reproduction_number: float) -> float:
    """Return (1 - 1/R0) * N when R0 > 1, else 0."""
    if reproduction_number <= 1:
        return 0.0
    return (1.0 - 1.0 / reproduction_number) * population


def millions(values: np.ndarray) -> np.ndarray:
    return values / 1_000_000.0


def plot_compartments(time, states, labels, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, label in enumerate(labels):
        ax.plot(time, millions(states[:, index]), linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Population, in millions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def plot_infected_comparison(results, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for label, time, states, infected_index in results:
        ax.plot(time, millions(states[:, infected_index]), linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Infected population, in millions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def run_sir_examples():
    base = {
        "susceptible_0": 80_000_000.0,
        "infected_0": 1.0,
        "recovered_0": 0.0,
    }

    beta_results = []
    for beta in [1e-8, 5e-8, 1e-7]:
        params = SIRParameters(beta=beta, gamma=0.9, **base)
        time, states = simulate_sir(params, final_time=2_500, dt=0.02)
        beta_results.append((f"beta={beta:g}", time, states, 1))
    plot_infected_comparison(
        beta_results,
        "SIR - influence of transmission rate beta",
        "sir_beta_comparison.png",
    )

    gamma_results = []
    for gamma in [2e-2, 1e-1, 9e-1]:
        params = SIRParameters(beta=1e-7, gamma=gamma, **base)
        time, states = simulate_sir(params, final_time=3_000, dt=0.02)
        gamma_results.append((f"gamma={gamma:g}", time, states, 1))
    plot_infected_comparison(
        gamma_results,
        "SIR - influence of recovery rate gamma",
        "sir_gamma_comparison.png",
    )

    r0_results = []
    for target_r0 in [5, 25, 50]:
        susceptible_0 = 60_000_000.0
        beta = 1e-7
        gamma = beta * susceptible_0 / target_r0
        params = SIRParameters(beta=beta, gamma=gamma, susceptible_0=susceptible_0, infected_0=1.0)
        time, states = simulate_sir(params, final_time=3_000, dt=0.02)
        r0_results.append((f"R0={target_r0}", time, states, 1))
    plot_infected_comparison(
        r0_results,
        "SIR - influence of basic reproduction number R0",
        "sir_r0_comparison.png",
    )


def run_cuir_examples():
    scenarios = [
        ("C0 < U0", 30_000_000.0, 70_000_000.0),
        ("C0 > U0", 70_000_000.0, 30_000_000.0),
        ("C0 = U0", 50_000_000.0, 50_000_000.0),
    ]
    results = []
    for label, c0, u0 in scenarios:
        params = CUIRParameters(
            beta_confined=1.6e-9,
            beta_unconfined=1.6e-8,
            gamma=0.04,
            confined_0=c0,
            unconfined_0=u0,
            infected_0=1.0,
        )
        time, states = simulate_cuir(params, final_time=1_600, dt=0.02)
        results.append((label, time, states, 2))
    plot_infected_comparison(
        results,
        "CUIR - effect of confined and unconfined populations",
        "cuir_confinement_comparison.png",
    )

    params = CUIRParameters(
        beta_confined=1.6e-10,
        beta_unconfined=1.6e-8,
        gamma=0.028,
        confined_0=70_000_000.0,
        unconfined_0=10_000_000.0,
        infected_0=2.0,
    )
    time, states = simulate_cuir(params, final_time=2_000, dt=0.02)
    plot_compartments(
        time,
        states,
        ["Confined", "Unconfined", "Infected", "Recovered"],
        "CUIR - low transmission among confined individuals",
        "cuir_low_beta_confined.png",
    )


def run_vsirs_examples():
    beta_results = []
    for beta in [1e-9, 1e-8, 1e-7]:
        params = VSIRSParameters(
            beta=beta,
            gamma=2e-1,
            kappa=1e-1,
            alpha=4e-2,
            susceptible_0=80_000_000.0,
            infected_0=1_000_000.0,
        )
        time, states = simulate_vsirs(params, final_time=800, dt=0.01)
        beta_results.append((f"beta={beta:g}", time, states, 2))
    plot_infected_comparison(
        beta_results,
        "VSIRS - influence of transmission rate beta",
        "vsirs_beta_comparison.png",
    )

    kappa_results = []
    for kappa in [1e-3, 1e-2, 1e-1]:
        params = VSIRSParameters(
            beta=1e-7,
            gamma=1e-1,
            kappa=kappa,
            alpha=4e-9,
            susceptible_0=80_000_000.0,
            infected_0=100_000.0,
        )
        time, states = simulate_vsirs(params, final_time=2_500, dt=0.02)
        kappa_results.append((f"kappa={kappa:g}", time, states, 2))
    plot_infected_comparison(
        kappa_results,
        "VSIRS - influence of vaccination rate kappa",
        "vsirs_kappa_comparison.png",
    )

    alpha_results = []
    for alpha in [9e-3, 9e-2, 9e-1]:
        params = VSIRSParameters(
            beta=1e-7,
            gamma=1e-1,
            kappa=1e-2,
            alpha=alpha,
            susceptible_0=80_000_000.0,
            infected_0=100_000.0,
        )
        time, states = simulate_vsirs(params, final_time=1_200, dt=0.005)
        alpha_results.append((f"alpha={alpha:g}", time, states, 2))
    plot_infected_comparison(
        alpha_results,
        "VSIRS - influence of immunity-loss rate alpha",
        "vsirs_alpha_comparison.png",
    )


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    run_sir_examples()
    run_cuir_examples()
    run_vsirs_examples()

    r0 = basic_reproduction_number(beta=1e-7, susceptible_0=80_000_000.0, gamma=0.9)
    threshold = herd_immunity_threshold(population=80_000_001.0, reproduction_number=r0)
    print(f"Figures written to: {FIGURES_DIR}")
    print(f"Example SIR R0: {r0:.3f}")
    print(f"Example herd immunity threshold: {threshold:,.0f} individuals")


if __name__ == "__main__":
    main()
