# Epidemic Spread Modeling with SIR, CUIR and VSIRS Models

Academic applied mathematics project on deterministic epidemic modeling with ordinary differential equations.

The report studies a baseline SIR model and two extensions:

- `SIR`: susceptible, infected and recovered individuals;
- `CUIR`: confined, unconfined, infected and recovered individuals;
- `VSIRS`: vaccinated, susceptible, infected and recovered individuals, with loss of immunity from `R` back to `S`.

The numerical simulations use the explicit Euler scheme described in the report:

```text
y[n + 1] = y[n] + dt * f(y[n])
```

## Content

- `Epidemic_Spread_Modeling_with_SIR_CUIR_and_VSIRS_Models.pdf`: original TER report.
- `epidemic_spread_models.py`: Python implementation of the SIR, CUIR and VSIRS models.
- `requirements.txt`: minimal Python dependencies.
- `figures/`: generated simulation figures.

## Models Implemented

### SIR

```text
dS/dt = -beta * S * I
dI/dt = beta * S * I - gamma * I
dR/dt = gamma * I
```

The script also computes:

- the basic reproduction number `R0 = beta * S0 / gamma`;
- the herd-immunity threshold `(1 - 1 / R0) * N` when `R0 > 1`.

### CUIR

```text
dC/dt = -beta_c * I * C
dU/dt = -beta_u * I * U
dI/dt = beta_c * I * C + beta_u * I * U - gamma * I
dR/dt = gamma * I
```

This model compares the effect of confined and unconfined susceptible populations, with `beta_c < beta_u`.

### VSIRS

```text
dV/dt = kappa * S
dS/dt = -kappa * S - beta * I * S + alpha * R
dI/dt = beta * I * S - gamma * I
dR/dt = gamma * I - alpha * R
```

This model adds vaccination and a return from recovered to susceptible individuals through loss of immunity.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the figures:

```bash
python epidemic_spread_models.py
```

The figures are written to `figures/`.

## Notes

The report uses explicit Euler simulations and several parameter studies. The script keeps this method and reproduces the main modeling logic with stable numerical scenarios inspired by the report parameters.
