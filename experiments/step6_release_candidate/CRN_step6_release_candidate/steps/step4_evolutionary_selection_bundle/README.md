# CRN – Step 4 Bundle: Evolutionary selection pressure for modular CRN-like architectures

This is Step 4 of the alternative protocol. It upgrades the Step-3 “trade-off picture” into an
explicit **selection model**:

- If the environment demands both (i) disordered-graph transport (where CRN/GKSL has an
  intermediate-noise optimum) and (ii) recurrent memory confinement (where high permeability
  causes leakage),
- and if there is any **non-zero cost** to separating regimes (gating, regulation, modular control),
  then we can ask **when** modular separation is still selected.

## Inputs

Copied from Step 3:

- `step4_evolutionary_selection/inputs/combined_tradeoff.csv`
- `step4_evolutionary_selection/inputs/memory_summary.csv`
- `step4_evolutionary_selection/inputs/transport_gap_sweep.csv`
- `step4_evolutionary_selection/inputs/transport_energy_sweep.csv` (optional annotation support)
- `step4_evolutionary_selection/inputs/metadata_step3.json`

## How to run

From the root of this bundle:

```bash
python -m pip install -r env/requirements.txt
python step4_evolutionary_selection/run_step4_evolutionary_selection.py
```

## Main outputs (reviewer-facing)

- `outputs/phase_diagram_modular_vs_global.png`
  A phase diagram over:
  - x-axis: `p_transport` (fraction of transport-like demands)
  - y-axis: `modularity_cost` (fitness penalty for having separate local knobs)

- `outputs/best_strategy_grid.csv`
  The numeric backend for the phase diagram:
  best global strategy, best modular strategy, and the winner.

- `outputs/replicator_dynamics_example.png` + `outputs/replicator_dynamics_example.csv`
  A deterministic replicator-mutation demo showing which strategy class takes over under selection.

## Interpretation boundary

This step **does not** claim a specific microscopic substrate.
It is a compact way to convert the Step-3 trade-off into an evolutionary statement:
*local regime separation* is selected when both objectives matter and the gating/regulation
cost is not prohibitive.
