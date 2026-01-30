# CRN – Step 3 Bundle: Architecture-level Trade-off (transport vs recurrent memory)

This bundle is Step 3 of the alternative protocol: move from a **single-task** toy result
to an **architecture-level** test that can speak to an evolutionary argument.

Core claim being tested (operational)
-------------------------------------
If the environment requires both:

1) fast/robust *transport* across a disordered graph (where CRN/GKSL exhibits an ENAQT-like optimum), and
2) *confinement/selective pattern completion* inside recurrent microcircuits (where high permeability causes leakage),

then **a single global “noise/permeability” regime cannot optimize both tasks at once**.
A modular architecture that can *locally* tune the regime (or gate the wave layer) can dominate.

What is inside
--------------
- `step3_architecture_tradeoff/outputs/transport_gap_sweep.csv`
  Copied from the canonical R12-CROWN outputs (Liouvillian gap g(kappa) vs dephasing kappa).

- `step3_architecture_tradeoff/outputs/memory_trials.csv`
  Trial-level data for the recurrent microcircuit (SBM) transport proxy, swept over `T_env`.

- `step3_architecture_tradeoff/outputs/memory_summary.csv`
  Mean ± CI summaries per `T_env`.

- `step3_architecture_tradeoff/outputs/combined_tradeoff.csv`
  Joined table for the shared noise grid: normalized transport score + bounded memory score.

- `step3_architecture_tradeoff/outputs/fitness_weight_sweep.csv`
  Sensitivity scan showing modular advantage for intermediate weightings.

- Plots (PNG) in the same folder.

How to reproduce
----------------
Install deps and run the script:

```bash
python -m pip install -r env/requirements.txt
python step3_architecture_tradeoff/run_step3_architecture_tradeoff.py
```

Key outputs to look at
----------------------
- `pareto_tradeoff_scatter.png`:
  points show what is achievable with a single global knob.
- `modular_gain_vs_weight.png`:
  modular architecture outperforms the best single-knob solution whenever both objectives matter.
