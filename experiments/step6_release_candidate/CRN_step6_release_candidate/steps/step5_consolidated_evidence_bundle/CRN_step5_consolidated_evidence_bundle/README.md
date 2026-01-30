# CRN alternative protocol — Step 5 (consolidated evidence bundle)

This bundle is the *consolidated, reviewer-facing* artifact pack for Steps 2–5 of the alternative protocol:

- Step 2: Cortex microcircuit toy (leakage vs confinement under permeability)
- Step 3: Architecture-level trade-off (transport vs recurrent memory)
- Step 4: Evolutionary selection model (when modular separation is selected)
- Step 5: Robustness audit (is Step 4 stable to proxy/metric choices?)

Directory structure
-------------------
- env/requirements.txt
  Minimal python deps.

- step2/outputs/
  Numeric artifacts from the cortex toy (trial-level CSV + summary CSV + plots).

- step3/outputs/
  Numeric backends for the trade-off:
  - transport_gap_sweep.csv
  - memory_trials.csv + memory_summary.csv
  - combined_tradeoff.csv
  - plots/*.png

- step4/outputs/
  Evolutionary selection outputs:
  - best_strategy_grid.csv (numeric backend for the phase diagram)
  - phase_diagram_modular_vs_global.png
  - replicator_dynamics_example.csv/.png
  - etc.

- step5_robustness_audit/
  - inputs/: copies of Step3 backends + Step4 grid (for comparison)
  - outputs/: Step5 robustness outputs + plots

- code/run_step5_robustness_audit.py
  Recomputes Step5 from the inputs folder.

Quick reproduce Step 5
----------------------
From the root of this bundle:

```bash
python -m pip install -r env/requirements.txt
python code/run_step5_robustness_audit.py
```

Main Step 5 outputs (for figures/tables)
----------------------------------------
- step5_robustness_audit/outputs/phase_diagram_consensus.png
  Heatmap of modular-win fraction across 9 proxy pairs.

- step5_robustness_audit/outputs/boundary_curves.png
  Cost-tolerance boundary curves (default Step4 vs consensus thresholds).

- step5_robustness_audit/outputs/robustness_consensus_grid.csv
  The numeric backend for the consensus phase diagram.

- step5_robustness_audit/outputs/boundary_compare_step4_vs_consensus.csv
  A compact table that is easy to cite in the manuscript.

Notes / interpretation boundary
-------------------------------
- Step 2 is a *linear transport proxy* (not a full nonlinear attractor network).
- Step 4–5 are *architecture-level* evolutionary arguments; they do not assert a specific microscopic carrier.
