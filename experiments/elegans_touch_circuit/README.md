# C. elegans touch subcircuit (CRN supporting experiment)

This folder contains a small, explicitly specified *touch → interneuron → motor* subcircuit used as a reproducibility‑friendly C. elegans benchmark.

What is included
- `crn_nematode_pipeline.py` — end‑to‑end runner for the C. elegans sweeps (GKSL proxy vs baselines).
- `touch_circuit_edges.csv` — the exact edge list (with edge class and weights) used by the pipeline.
- CSV outputs used as numeric backends:
  - `crn_elegans_thermal_sweep.csv` — performance vs environment temperature (T_env); reports the ~1.39× regime at low T_env.
  - `crn_elegans_disorder_sweep.csv`, `crn_elegans_target_sensitivity.csv`, `crn_elegans_main_test.csv` — supplementary diagnostics.
- `elegans_touch_metrics.json` — basic graph metrics and bookkeeping.

How to run
```bash
python experiments/elegans_touch_circuit/crn_nematode_pipeline.py
```

Notes on provenance and scope
- This is **not** the full C. elegans connectome. It is a small subcircuit intended to test *targeted transport* under constraints.
- Neuron naming follows standard C. elegans conventions (e.g., ALML/ALMR/AVM/PLM, AVA/AVB/AVD/PVC).
- The circuit is stored explicitly (edge list + weights) to avoid hidden preprocessing steps and to make the benchmark deterministic.

If you need the extraction to be *strictly* reproducible from a public full‑connectome source (e.g., Varshney et al. 2011), add a dedicated extraction script and cite the exact neuron set + selection criteria.
