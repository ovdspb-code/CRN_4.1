# CRN – Step 2 Reproducibility Bundle (artifact-first)

This folder is a *minimal, reviewer-facing* reproducibility snapshot for Step 2 of the
alternative protocol: move from verbal conclusions to runnable code + numeric artifacts.

It contains two blocks:

1) `vendor/R12-CROWN/`
   The canonical CRN GKSL simulation suite (as referenced in the manuscript supplements).
   Run it to regenerate its own CSV/PNG outputs and validation report.

2) `step2_cortex_transport/`
   A compact toy experiment on an L2/3-like recurrent microcircuit (SBM with 3 assemblies)
   comparing barrier-limited thermal hopping vs the maximal-permeability limit
   (`T_env -> inf`, energies ignored).

## Quick start

From the root of this bundle:

```bash
python -m pip install -r env/requirements.txt

# 1) R12-CROWN (CRN core suite)
cd vendor/R12-CROWN
python run_all.py
python validate_checkpoints.py
cd ../..

# 2) Step-2 cortex toy experiment
python step2_cortex_transport/run_cortex_transport.py --trials 100 --t_end 100 --seed 1
```

Outputs:

- `vendor/R12-CROWN/outputs/` (PNG + CSV + validation_report.csv)
- `step2_cortex_transport/outputs/` (trial-level CSV + summary CSV + PNG plots + metadata.json)

## Notes for interpretation (important)

- The cortex experiment is a *linear transport proxy*, not a full attractor network.
  It tests leakage/confinement under different permeability assumptions, but it does
  not model recurrent nonlinear stabilization (thresholds, inhibition, etc.).

- Approximating “CRN” as `T_env -> inf` is an intentionally extreme limit used to
  reproduce the “barriers turned off” intuition. It is not the full CRN/GKSL wave model.

