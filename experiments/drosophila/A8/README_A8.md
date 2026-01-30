# A8: Objective sweep over (lambda, chi)

This step is a **post-processing** layer over A7 outputs.

## Inputs
Provide `--a7_dir` pointing to a folder that contains:

- `A7_gksl_sweep.csv`
- `A7_baselines.csv`
- `A7_meta.json`

## What it does
For each `(lambda, chi, epsilon)` it:

- builds the GKSL mean curve over `kappa` and selects:
  - `GKSL_bestIPC`  (max InfoPerCost)
  - `GKSL_bestU`    (max Utility)
  - `GKSL_bestSel`  (max Selectivity_end under mean P_sink_T_end >= pT_min)

- reports **trialwise** mean/std/median at the chosen `kappa` (useful for error bars)

- computes the same objectives for baselines:
  - CRW
  - CRW_thermal at selected `T_env` values (default: 0.1 and 1.0)

## Outputs
Writes to `--out_dir`:

- `A8_objective_sweep.csv`
- `A8_meta.json`

## Example
```bash
python3 A8_sweep_objectives_energy.py \
  --a7_dir  /path/to/A7_OUT \
  --out_dir /path/to/A8_OUT \
  --lambda_grid 0.5,1.0,2.0 \
  --chi_grid 0.001,0.01,0.05 \
  --pT_min 0.005
```

Notes:
- `Selectivity_end` and `InfoPerCost` are nonlinear; this script reports both the mean-based optimum and trialwise stats.
