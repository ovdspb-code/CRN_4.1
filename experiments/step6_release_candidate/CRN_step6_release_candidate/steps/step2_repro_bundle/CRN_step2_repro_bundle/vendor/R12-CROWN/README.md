# CRN Simulation Suite (R12-CROWN)

This folder contains a lightweight, portable simulation suite supporting the accompanying CRN manuscript.

## Files

- `qrn_core.py` — GKSL (Lindblad) engine on a graph with node-local dephasing and an explicit sink state `|s⟩`.
- `qrn_analysis.py` — spectral gap sweep, robustness window widths, threshold-sink robustness, and checkpoint validation.
- `qrn_viz.py` — plotting helpers (PNG).
- `run_all.py` — runs the suite and writes figures + CSV tables to `./outputs`.
- `compare_baselines.py` — generates Supplementary Figure S6 (CRN vs classical random walk baseline) and writes outputs to `./outputs`.
- `compare_baselines_barrier.py` — generates Supplementary Figure S6-2 (barrier-limited baseline) and writes outputs to `./outputs`.
- `validate_checkpoints.py` — PASS/FAIL validation for key numerical checkpoints (see `checkpoints.yaml`).
- `MANIFEST.md` — file manifest and integrity hashes for key outputs.

## Dependencies

- Python
- `numpy`
- `scipy`
- `matplotlib`

## Quick start

From inside this folder:

```bash
python run_all.py
```

Baseline comparison used in the manuscript supplement:

```bash
python compare_baselines.py
```

Additional barrier-limited baseline:

```bash
python compare_baselines_barrier.py
```

Outputs appear in `./outputs/`.

## Notes

- The sink is implemented as an explicit basis state, keeping the GKSL dynamics trace-preserving.
- The provided outputs are also included under `./outputs/` for convenience; rerunning regenerates them from source.
