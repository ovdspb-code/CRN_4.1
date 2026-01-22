# CRN Simulation Suite (R12-CROWN)

This folder contains a lightweight, portable simulation suite supporting the accompanying CRN manuscript.

## Files

- `qrn_core.py` — GKSL (Lindblad) engine on a graph with node-local dephasing and an explicit sink state `|s⟩`.
- `qrn_analysis.py` — spectral gap sweep, robustness window widths, threshold-sink robustness.
- `qrn_energy.py` — coarse-grained energy checkpoint / reset-bottleneck sweep.
- `qrn_viz.py` — plotting helpers (PNG).
- `run_all.py` — runs the suite and writes figures + CSV tables to `./outputs`.
- `validate_checkpoints.py` — PASS/FAIL validation for key numerical checkpoints (see `checkpoints.yaml`).
- `MANIFEST.md` — file manifest and integrity hashes for key outputs.

## Dependencies

- Python **3.10–3.11 recommended** (macOS: avoids slow/failed SciPy builds)
- `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`

Install (from repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Quick start

From inside this folder:

```bash
python run_all.py
python validate_checkpoints.py
```

Outputs appear in `./outputs/`.

## Notes

- The sink is implemented as an explicit basis state, keeping the GKSL dynamics trace-preserving.
- The provided outputs are also included under `./outputs/` for convenience; rerunning regenerates them from source.
