# CRN — Supplementary Simulation Artifacts (R12-CROWN)

This repository contains a fully reproducible simulation suite, figure-generation scripts, and precomputed numerical
artifacts supporting the accompanying CRN manuscript:

“Thermodynamic Advantage of Transient Coherent Dynamics in Hierarchical Decision Architectures: Coherent Resonant Netting”.

Included:

- **R12-CROWN/** — main simulation suite (code + plotting scripts + precomputed outputs)
- **R12-CROWN-LargeN-N1000/** — large‑N stress test (code + precomputed outputs)
- **artifacts/** — convenience mirror of precomputed outputs (figures/CSVs/metadata in one place)

Precomputed artifacts include:

- PNG figures (plots used in the paper and Supplementary Materials)
- CSV tables with numerical results
- JSON metadata (run configs + seeds)
- `validation_report.csv` (PASS/FAIL checkpoint summary)

## Quick reproduction (main suite)

From `R12-CROWN/`:

```bash
python run_all.py
```

Outputs will be written to `./outputs/`.

## Quick reproduction (large‑N, N=1000)

From `R12-CROWN-LargeN-N1000/`:

```bash
python run_largeN_stress_test.py --config config_largeN.json
```

Outputs will be written to `./outputs/`.

## Notes

- Runs are deterministic given the provided configs and seeds.
- See the README files inside each subfolder for additional details.
