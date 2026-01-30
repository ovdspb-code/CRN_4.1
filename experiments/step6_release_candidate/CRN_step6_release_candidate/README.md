# CRN alternative protocol — Step 6 (release candidate)

This directory is the **single-archive, reviewer-facing release candidate** for the *alternative protocol* developed to test the (architectural) evolutionary advantage hypothesis for a CRN-like mechanism.

It packages complete artifacts for:
- **Step 2** — Cortex microcircuit toy: permeability vs confinement under strong disorder.
- **Step 3** — Architecture trade-off: transport objective vs recurrent memory objective.
- **Step 4** — Evolutionary selection model: when *modular regime separation* is selected.
- **Step 5** — Robustness audit: stability of the selection boundary to proxy/metric choices.
- **Step 6** — This release wrapper: one archive, integrity manifest, and paper-ready tables.

The release contains both **code** and **numeric backends** (CSV) + **figures** (PNG).  
The CSV tables are the primary citation targets for a manuscript/SI.

---

## Directory structure

- `steps/`
  - `step2_repro_bundle/CRN_step2_repro_bundle/`
    - `step2_cortex_transport/run_cortex_transport.py`
    - `step2_cortex_transport/outputs/*.csv,*.png`
    - `vendor/R12-CROWN/` (GKSL transport/energy baseline suite)

  - `step3_architecture_tradeoff_bundle/CRN_step3_bundle/`
    - `step3_architecture_tradeoff/run_step3_architecture_tradeoff.py`
    - `step3_architecture_tradeoff/outputs/*.csv,*.png`

  - `step4_evolutionary_selection_bundle/`
    - `step4_evolutionary_selection/run_step4_evolutionary_selection.py`
    - `step4_evolutionary_selection/inputs/*.csv`
    - `step4_evolutionary_selection/outputs/*.csv,*.png`

  - `step5_consolidated_evidence_bundle/CRN_step5_consolidated_evidence_bundle/`
    - `code/run_step5_robustness_audit.py`
    - `step2/outputs/*` (copied numeric backends)
    - `step3/outputs/*` (copied numeric backends)
    - `step4/outputs/*` (copied numeric backends)
    - `step5_robustness_audit/inputs/*, outputs/*`

- `paper/`
  - `tables/` — convenience copies of the most-cited CSVs.
  - `figures/` — convenience copies of the key PNGs.

- `env/requirements.txt` — minimal Python requirements.

- `MANIFEST_SHA256.txt` — integrity manifest for the whole release.

---

## How to reproduce (local)

Create a clean Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r env/requirements.txt
```

Then run each step bundle in-place:

**Step 2**
```bash
python steps/step2_repro_bundle/CRN_step2_repro_bundle/step2_cortex_transport/run_cortex_transport.py
```

**Step 3**
```bash
python steps/step3_architecture_tradeoff_bundle/CRN_step3_bundle/step3_architecture_tradeoff/run_step3_architecture_tradeoff.py
```

**Step 4**
```bash
python steps/step4_evolutionary_selection_bundle/step4_evolutionary_selection/run_step4_evolutionary_selection.py
```

**Step 5**
```bash
python steps/step5_consolidated_evidence_bundle/CRN_step5_consolidated_evidence_bundle/code/run_step5_robustness_audit.py
```

Note: Step 3 requires `transport_gap_sweep.csv` to be present in its `outputs/` directory.  
This file is included and was generated with the included `vendor/R12-CROWN` suite.

---

## Paper-ready backends (most useful files)

If you only need the numbers (not the code), the following CSVs are the core citation targets:

- Step 2 (cortex toy):
  - `paper/tables/step2_cortex_transport_trials.csv`
  - `paper/tables/step2_cortex_transport_summary.csv`

- Step 3 (trade-off table):
  - `paper/tables/step3_combined_tradeoff.csv`
  - `paper/tables/step3_memory_summary.csv`
  - `paper/tables/step3_transport_gap_sweep.csv`

- Step 4 (phase diagram backend):
  - `paper/tables/step4_best_strategy_grid.csv`

- Step 5 (robustness audit backends):
  - `paper/tables/step5_robustness_consensus_grid.csv`
  - `paper/tables/step5_boundary_compare_step4_vs_consensus.csv`
  - `paper/tables/key_numbers.csv`

