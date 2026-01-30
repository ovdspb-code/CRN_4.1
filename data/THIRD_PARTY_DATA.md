# Third‑party datasets (provenance & terms)

This project uses connectome datasets from external publications and services.

We do **not** claim any rights over those datasets. Redistribution on GitHub may be subject to the original terms.

Recommended practice in this repo
- Keep only **processed/derived adjacency** and **analysis outputs** under `data/derived/`.
- Provide **download pointers and citations** for any raw third‑party sources.
- When in doubt, **do not vendor raw third‑party tables**; vendor only derived/aggregated outputs and provide a deterministic download + preprocessing script.

Datasets referenced in this work

1) **C. elegans (touch subcircuit)**
- The CRN benchmark here is a small touch→interneuron→motor subcircuit that is stored explicitly as an edge list (`experiments/elegans_touch_circuit/touch_circuit_edges.csv`).
- If you want extraction to be reproducible from a public full‑connectome source (e.g., Varshney et al. 2011), add an extraction script and cite the exact neuron set + selection criteria.

2) **Drosophila larva connectome subsystem**
- The Bridge‑A scripts fetch a Drosophila larva connectome dataset via Netzschleuder (see `experiments/bridgeA_stepA1_real_connectome/`).
- This repo vendors only derived outputs (CSV backends and plots) under `data/derived/drosophila/`.

3) **Mouse cortex proxy**
- The mouse results use a stochastic block model proxy (SBM hierarchical feed‑forward), not a redistributed mouse connectome adjacency matrix.

Before making the repository public, verify that redistribution of any *raw* supplementary tables (if any are added later) is permitted by the original licenses/terms.
