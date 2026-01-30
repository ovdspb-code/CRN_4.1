# Data & code availability (template)

Use / adapt the following for the manuscript’s “Data and Code Availability” section.

Code
- The full analysis code and derived figure backends are available in the CRN reproducibility repository (GitHub): <REPO_URL>.
- An archival snapshot of a tagged release is deposited on Zenodo: <ZENODO_CODE_DOI_CONCEPT> (concept DOI) / <ZENODO_CODE_DOI_VERSION> (version DOI).

Data and numeric backends
- Drosophila larva connectome results: derived numeric backends and plots are included in this repository under `data/derived/drosophila/`. Raw connectome tables are obtained programmatically via the Bridge‑A scripts (`experiments/bridgeA_stepA1_real_connectome/`) from publicly available sources (see `data/THIRD_PARTY_DATA.md`).
- C. elegans touch circuit benchmark: the exact edge list and sweep outputs are included under `experiments/elegans_touch_circuit/` (e.g., `touch_circuit_edges.csv`, `crn_elegans_thermal_sweep.csv`).
- Mouse cortex proxy benchmark: the mouse result uses a stochastic block model proxy (SBM hierarchical feed‑forward) generated on the fly; scripts and backends are included under `experiments/step6_release_candidate/.../step2_cortex_transport/`.
- Companion evolutionary game theory dataset: archived on Zenodo as **10.5281/zenodo.18379851** and mirrored in this repository under `datasets/zenodo_18379851_evolutionary_game/`.

Reproducibility notes
- The repository contains a `MANIFEST_SHA256.txt` file for integrity checks of the released bundle.
- Python dependencies are listed in `env/requirements.txt`.
