# Drosophila larva (Bridge‑A) — A7/A8 pipeline

This folder mirrors the workflow used to produce the Drosophila results in the core paper / SI.

## Inputs

The scripts expect an `A7_outputs/` folder produced by earlier steps (A7/A3) **and** the extracted connectome CSVs.
Because third‑party data redistribution may be restricted, this repo does not hard‑bundle the raw connectome tables.
See `data/THIRD_PARTY_DATA.md`.

## Minimal run order

1) A7 — disorder + thermal baselines

```bash
python experiments/drosophila/A7/A7_run_disorder_thermal_benchmark.py --help
```

2) A8 — objective sweep

```bash
python experiments/drosophila/A8/A8_sweep_objectives_energy.py --help
```

3) A8.4 — localization diagnostics

```bash
python experiments/drosophila/A8_4_5/A8_4_5_bundle_localization_arch/A8_4_localization_diagnostics.py --help
```

4) A8.5 — architecture dependence (original vs rewired vs lesion)

```bash
python experiments/drosophila/A8_4_5/A8_4_5_bundle_localization_arch/A8_5_architecture_dependence.py --help
```

5) A8.6 — postprocess across κ (ratio‑of‑means + P_good + pairwise tests)

```bash
python experiments/drosophila/A8_6/A8_6_postprocess_A85.py --help
```

## Outputs already included

For convenience, this repo includes the **derived CSV backends** and **example plots** from the completed run used in the manuscript.
See `data/derived/drosophila/` and `paper/figures_*`.
