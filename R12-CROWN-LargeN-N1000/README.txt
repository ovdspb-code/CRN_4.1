CRN — Supplement R12‑CROWN‑LargeN (N=1000 stress test) This package is intentionally self‑contained. It includes:
- run_largeN_stress_test.py
- config_largeN.json
- outputs/ (precomputed CSV/PNG + metadata) How to reproduce:
1) Install dependencies: numpy, scipy, pandas, matplotlib
2) From this folder run: python run_largeN_stress_test.py --config config_largeN.json What to expect:
- The script regenerates a 3‑panel figure: (A) ENAQT‑like non‑monotonic success vs dephasing (with FWHM window) (B) Reset bottleneck + energy tax ratio (coarse‑grained) (C) Threshold readout trace (self‑quenching sink), interpreted as an internal dissipative element. Implementation note:
- Large‑N uses a stochastic wave‑amplitude proxy (phase‑kick dephasing) to avoid density‑matrix O(N^2) scaling.
- Success is computed as P_success(t) = 1 − ||psi(t)||^2 (norm loss into the sink).
