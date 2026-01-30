Step 5 (robustness audit) – what this adds

This step stress-tests the Step-4 evolutionary “modularity selection” claim against alternative, reasonable
choices of transport and memory proxies derived from the same Step-3 numeric backends.

We evaluate 9 metric-pairs:

Transport proxies:
  - T_norm: normalized Liouvillian gap on the noise grid
  - T_gap_norm: min-max normalized raw gap
  - T_loggap_norm: min-max normalized log(gap)

Memory proxies:
  - M_frac: Recall/(Recall+Leakage) (bounded)
  - M_ratio: (Recall/Leakage)/(1+Recall/Leakage) (bounded)
  - M_diff_norm: min-max normalized (Recall−Leakage)

For each (p_transport, modularity_cost) we compute modular_win_rate ∈ {0/9,…,9/9}.

Core result:
  - Modular architectures remain preferred in a broad mid-range of p_transport even when demanding
    consensus across all 9 proxy choices.
  - A null control with *no tradeoff* (memory curve set equal to transport curve) shows modularity
    never helps (only ties at cost=0, otherwise loses), so the effect is not a coding artifact.

Key outputs:
  - phase_diagram_consensus.png
  - boundary_curves.png
  - robustness_consensus_grid.csv
  - boundary_compare_step4_vs_consensus.csv
