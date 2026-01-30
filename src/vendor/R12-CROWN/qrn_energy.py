"""QRN Energy Module (Release 4.0)

Implements the coarse-grained throughput / regime / energy-tax calculations
introduced in the Release-4 Roadmap steps R2/R3 (reset bottleneck) and R1
(energy checkpoint).

This module is intentionally lightweight: it does not attempt to infer the
wave-layer power tax P_tax from the GKSL toy model. Instead, it exposes a
transparent mapping from externally supplied parameters (P_tax, E_comm,
timescales) to the key derived metrics used in the paper:

  - Effective cycle time: tau_cycle = tau_trans + tau_fix + tau_reset
  - Fixation throughput:  f_fix = 1 / tau_cycle
  - Energy tax per act:   E_tax = P_tax / f_fix = P_tax * tau_cycle
  - Checkpoint ratio:     E_tax / E_comm

The regime label is based on the ratio r = tau_reset / tau_trans, as used
in the substrate-narrowing discussion.

Dependencies: numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyConfig:
    """Config for the coarse-grained energy audit."""

    # Reference cost of one fixation/broadcast event (J, or normalized units).
    E_comm: float = 1.0
    # Power tax of the wave layer (J/s), per effective cycle.
    P_tax: float = 1e-3
    # Fixed amplification / commit latency (s). (Placeholder, but explicit.)
    tau_fix: float = 1e-3

    # Regime thresholds for r = tau_reset / tau_trans.
    r_reset_limited: float = 10.0
    r_transport_limited: float = 0.1


class EnergyCalculator:
    """Compute throughput regimes and energy-tax checkpoints."""

    def __init__(self, cfg: Optional[EnergyConfig] = None):
        self.cfg = cfg if cfg is not None else EnergyConfig()

        if self.cfg.E_comm <= 0:
            raise ValueError("E_comm must be positive")
        if self.cfg.P_tax < 0:
            raise ValueError("P_tax must be non-negative")
        if self.cfg.tau_fix < 0:
            raise ValueError("tau_fix must be non-negative")

    def calculate_metrics(self, tau_trans: float, tau_reset_list: Iterable[float]) -> pd.DataFrame:
        """Sweep tau_reset and return a table with throughput and energy checkpoints."""
        tau_trans = float(tau_trans)
        if tau_trans <= 0:
            raise ValueError("tau_trans must be positive")

        rows = []
        for tau_r in tau_reset_list:
            tau_r = float(tau_r)
            if tau_r <= 0:
                raise ValueError("tau_reset values must be positive")

            tau_cycle = tau_trans + self.cfg.tau_fix + tau_r
            f_fix = 1.0 / tau_cycle
            r = tau_r / tau_trans

            if r > self.cfg.r_reset_limited:
                regime = "Reset-Limited"
            elif r < self.cfg.r_transport_limited:
                regime = "Transport-Limited"
            else:
                regime = "Transitional"

            E_tax = self.cfg.P_tax * tau_cycle
            tax_ratio = E_tax / self.cfg.E_comm
            rows.append(
                {
                    "tau_reset": tau_r,
                    "tau_cycle": tau_cycle,
                    "f_fix": f_fix,
                    "r": r,
                    "regime": regime,
                    "E_tax": E_tax,
                    "tax_ratio": tax_ratio,
                    "red_line_status": self.get_red_line_status(tax_ratio),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def get_red_line_status(tax_ratio: float) -> str:
        """Map E_tax/E_comm to the agreed graded status labels.

        Decision adopted earlier in the patchlog discussion:
          - Red line remains 1e-1 (hard de-scoping threshold).
          - 1e-2–1e-1 is allowed but explicitly "marginal".
        """
        tax_ratio = float(tax_ratio)
        if tax_ratio < 1e-2:
            return "Safe (strong advantage)"
        if tax_ratio < 1e-1:
            return "Marginal (advantage persists, but small; requires Q3/Q4 tightening)"
        return "Red line (>= 0.1; de-scope energy-advantage claim)"
