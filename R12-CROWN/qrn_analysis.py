"""QRN Analysis Module (Release 4.0)

Computes spectral and dynamical diagnostics used in the Release-4 simulation
supplement:
- Liouvillian spectral gap g(gamma, kappa)
- Robustness window width W_kappa
- Targeting traces P_success(t) and E[V](t)
- Nonlinear sink robustness proxy (drop relative to linear sink)
- Lightweight validation report card (PASS/FAIL/N/A)

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply

from qrn_core import QRNSystem


@dataclass
class SweepResult:
    kappa: np.ndarray
    gap: np.ndarray


class QRNAnalyzer:
    @staticmethod
    def calculate_spectral_gap(system: QRNSystem, kappa: float, tol: float = 1e-12) -> Tuple[float, np.ndarray]:
        """Compute the Liouvillian spectral gap g for a given dephasing kappa.

        g := min_{lambda != 0} |Re(lambda)|, where lambda are eigenvalues of the Liouvillian.

        Notes:
        - Gap is well-defined for linear systems.
        - For nonlinear (threshold) sinks, use the corresponding linear configuration.
        """
        system.add_dephasing(float(kappa))
        L = system.liouvillian_superoperator(None).toarray()
        evals = la.eigvals(L)

        re = np.abs(np.real(evals))
        re_sorted = np.sort(re)
        nonzero = re_sorted[re_sorted > tol]
        if nonzero.size == 0:
            return 0.0, evals
        return float(nonzero[0]), evals

    @staticmethod
    def run_kappa_sweep(
        N: int,
        gamma: float,
        kappa_list: Sequence[float],
        topology: str = "chain",
        eta: float = 1.0,
    ) -> Tuple[SweepResult, Dict[float, np.ndarray]]:
        """Run a kappa sweep (gap vs kappa) and store three representative eigen-clouds."""
        sys = QRNSystem(N=N, topology=topology, gamma=gamma)
        sys.add_sink(target_node=N - 1, eta_base=eta, threshold_type="linear")

        kappas = np.asarray(kappa_list, dtype=float)
        gaps = np.zeros_like(kappas)
        for i, k in enumerate(kappas):
            g, _ = QRNAnalyzer.calculate_spectral_gap(sys, float(k))
            gaps[i] = g

        eigen_clouds: Dict[float, np.ndarray] = {}
        if kappas.size >= 3:
            k_under = float(kappas[0])
            k_over = float(kappas[-1])
            k_opt = float(kappas[int(np.argmax(gaps))])
            for k in [k_under, k_opt, k_over]:
                _, evals = QRNAnalyzer.calculate_spectral_gap(sys, float(k))
                eigen_clouds[float(k)] = evals

        return SweepResult(kappa=kappas, gap=gaps), eigen_clouds

    @staticmethod
    def calculate_window_width(sweep: SweepResult, threshold_ratio: float = 0.5) -> float:
        """Robustness width W_kappa (in decades) where gap >= threshold_ratio * max(gap)."""
        max_val = float(np.max(sweep.gap))
        if max_val <= 0:
            return 0.0
        thr = threshold_ratio * max_val
        mask = sweep.gap >= thr
        if not np.any(mask):
            return 0.0
        k_min = float(np.min(sweep.kappa[mask]))
        k_max = float(np.max(sweep.kappa[mask]))
        if k_min <= 0 or k_max <= 0:
            return 0.0
        return float(np.log10(k_max / k_min))

    @staticmethod
    def simulate_targeting_traces(
        N: int,
        gamma: float,
        kappa_values: Sequence[float],
        topology: str = "chain",
        eta: float = 1.0,
        t_max: float = 20.0,
        n_t: int = 400,
        potential: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, Dict[float, Dict[str, np.ndarray]]]:
        """Simulate P_success(t) and E[V](t) traces for selected kappas."""
        sys = QRNSystem(N=N, topology=topology, gamma=gamma, potential=potential)
        sys.add_sink(target_node=N - 1, eta_base=eta, threshold_type="linear")

        rho0 = np.zeros((sys.d, sys.d), dtype=complex)
        rho0[0, 0] = 1.0
        y0 = rho0.reshape((-1,), order="F")

        t_eval = np.linspace(0.0, float(t_max), int(n_t))
        traces: Dict[float, Dict[str, np.ndarray]] = {}

        for k in kappa_values:
            sys.add_dephasing(float(k))
            L = sys.liouvillian_superoperator(None)
            Y = expm_multiply(L, y0, start=0.0, stop=float(t_max), num=int(n_t), endpoint=True)
            Y = np.asarray(Y).T  # (d^2, n_t)
            P = sys.success_probability_trace(Y)
            EV = sys.expected_potential_trace(Y)
            traces[float(k)] = {"P_success": P, "EV": EV}

        return t_eval, traces

    @staticmethod
    def efficiency_landscape(
        N: int,
        topology: str,
        gamma_grid: Sequence[float],
        kappa_grid: Sequence[float],
        eta: float,
        T: float,
        potential: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Compute P_success(T) over a coarse (gamma, kappa) grid."""
        gamma_grid = np.asarray(gamma_grid, dtype=float)
        kappa_grid = np.asarray(kappa_grid, dtype=float)
        out = np.zeros((gamma_grid.size, kappa_grid.size), dtype=float)

        rho0 = None
        y0 = None

        for i, g in enumerate(gamma_grid):
            sys = QRNSystem(N=N, topology=topology, gamma=float(g), potential=potential)
            sys.add_sink(target_node=N - 1, eta_base=eta, threshold_type="linear")

            if rho0 is None:
                rho0 = np.zeros((sys.d, sys.d), dtype=complex)
                rho0[0, 0] = 1.0
                y0 = rho0.reshape((-1,), order="F")

            for j, k in enumerate(kappa_grid):
                sys.add_dephasing(float(k))
                L = sys.liouvillian_superoperator(None)
                yT = expm_multiply(L * float(T), y0)
                rhoT = yT.reshape((sys.d, sys.d), order="F")
                out[i, j] = sys.sink_population(rhoT)

        return out

    @staticmethod
    def nonlinearity_drop(
        N: int,
        gamma: float,
        kappa: float,
        eta_base: float,
        eta_high: float,
        I_th: float,
        T: float,
        topology: str = "chain",
        potential: Optional[Sequence[float]] = None,
        n_t: int = 600,
    ) -> float:
        """Relative drop of P_success(T) when switching from linear to threshold sink."""
        # Linear
        sys_lin = QRNSystem(N=N, topology=topology, gamma=gamma, potential=potential)
        sys_lin.add_sink(target_node=N - 1, eta_base=eta_base, threshold_type="linear")
        sys_lin.add_dephasing(kappa)

        rho0 = np.zeros((sys_lin.d, sys_lin.d), dtype=complex)
        rho0[0, 0] = 1.0
        y0 = rho0.reshape((-1,), order="F")

        yT_lin = expm_multiply(sys_lin.liouvillian_superoperator(None) * float(T), y0)
        rhoT_lin = yT_lin.reshape((sys_lin.d, sys_lin.d), order="F")
        P_lin = sys_lin.sink_population(rhoT_lin)

        # Threshold (nonlinear): ODE integration
        sys_thr = QRNSystem(N=N, topology=topology, gamma=gamma, potential=potential)
        sys_thr.add_sink(
            target_node=N - 1,
            eta_base=eta_base,
            threshold_type="threshold",
            I_th=I_th,
            eta_high=eta_high,
            steepness=20.0,
        )
        sys_thr.add_dephasing(kappa)

        t_eval = np.linspace(0.0, float(T), int(n_t))
        _, Y = sys_thr.solve_dynamics((0.0, float(T)), rho0, t_eval=t_eval, method="RK23", rtol=1e-5, atol=1e-7)
        rhoT_thr = Y[:, -1].reshape((sys_thr.d, sys_thr.d), order="F")
        P_thr = sys_thr.sink_population(rhoT_thr)

        if P_lin <= 1e-12:
            return 0.0
        return float(max(0.0, (P_lin - P_thr) / P_lin))

    @staticmethod
    def generate_validation_report(
        width_dec: float,
        p_tax_ratio: Optional[float],
        nonlin_drop: Optional[float],
        width_bound: float = 0.5,
        tax_bound: float = 0.1,
        nonlin_bound: float = 0.3,
    ) -> List[Dict[str, str]]:
        """Minimal PASS/FAIL card.

        p_tax_ratio is not computed by the GKSL toy model; supply it from the
        energy-audit part of the release (R1/R3). If None, it is reported as N/A.
        """

        def _fmt(x: Optional[float], fmt: str) -> str:
            return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)

        def _status(x: Optional[float], op: str, bound: float) -> str:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "N/A"
            if op == "lt":
                return "PASS" if x < bound else "FAIL"
            if op == "gt":
                return "PASS" if x > bound else "FAIL"
            raise ValueError(op)

        return [
            {
                "ID": "Q3-Check",
                "Metric": "Energy Tax Ratio (E_tax/E_comm)",
                "Bound": f"< {tax_bound}",
                "Value": _fmt(p_tax_ratio, ".3f"),
                "Status": _status(p_tax_ratio, "lt", tax_bound),
            },
            {
                "ID": "P1-Robust",
                "Metric": "Window Width W_kappa (dec)",
                "Bound": f"> {width_bound}",
                "Value": _fmt(width_dec, ".2f"),
                "Status": _status(width_dec, "gt", width_bound),
            },
            {
                "ID": "R5-Sink",
                "Metric": "Threshold Sink Drop (rel)",
                "Bound": f"< {nonlin_bound}",
                "Value": _fmt(nonlin_drop, ".3f"),
                "Status": _status(nonlin_drop, "lt", nonlin_bound),
            },
        ]
