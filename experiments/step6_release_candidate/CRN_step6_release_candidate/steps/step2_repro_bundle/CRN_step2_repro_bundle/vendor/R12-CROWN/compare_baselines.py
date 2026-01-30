"""Baseline comparison against a classical random walk (CRW).

This script generates Supplementary Figure S6 used in the manuscript:
- Coherent resonant netting (CRN) / GKSL wave dynamics on a graph
- Classical continuous-time random walk (CRW) on the same graph

The purpose is to demonstrate that ENAQT-like, noise-assisted transport in CRN
is not reducible to generic classical diffusion: in the coherent limit
(\kappa \to 0) the wave dynamics can exhibit interference trapping (dark-state
subspace), which is absent in the classical baseline.

Outputs (written to ./outputs):
  - S6_baseline_comparison.png
  - S6_baseline_comparison.csv
  - S6_baseline_metadata.json

Usage:
  python compare_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from qrn_analysis import QRNAnalyzer
from qrn_core import QRNSystem


def _adjacency_from_hamiltonian_raw(raw: np.ndarray) -> np.ndarray:
    """Convert the internal 'raw' matrix (-Adjacency) into Adjacency."""
    # In this release, TopologyGenerator returns (-1)*Adjacency on off-diagonals.
    A = -np.real(raw)
    A[A < 0] = 0.0
    # Ensure zero diagonal
    np.fill_diagonal(A, 0.0)
    return A


def run_classical_random_walk(
    A: np.ndarray,
    gamma: float,
    eta: float,
    target_node: int,
    t_max: float,
    n_t: int,
    initial_node: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time classical random walk with an explicit absorbing sink.

    We use the generator:
        dP/dt = Q P,   Q = -gamma * L
    where L is the graph Laplacian (D - A).

    Sink coupling is modeled as an additional absorbing state s with
    transition rate eta from the target node to the sink.
    """
    A = np.asarray(A, dtype=float)
    N = int(A.shape[0])
    if A.shape != (N, N):
        raise ValueError("Adjacency A must be square")

    # Laplacian L = D - A
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    # Node generator
    Q_nodes = -float(gamma) * L

    # Augment with sink state
    M = np.zeros((N + 1, N + 1), dtype=float)
    M[:N, :N] = Q_nodes

    target_node = int(target_node)
    if not (0 <= target_node < N):
        raise ValueError("target_node out of range")

    # Absorption: target -> sink at rate eta
    M[target_node, target_node] -= float(eta)
    M[N, target_node] += float(eta)

    # Time grid
    t_eval = np.linspace(0.0, float(t_max), int(n_t))

    # Initial distribution
    P0 = np.zeros((N + 1,), dtype=float)
    P0[int(initial_node)] = 1.0

    # Use expm_multiply to compute P(t) for all t in t_eval
    # expm_multiply expects a sparse matrix for efficiency
    M_sp = csr_matrix(M)

    P_traj = expm_multiply(M_sp, P0, start=0.0, stop=float(t_max), num=int(n_t), endpoint=True)
    P_traj = np.asarray(P_traj)  # (n_t, N+1)

    P_sink = P_traj[:, N]
    return t_eval, P_sink


def run_crn_traces(
    N: int,
    topology: str,
    gamma: float,
    eta: float,
    kappa_values,
    t_max: float,
    n_t: int,
    topology_kwargs=None,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """Return P_success(t) traces for selected kappa values."""
    sys = QRNSystem(N=N, topology=topology, gamma=gamma, topology_kwargs=topology_kwargs)
    sys.add_sink(target_node=N - 1, eta_base=eta, threshold_type="linear")

    rho0 = np.zeros((sys.d, sys.d), dtype=complex)
    rho0[0, 0] = 1.0
    y0 = rho0.reshape((-1,), order="F")

    t_eval = np.linspace(0.0, float(t_max), int(n_t))

    traces: Dict[float, np.ndarray] = {}
    for k in kappa_values:
        sys.add_dephasing(float(k))
        L = sys.liouvillian_superoperator(None)
        Y = expm_multiply(L, y0, start=0.0, stop=float(t_max), num=int(n_t), endpoint=True)
        Y = np.asarray(Y).T  # (d^2, n_t)
        traces[float(k)] = sys.success_probability_trace(Y)

    return t_eval, traces


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # --- Configuration (matches the manuscript's baseline-control intent) ---
    N = 10
    topology = "lollipop"
    topology_kwargs = {"ring_size": 6}

    gamma = 2.0
    eta = 1.0

    t_max = 20.0
    n_t = 400

    # Pick representative dephasing values:
    # - coherent limit (interference trapping)
    # - near-optimal (ENAQT-like NAT)
    # - over-damped / Zeno-like
    kappa_low = 0.0
    # Use the spectral gap optimum as an operational kappa* marker.
    kappa_grid = np.logspace(-3, 2, 21)
    sweep, _ = QRNAnalyzer.run_kappa_sweep(N=N, gamma=gamma, kappa_list=kappa_grid, topology=topology, eta=eta)
    kappa_opt = float(sweep.kappa[int(np.argmax(sweep.gap))])
    kappa_high = 10.0

    # --- CRN traces ---
    t, crn = run_crn_traces(
        N=N,
        topology=topology,
        topology_kwargs=topology_kwargs,
        gamma=gamma,
        eta=eta,
        kappa_values=[kappa_low, kappa_opt, kappa_high],
        t_max=t_max,
        n_t=n_t,
    )

    # --- Classical baseline (same topology, same sink rate) ---
    # We reuse the internal topology generator through QRNSystem to avoid drift.
    sys_tmp = QRNSystem(N=N, topology=topology, gamma=gamma, topology_kwargs=topology_kwargs)
    raw = sys_tmp.H[:N, :N] / gamma  # internal convention: H = gamma * (-Adjacency)
    A = _adjacency_from_hamiltonian_raw(raw)

    t_cl, P_cl = run_classical_random_walk(A=A, gamma=gamma, eta=eta, target_node=N - 1, t_max=t_max, n_t=n_t)

    # --- Plot ---
    fig = plt.figure(figsize=(8.5, 5.4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t_cl, P_cl, linestyle="--", linewidth=2, color="black", label="Classical random walk (CRW)")

    ax.plot(t, crn[kappa_low], linewidth=2, label=fr"CRN (coherent, $\kappa={kappa_low:g}$)")
    ax.plot(t, crn[kappa_opt], linewidth=2, label=fr"CRN (ENAQT-like, $\kappa^*\approx{kappa_opt:.3g}$)")
    ax.plot(t, crn[kappa_high], linewidth=2, label=fr"CRN (over-damped, $\kappa={kappa_high:g}$)")

    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$P_{success}(t)$")
    ax.set_title(
        "Baseline control against classical mimicry\n"
        f"Lollipop graph (N={N}), $\gamma={gamma}$, $\eta={eta}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="lower right")

    # Light annotations (kept minimal for Frontiers readability)
    ax.text(
        0.02,
        0.96,
        "Low-noise coherent trapping\nvs. classical diffusion",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    fig.tight_layout()
    fig_path = out_dir / "S6_baseline_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    # --- Save raw data for reproducibility ---
    csv_path = out_dir / "S6_baseline_comparison.csv"
    header = [
        "time",
        "P_success_CRW",
        f"P_success_CRN_kappa_{kappa_low:g}",
        f"P_success_CRN_kappa_{kappa_opt:.6g}",
        f"P_success_CRN_kappa_{kappa_high:g}",
    ]
    data = np.column_stack([t, P_cl, crn[kappa_low], crn[kappa_opt], crn[kappa_high]])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")

    meta = {
        "figure": "S6_baseline_comparison",
        "topology": topology,
        "topology_kwargs": topology_kwargs,
        "N": N,
        "gamma": gamma,
        "eta": eta,
        "t_max": t_max,
        "n_t": n_t,
        "kappa_low": kappa_low,
        "kappa_opt": kappa_opt,
        "kappa_high": kappa_high,
        "kappa_opt_determination": "argmax spectral gap g(kappa) over logspace(-3,2,21)",
        "initial_node": 0,
        "target_node": N - 1,
        "notes": "CRW uses continuous-time random walk with Laplacian generator and explicit absorbing sink; CRN uses GKSL with node-local dephasing and the same sink rate.",
    }
    (out_dir / "S6_baseline_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Saved:")
    print(f"  {fig_path}")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'S6_baseline_metadata.json'}")


if __name__ == "__main__":
    main()
