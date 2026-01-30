"""QRN Visualization Module (Release 4.0)

Generates publication-ready figures (PNG):
- Spectral eigenvalue clouds (under / optimal / over damping)
- Gap vs kappa with robustness window
- Targeting and potential-descent traces (P_success(t), E[V](t))
- Efficiency landscape P_success(T; gamma, kappa)
- Topology overlay (normalized gap curves)
- Energy sweep: throughput and E_tax/E_comm vs tau_reset (reset bottleneck)
- Compact dashboard combining the three knobs

Dependencies: matplotlib, numpy
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_cloud(eigen_clouds: Mapping[float, np.ndarray], save_path: str) -> None:
    """Plot complex Liouvillian eigenvalues for three representative kappas."""
    kappas = sorted(eigen_clouds.keys())
    if len(kappas) != 3:
        raise ValueError("eigen_clouds must contain exactly 3 keys (under/opt/over)")

    titles = ["Under-damped", "Near-optimal", "Over-damped (Zeno-like)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, k, title in zip(axes, kappas, titles):
        evals = eigen_clouds[k]
        ax.scatter(np.real(evals), np.imag(evals), alpha=0.6, s=14)
        ax.set_title(f"{title}\n$\\kappa={k:.3g}$")
        ax.set_xlabel(r"Re($\lambda$)")
        if ax is axes[0]:
            ax.set_ylabel(r"Im($\lambda$)")
        ax.axvline(0.0, linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Highlight the gap as a shaded strip.
        re = np.abs(np.real(evals))
        re_sorted = np.sort(re)
        g = re_sorted[1] if re_sorted.size > 1 else 0.0
        ax.axvspan(-g, 0.0, alpha=0.12)
        ax.text(0.02, 0.04, f"gap $g\\approx{g:.3g}$", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_gap_curve(kappa: np.ndarray, gap: np.ndarray, width_dec: float, save_path: str) -> None:
    """Plot Liouvillian gap vs dephasing rate with robustness window summary."""
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(kappa, gap, linewidth=2)
    ax.set_xlabel(r"Dephasing rate $\kappa$")
    ax.set_ylabel(r"Liouvillian gap $g$")
    ax.set_title(
        f"ENAQT-like optimum and robustness window ($W_\\kappa\\approx{width_dec:.2f}$ decades)"
    )
    ax.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_targeting_traces(
    t: np.ndarray,
    traces: Mapping[float, Mapping[str, np.ndarray]],
    save_path: str,
    subtitle: Optional[str] = None,
) -> None:
    """Two-panel figure: P_success(t) and E[V](t) for selected kappas."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    for k, d in traces.items():
        axes[0].plot(t, d["P_success"], linewidth=2, label=fr"$\kappa={k:.3g}$")
        axes[1].plot(t, d["EV"], linewidth=2, label=fr"$\kappa={k:.3g}$")

    axes[0].set_xlabel("time (a.u.)")
    axes[0].set_ylabel(r"$P_{success}(t)=1-\mathrm{Tr}(\rho_{nodes}(t))$")
    axes[0].set_title("Targeting / capture")
    axes[0].grid(True, alpha=0.35)

    axes[1].set_xlabel("time (a.u.)")
    axes[1].set_ylabel(r"$\mathbb{E}[V](t)$")
    axes[1].set_title("Free-energy proxy descent")
    axes[1].grid(True, alpha=0.35)

    axes[0].legend(loc="lower right", frameon=True)
    axes[1].legend(loc="upper right", frameon=True)

    if subtitle:
        fig.suptitle(subtitle, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_landscape(
    gamma_grid: np.ndarray,
    kappa_grid: np.ndarray,
    P: np.ndarray,
    save_path: str,
    title: str = r"$P_{success}(T;\gamma,\kappa)$ landscape",
) -> None:
    """Heatmap of capture probability at a fixed horizon T across (gamma, kappa)."""
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        P.T,
        origin="lower",
        aspect="auto",
        extent=[gamma_grid.min(), gamma_grid.max(), kappa_grid.min(), kappa_grid.max()],
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(title)
    ax.set_yscale("log")
    fig.colorbar(im, ax=ax, label=r"$P_{success}(T)$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_topology_overlay(results: Mapping[str, Mapping[str, np.ndarray]], save_path: str) -> None:
    """Plot normalized gap curves for multiple graph topologies.

    results[name] must have keys 'kappa' and 'gap'.
    """
    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    for name, d in results.items():
        kappa = np.asarray(d["kappa"], dtype=float)
        gap = np.asarray(d["gap"], dtype=float)
        k_opt = float(kappa[int(np.argmax(gap))])
        g_max = float(np.max(gap))
        ax.semilogx(kappa / k_opt, gap / g_max, linewidth=2, label=name)

    ax.axvline(1.0, linestyle=":", linewidth=1)
    ax.set_xlabel(r"Normalized noise $\kappa/\kappa^*$")
    ax.set_ylabel(r"Normalized gap $g/g_{max}$")
    ax.set_title("Topological invariance (normalized)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_energy_sweep(energy_df, save_path: str) -> None:
    """R3/R12: Plot throughput and E_tax/E_comm vs reset time (dual axis)."""

    t_reset = np.asarray(energy_df["tau_reset"], dtype=float)
    f_fix = np.asarray(energy_df["f_fix"], dtype=float)
    tax_ratio = np.asarray(energy_df["tax_ratio"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.0, 5.8))

    ax1.loglog(t_reset, f_fix, linewidth=2, label=r"$f_{fix}$")
    ax1.set_xlabel(r"Reset time $\tau_{reset}$ [s]")
    ax1.set_ylabel(r"Fixation rate $f_{fix}$ [Hz]")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    # Manuscript convention: reserve ρ(t) for the GKSL state descriptor, and use
    # χ := E_tax / E_comm for the operational energetic checkpoint.
    ax2.loglog(t_reset, tax_ratio, linestyle="--", linewidth=2, label=r"$\chi = E_{tax}/E_{comm}$")
    ax2.set_ylabel(r"Energy checkpoint $\chi$")

    # Checkpoints and illustrative bands
    ax2.axhline(1e-1, linestyle=":", linewidth=1, alpha=0.6)
    ax2.axhspan(1e-4, 1e-2, alpha=0.08)
    ax1.axvspan(1e-2, 1e-1, alpha=0.12)

    ax2.text(float(t_reset.min()), 1.15e-1, "Red line ($\chi=0.1$)", fontsize=8)
    ax2.text(float(t_reset.min()), 4e-3, "Safe zone ($\chi<0.01$)", fontsize=8)
    ax1.text(1.2e-2, float(f_fix.min()) * 1.6, "Cortical band", fontsize=8)

    # Unified legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

    ax1.set_title("Reset bottleneck sweep: throughput vs energy checkpoint")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_combined_dashboard_v2(
    gap_df,
    energy_df,
    clouds_data: Mapping[float, np.ndarray],
    save_path: str,
) -> None:
    """Compact 1x3 dashboard (Transport knob / Reset knob / Spectrum)."""

    fig = plt.figure(figsize=(18.0, 5.4))
    gs = fig.add_gridspec(1, 3)

    # Panel A: gap vs kappa
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(gap_df["kappa"], gap_df["gap"], linewidth=2)
    ax1.set_title("A. Transport optimization\n(The noise knob)")
    ax1.set_xlabel(r"$\kappa$")
    ax1.set_ylabel(r"Gap $g$")
    ax1.grid(True, which="both", alpha=0.3)

    # Panel B: reset knob (throughput + tax)
    ax2 = fig.add_subplot(gs[0, 1])
    t_reset = np.asarray(energy_df["tau_reset"], dtype=float)
    f_fix = np.asarray(energy_df["f_fix"], dtype=float)
    tax_ratio = np.asarray(energy_df["tax_ratio"], dtype=float)

    ax2.loglog(t_reset, f_fix, linewidth=2, label=r"$f_{fix}$")
    ax2_2 = ax2.twinx()
    ax2_2.loglog(t_reset, tax_ratio, linestyle="--", linewidth=2, label=r"$E_{tax}/E_{comm}$")
    ax2.axvspan(1e-2, 1e-1, alpha=0.12, label="Cortical band")
    ax2_2.axhline(1e-1, linestyle=":", linewidth=1, alpha=0.6)

    ax2.set_title("B. Reset bottleneck\n(The reset knob)")
    ax2.set_xlabel(r"$\tau_{reset}$ [s]")
    ax2.set_ylabel(r"$f_{fix}$ [Hz]")
    ax2_2.set_ylabel(r"$E_{tax}/E_{comm}$")
    ax2.grid(True, which="both", alpha=0.3)

    # Merge legends
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

    # Panel C: spectral cloud at kappa* (use middle key if present)
    ax3 = fig.add_subplot(gs[0, 2])
    if clouds_data:
        k_sorted = sorted(clouds_data.keys())
        k_show = k_sorted[1] if len(k_sorted) >= 2 else k_sorted[0]
        evals = clouds_data[k_show]
        ax3.scatter(np.real(evals), np.imag(evals), s=10, alpha=0.7)
        ax3.set_title(f"C. Spectral anatomy\n($\\kappa\\approx{k_show:.3g}$)")
    else:
        ax3.set_title("C. Spectral anatomy")
    ax3.set_xlabel(r"Re($\lambda$)")
    ax3.set_ylabel(r"Im($\lambda$)")
    ax3.axvline(0.0, linewidth=1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
