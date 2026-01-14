"""Run the QRN Simulation Suite (Release 4.0).

Portable (numpy/scipy/matplotlib) and Colab-friendly.
Writes figures and CSV tables to ./outputs.

Usage:
  python run_all.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qrn_analysis import QRNAnalyzer
from qrn_energy import EnergyCalculator, EnergyConfig
from qrn_viz import (
    plot_efficiency_landscape,
    plot_gap_curve,
    plot_energy_sweep,
    plot_combined_dashboard_v2,
    plot_spectral_cloud,
    plot_targeting_traces,
    plot_topology_overlay,
)


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("[1/5] Gap sweep + spectral cloud")

    # Demo configuration (easy to tweak)
    N = 8
    topology = "chain"
    gamma = 2.0
    eta = 0.8
    T = 15.0  # horizon (a.u.)

    # Illustrative diagonal potential: bias toward target node.
    V = np.linspace(2.0, 0.0, N)

    # 1) Noise knob: spectral gap sweep + robustness window
    # Use a moderate resolution to keep the suite fast enough for Colab / CI.
    kappa_list = np.logspace(-2, 2, 11)
    sweep, clouds = QRNAnalyzer.run_kappa_sweep(N=N, gamma=gamma, kappa_list=kappa_list, topology=topology, eta=eta)
    width_dec = QRNAnalyzer.calculate_window_width(sweep, threshold_ratio=0.5)
    plot_gap_curve(sweep.kappa, sweep.gap, width_dec, save_path=str(out_dir / "gap_vs_kappa.png"))

    kappa_opt = float(sweep.kappa[int(np.argmax(sweep.gap))])
    if clouds:
        plot_spectral_cloud(clouds, save_path=str(out_dir / "spectral_cloud.png"))

    print("[2/5] Targeting traces")

    # 2) Targeting traces
    kappa_trace = [0.0, kappa_opt, float(kappa_list[-1])]
    t, traces = QRNAnalyzer.simulate_targeting_traces(
        N=N,
        gamma=gamma,
        kappa_values=kappa_trace,
        topology=topology,
        eta=eta,
        t_max=T,
        n_t=300,
        potential=V,
    )
    plot_targeting_traces(
        t,
        traces,
        save_path=str(out_dir / "targeting_traces.png"),
        subtitle=f"N={N}, topology={topology}, gamma={gamma}, eta={eta}, horizon T={T} (a.u.)",
    )

    print("[3/5] Topology overlay")

    # 4) Topology overlay (now includes complex graphs: WS / modular)
    topo_results = {}
    for topo in ["chain", "watts_strogatz"]:
        sw, _ = QRNAnalyzer.run_kappa_sweep(N=N, gamma=gamma, kappa_list=kappa_list, topology=topo, eta=eta)
        topo_results[topo] = {"kappa": sw.kappa, "gap": sw.gap}
    plot_topology_overlay(topo_results, save_path=str(out_dir / "topology_overlay.png"))

    print("[4/5] Reset knob energy audit")

    # 4b) Reset knob: regime sweep (coarse-grained energy audit)
    # These values are illustrative and can be re-parameterized to match a
    # specific substrate track.
    energy_cfg = EnergyConfig(E_comm=1.0, P_tax=1.0, tau_fix=1e-3)
    energy = EnergyCalculator(energy_cfg)
    tau_trans = 1e-4
    tau_reset_list = np.logspace(-5, -1, 13)
    energy_df = energy.calculate_metrics(tau_trans=tau_trans, tau_reset_list=tau_reset_list)

    plot_energy_sweep(energy_df, save_path=str(out_dir / "energy_regime.png"))
    plot_combined_dashboard_v2(
        gap_df={"kappa": sweep.kappa, "gap": sweep.gap},
        energy_df=energy_df,
        clouds_data=clouds,
        save_path=str(out_dir / "final_dashboard.png"),
    )

    print("[5/5] Threshold sink robustness")

    # 5) Nonlinear sink robustness proxy (R5)
    nonlin_drop = QRNAnalyzer.nonlinearity_drop(
        N=N,
        gamma=gamma,
        kappa=kappa_opt,
        eta_base=eta,
        eta_high=5.0,
        I_th=0.5,
        T=T,
        topology=topology,
        potential=V,
        n_t=300,
    )

    # Use a representative cortical-scale reset time (≈10 ms) to populate the
    # single-number checkpoint in the validation card.
    idx_cort = int(np.argmin(np.abs(energy_df["tau_reset"].to_numpy() - 1e-2)))
    p_tax_ratio = float(energy_df.iloc[idx_cort]["tax_ratio"])
    report = QRNAnalyzer.generate_validation_report(width_dec=width_dec, p_tax_ratio=p_tax_ratio, nonlin_drop=nonlin_drop)

    # Write report as CSV
    report_path = out_dir / "validation_report.csv"
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    # Also write sweep as CSV
    sweep_path = out_dir / "gap_sweep.csv"
    with sweep_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "gap"])
        for k, g in zip(sweep.kappa, sweep.gap):
            w.writerow([float(k), float(g)])

    # Write energy sweep as CSV
    (out_dir / "energy_sweep.csv").write_text(energy_df.to_csv(index=False))

    print("=== QRN Simulation Suite complete ===")
    print(f"kappa_opt (by spectral gap): {kappa_opt:.4g}")
    print(f"Robustness width W_kappa (0.5 max): {width_dec:.3f} decades")
    print(f"Threshold sink drop at kappa_opt: {nonlin_drop:.3f} (relative)")
    print(f"Energy checkpoint at tau_reset≈10 ms: E_tax/E_comm = {p_tax_ratio:.3g} ({energy_df.iloc[idx_cort]['red_line_status']})")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
