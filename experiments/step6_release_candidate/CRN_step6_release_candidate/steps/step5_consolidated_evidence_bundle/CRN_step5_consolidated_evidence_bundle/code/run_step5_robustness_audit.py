#!/usr/bin/env python3
"""
CRN alternative protocol — Step 5: Robustness audit for the evolutionary-modularity claim.

What Step 5 does
----------------
Step 4 showed, using one specific choice of proxies, that a modular architecture (separate local control of
noise/permeability for transport vs recurrent memory) can dominate a single global-knob architecture under
selection, as long as the modularity/gating cost is not too high.

Step 5 tests whether that conclusion is stable under alternative *reasonable* proxy definitions computed
from the same numeric backends produced in Step 3.

Inputs (relative to this script)
--------------------------------
- ../step5_robustness_audit/inputs/combined_tradeoff.csv
- ../step5_robustness_audit/inputs/memory_summary.csv
(Optional comparison)
- ../step5_robustness_audit/inputs/best_strategy_grid_step4_default.csv

Outputs
-------
- ../step5_robustness_audit/outputs/robustness_consensus_grid.csv
- ../step5_robustness_audit/outputs/boundary_thresholds.csv
- ../step5_robustness_audit/outputs/boundary_compare_step4_vs_consensus.csv
- ../step5_robustness_audit/outputs/phase_diagram_consensus.png
- ../step5_robustness_audit/outputs/boundary_curves.png
- ../step5_robustness_audit/outputs/control_no_tradeoff.csv
- ../step5_robustness_audit/outputs/control_no_tradeoff.png
- ../step5_robustness_audit/outputs/key_numbers.csv
- ../step5_robustness_audit/outputs/metadata_step5.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Step5Config:
    p_transport_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.0, 1.0, 21), 3))
    modular_cost_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.0, 0.20, 41), 3))
    consensus_thresholds: Tuple[float, ...] = (1.0, 0.8, 0.5)
    dpi: int = 200


def minmax_norm(x: np.ndarray) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def best_fitness(
    noise_grid: List[float],
    lookup: Dict[float, Dict[str, float]],
    metricT: str,
    metricM: str,
    pT: float,
    modular_cost: float,
) -> Tuple[float, float]:
    best_g = -1e18
    best_m = -1e18
    # global knob
    for n in noise_grid:
        t = lookup[n][metricT]
        m = lookup[n][metricM]
        f = pT * t + (1.0 - pT) * m
        if f > best_g:
            best_g = f
    # modular knob
    for nt in noise_grid:
        t = lookup[nt][metricT]
        for nm in noise_grid:
            m = lookup[nm][metricM]
            f = pT * t + (1.0 - pT) * m - modular_cost
            if f > best_m:
                best_m = f
    return float(best_g), float(best_m)


def main() -> None:
    cfg = Step5Config()

    root = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(root, "..", "step5_robustness_audit", "inputs")
    out_dir = os.path.join(root, "..", "step5_robustness_audit", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    trade = pd.read_csv(os.path.join(in_dir, "combined_tradeoff.csv"))
    mem = pd.read_csv(os.path.join(in_dir, "memory_summary.csv"))

    # noise grid is taken from Step 3 combined table
    noise_grid = [float(x) for x in trade["noise"].tolist()]

    # build lookup table with proxy variants
    lookup: Dict[float, Dict[str, float]] = {}
    for _, r in trade.iterrows():
        n = float(r["noise"])
        gap = float(r["transport_gap"])
        lookup[n] = {
            "T_norm": float(r["transport_norm"]),
            "T_gap": gap,
            "T_loggap": float(np.log(gap)),
            "M_frac": float(r["memory_frac"]),
        }

    # memory additional proxies from memory_summary.csv (match on T_env)
    ratio_map = dict(zip(mem["T_env"], mem["selectivity_ratio_of_means"]))
    diff_map = dict(zip(mem["T_env"], mem["recall_minus_leakage_mean"]))

    for n in noise_grid:
        ratio = float(ratio_map[n])
        lookup[n]["M_ratio"] = ratio / (1.0 + ratio)  # bounded
        lookup[n]["M_diff_raw"] = float(diff_map[n])

    # normalize variants
    tgap = np.array([lookup[n]["T_gap"] for n in noise_grid], dtype=float)
    tlog = np.array([lookup[n]["T_loggap"] for n in noise_grid], dtype=float)
    mdiff = np.array([lookup[n]["M_diff_raw"] for n in noise_grid], dtype=float)

    tgap_n = minmax_norm(tgap)
    tlog_n = minmax_norm(tlog)
    mdiff_n = minmax_norm(mdiff)

    for i, n in enumerate(noise_grid):
        lookup[n]["T_gap_norm"] = float(tgap_n[i])
        lookup[n]["T_loggap_norm"] = float(tlog_n[i])
        lookup[n]["M_diff_norm"] = float(mdiff_n[i])

    transport_variants = ["T_norm", "T_gap_norm", "T_loggap_norm"]
    memory_variants = ["M_frac", "M_ratio", "M_diff_norm"]

    # consensus grid
    rows = []
    for pT in cfg.p_transport_grid:
        for c in cfg.modular_cost_grid:
            wins = 0
            total = 0
            for mt in transport_variants:
                for mm in memory_variants:
                    total += 1
                    g, m = best_fitness(noise_grid, lookup, mt, mm, float(pT), float(c))
                    if m > g + 1e-12:
                        wins += 1
            rows.append({
                "p_transport": float(pT),
                "modular_cost": float(c),
                "wins": int(wins),
                "total": int(total),
                "modular_win_rate": float(wins / total),
            })

    df_cons = pd.DataFrame.from_records(rows)
    df_cons.to_csv(os.path.join(out_dir, "robustness_consensus_grid.csv"), index=False)

    # boundary tables
    def boundary(thresh: float) -> pd.DataFrame:
        out = []
        for pT, grp in df_cons.groupby("p_transport"):
            ok = grp[grp["modular_win_rate"] >= thresh]
            out.append({
                "p_transport": float(pT),
                "threshold": float(thresh),
                "max_modular_cost": float(ok["modular_cost"].max()) if len(ok) > 0 else float("nan"),
            })
        return pd.DataFrame.from_records(out)

    df_bound = pd.concat([boundary(t) for t in cfg.consensus_thresholds], ignore_index=True)
    df_bound.to_csv(os.path.join(out_dir, "boundary_thresholds.csv"), index=False)

    # compare with Step-4 default if present
    compare_path = os.path.join(in_dir, "best_strategy_grid_step4_default.csv")
    if os.path.exists(compare_path):
        step4 = pd.read_csv(compare_path)
        step4_boundary = []
        for pT, grp in step4.groupby("p_transport"):
            ok = grp[grp["winner"] == "modular"]
            step4_boundary.append({
                "p_transport": float(pT),
                "max_cost_step4_default": float(ok["modular_cost"].max()) if len(ok) > 0 else float("nan"),
            })
        df_step4 = pd.DataFrame.from_records(step4_boundary)
        pivot = df_bound[df_bound["threshold"].isin([1.0, 0.8])].pivot(index="p_transport", columns="threshold", values="max_modular_cost").reset_index()
        pivot = pivot.rename(columns={1.0: "max_cost_consensus_all", 0.8: "max_cost_consensus_ge0.8"})
        df_cmp = df_step4.merge(pivot, on="p_transport", how="left")
        df_cmp.to_csv(os.path.join(out_dir, "boundary_compare_step4_vs_consensus.csv"), index=False)
    else:
        df_cmp = None

    # plots
    pivot_img = df_cons.pivot(index="modular_cost", columns="p_transport", values="modular_win_rate").sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(
        pivot_img.values,
        aspect="auto",
        origin="lower",
        extent=[float(pivot_img.columns.min()), float(pivot_img.columns.max()), float(pivot_img.index.min()), float(pivot_img.index.max())],
    )
    ax.set_xlabel("p_transport")
    ax.set_ylabel("modularity_cost")
    ax.set_title("Consensus modular-win rate across metric variants")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("modular_win_rate (0..1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_diagram_consensus.png"), dpi=cfg.dpi)
    plt.close(fig)

    if df_cmp is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(df_cmp["p_transport"], df_cmp["max_cost_step4_default"], label="Step4 default metrics")
        ax.plot(df_cmp["p_transport"], df_cmp["max_cost_consensus_ge0.8"], label="Consensus ≥0.8")
        ax.plot(df_cmp["p_transport"], df_cmp["max_cost_consensus_all"], label="Consensus =1.0")
        ax.set_xlabel("p_transport")
        ax.set_ylabel("max modularity cost where modular wins")
        ax.set_title("Modularity selection boundary (cost tolerance)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "boundary_curves.png"), dpi=cfg.dpi)
        plt.close(fig)

    # Control: no tradeoff (memory=transport) ⇒ modular never helps
    ctrl_rows = []
    for pT in cfg.p_transport_grid:
        for c in (0.0, 0.05, 0.10):
            # define M := T_norm, T := T_norm
            best_g = -1e18
            best_m = -1e18
            for n in noise_grid:
                t = lookup[n]["T_norm"]
                f = float(pT) * t + (1.0 - float(pT)) * t
                best_g = max(best_g, f)
            for nt in noise_grid:
                t = lookup[nt]["T_norm"]
                for nm in noise_grid:
                    m = lookup[nm]["T_norm"]
                    f = float(pT) * t + (1.0 - float(pT)) * m - float(c)
                    best_m = max(best_m, f)
            ctrl_rows.append({
                "p_transport": float(pT),
                "modular_cost": float(c),
                "best_global": float(best_g),
                "best_modular": float(best_m),
                "delta_modular_minus_global": float(best_m - best_g),
            })
    df_ctrl = pd.DataFrame.from_records(ctrl_rows)
    df_ctrl.to_csv(os.path.join(out_dir, "control_no_tradeoff.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in sorted(df_ctrl["modular_cost"].unique()):
        sub = df_ctrl[df_ctrl["modular_cost"] == c]
        ax.plot(sub["p_transport"], sub["delta_modular_minus_global"], label=f"cost={c:g}")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("p_transport")
    ax.set_ylabel("Δ fitness (modular - global)")
    ax.set_title("Control: no tradeoff ⇒ modular never helps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "control_no_tradeoff.png"), dpi=cfg.dpi)
    plt.close(fig)

    # key numbers for paper / SI
    # transport optimum and memory optimum are single-grid summaries
    opt_transport = trade.loc[trade["transport_norm"].idxmax()]
    opt_memory = trade.loc[trade["memory_frac"].idxmax()]
    key = [
        {"block": "Step3_tradeoff", "condition": "transport", "metric": "noise_at_max_transport_norm", "value": float(opt_transport["noise"])},
        {"block": "Step3_tradeoff", "condition": "transport", "metric": "max_transport_norm", "value": float(opt_transport["transport_norm"])},
        {"block": "Step3_tradeoff", "condition": "memory", "metric": "noise_at_max_memory_frac", "value": float(opt_memory["noise"])},
        {"block": "Step3_tradeoff", "condition": "memory", "metric": "max_memory_frac", "value": float(opt_memory["memory_frac"])},
    ]
    if df_cmp is not None:
        row05 = df_cmp[df_cmp["p_transport"] == 0.5].iloc[0]
        key += [
            {"block": "Step4_selection", "condition": "p_transport=0.5", "metric": "max_cost_step4_default", "value": float(row05["max_cost_step4_default"])},
            {"block": "Step5_robustness", "condition": "p_transport=0.5", "metric": "max_cost_consensus_ge0.8", "value": float(row05["max_cost_consensus_ge0.8"])},
            {"block": "Step5_robustness", "condition": "p_transport=0.5", "metric": "max_cost_consensus_all", "value": float(row05["max_cost_consensus_all"])},
        ]
    pd.DataFrame.from_records(key).to_csv(os.path.join(out_dir, "key_numbers.csv"), index=False)

    # metadata
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "noise_grid": noise_grid,
        "transport_variants": transport_variants,
        "memory_variants": memory_variants,
        "notes": [
            "Step 5 audits Step-4 selection robustness to proxy choices (9 metric pairs).",
            "It does not claim any microscopic substrate; it is an architecture-level sensitivity check.",
        ],
    }
    with open(os.path.join(out_dir, "metadata_step5.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Step-5 robustness audit complete. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
