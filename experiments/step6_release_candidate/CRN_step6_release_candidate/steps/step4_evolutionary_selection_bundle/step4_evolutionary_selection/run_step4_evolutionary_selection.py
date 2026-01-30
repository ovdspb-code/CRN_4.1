#!/usr/bin/env python3
"""
Step 4: Evolutionary selection pressure for CRN-like modular architectures.

Goal (operational)
------------------
Given two recurring environmental demands:

(1) Transport / selection on a disordered graph, where CRN/GKSL exhibits an intermediate-noise
    (ENAQT-like / NAT-like) optimum (proxied here by the normalized Liouvillian gap g(kappa)).

(2) Recurrent associative memory / confinement, where high permeability increases leakage
    (proxied here by a bounded "memory selectivity" metric: Recall / (Recall + Leakage)).

Test whether a *modular* architecture (separate local control of the noise/permeability regime)
has higher expected fitness than any *single global knob* architecture, once a plausible
modularity cost is included (extra control circuitry, gating, regulation).

Inputs
------
- inputs/combined_tradeoff.csv:
    columns: noise, transport_norm, memory_frac  (generated in Step 3)
- inputs/transport_energy_sweep.csv (optional; used only to annotate chi/red-line examples)

Outputs (./outputs/)
--------------------
- strategy_catalog.csv
- fitness_landscape.csv
- best_strategy_grid.csv
- phase_diagram_modular_vs_global.png
- best_fitness_vs_ptransport.png
- replicator_dynamics_example.csv
- replicator_dynamics_example.png
- metadata_step4.json
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class Step4Config:
    # Environment mixture: p_transport is swept on [0, 1]
    p_transport_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.0, 1.0, 21), 3))

    # Modularity cost: subtract from modular strategies only (swept)
    modular_cost_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.0, 0.20, 41), 3))

    # Replicator-mutation demo (one representative point)
    demo_p_transport: float = 0.50
    demo_modular_cost: float = 0.05
    demo_generations: int = 200
    demo_beta: float = 6.0  # selection intensity for softmax-like update
    demo_mu: float = 0.01   # mutation rate

    # Output cosmetics
    dpi: int = 200


# -----------------------------
# Helpers
# -----------------------------

def load_tradeoff_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"noise", "transport_norm", "memory_frac"}
    if not required.issubset(df.columns):
        raise ValueError(f"tradeoff table must contain columns: {sorted(required)}")
    df = df.copy()
    df["noise"] = df["noise"].astype(float)
    df["transport_norm"] = df["transport_norm"].astype(float)
    df["memory_frac"] = df["memory_frac"].astype(float)
    return df.sort_values("noise").reset_index(drop=True)


def build_strategy_catalog(noise_grid: List[float]) -> pd.DataFrame:
    """
    Enumerate strategies:

    - Global strategies: one knob n shared by both modules.
      label: G(n)

    - Modular strategies: independent knobs (n_T, n_M)
      label: M(n_T,n_M)
    """
    records: List[Dict[str, object]] = []
    # Global
    for n in noise_grid:
        records.append({
            "strategy": f"G({n:g})",
            "class": "global",
            "noise_transport": float(n),
            "noise_memory": float(n),
        })
    # Modular
    for nt in noise_grid:
        for nm in noise_grid:
            records.append({
                "strategy": f"M({nt:g},{nm:g})",
                "class": "modular",
                "noise_transport": float(nt),
                "noise_memory": float(nm),
            })
    return pd.DataFrame.from_records(records)


def build_lookup(df_trade: pd.DataFrame) -> Dict[float, Dict[str, float]]:
    """Map noise -> metrics dict (transport_norm, memory_frac)."""
    out: Dict[float, Dict[str, float]] = {}
    for _, r in df_trade.iterrows():
        n = float(r["noise"])
        out[n] = {
            "transport_norm": float(r["transport_norm"]),
            "memory_frac": float(r["memory_frac"]),
        }
    return out


def fitness_for_strategy(
    row: pd.Series,
    lookup: Dict[float, Dict[str, float]],
    p_transport: float,
    modular_cost: float,
) -> float:
    nt = float(row["noise_transport"])
    nm = float(row["noise_memory"])
    t = lookup[nt]["transport_norm"]
    m = lookup[nm]["memory_frac"]
    base = p_transport * t + (1.0 - p_transport) * m
    if row["class"] == "modular":
        base -= modular_cost
    return float(base)


def softmax_replicator(
    fitness: np.ndarray,
    generations: int,
    beta: float,
    mu: float,
) -> np.ndarray:
    """
    Deterministic 'replicator-mutation' style dynamic with softmax-like selection:

      p_{t+1} ∝ p_t * exp(beta * fitness)

    then apply uniform mutation with rate mu.
    """
    n = fitness.size
    p = np.full((n,), 1.0 / n, dtype=float)
    hist = np.zeros((generations + 1, n), dtype=float)
    hist[0] = p

    # stabilize exp by subtracting max
    f = fitness - float(np.max(fitness))

    for t in range(1, generations + 1):
        w = p * np.exp(beta * f)
        if float(w.sum()) <= 0:
            w = np.full_like(p, 1.0 / n)
        else:
            w = w / float(w.sum())
        # mutation
        p = (1.0 - mu) * w + mu * (1.0 / n)
        hist[t] = p

    return hist


# -----------------------------
# Main driver
# -----------------------------

def main() -> None:
    cfg = Step4Config()

    root = os.path.dirname(__file__)
    in_dir = os.path.join(root, "inputs")
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    df_trade = load_tradeoff_table(os.path.join(in_dir, "combined_tradeoff.csv"))
    noise_grid = [float(x) for x in df_trade["noise"].tolist()]
    lookup = build_lookup(df_trade)

    # 1) Strategy catalog
    df_strat = build_strategy_catalog(noise_grid)
    df_strat.to_csv(os.path.join(out_dir, "strategy_catalog.csv"), index=False)

    # 2) Fitness landscape over (p_transport, modular_cost)
    rows = []
    best_rows = []
    for pT in cfg.p_transport_grid:
        for cmod in cfg.modular_cost_grid:
            # compute fitness for all strategies
            fit = df_strat.apply(lambda r: fitness_for_strategy(r, lookup, pT, cmod), axis=1)
            df_tmp = df_strat.copy()
            df_tmp["p_transport"] = float(pT)
            df_tmp["modular_cost"] = float(cmod)
            df_tmp["fitness"] = fit.values
            rows.append(df_tmp)

            # best by class
            g_best = df_tmp[df_tmp["class"] == "global"].sort_values("fitness", ascending=False).iloc[0]
            m_best = df_tmp[df_tmp["class"] == "modular"].sort_values("fitness", ascending=False).iloc[0]

            winner = "modular" if float(m_best["fitness"]) > float(g_best["fitness"]) else "global"
            if math.isclose(float(m_best["fitness"]), float(g_best["fitness"]), rel_tol=1e-12, abs_tol=1e-12):
                winner = "tie"

            best_rows.append({
                "p_transport": float(pT),
                "modular_cost": float(cmod),
                "best_global_strategy": str(g_best["strategy"]),
                "best_global_fitness": float(g_best["fitness"]),
                "best_modular_strategy": str(m_best["strategy"]),
                "best_modular_fitness": float(m_best["fitness"]),
                "winner": winner,
                "modular_gain": (float(m_best["fitness"]) / float(g_best["fitness"])) if float(g_best["fitness"]) > 0 else float("nan"),
            })

    df_land = pd.concat(rows, ignore_index=True)
    df_land.to_csv(os.path.join(out_dir, "fitness_landscape.csv"), index=False)

    df_best = pd.DataFrame.from_records(best_rows)
    df_best.to_csv(os.path.join(out_dir, "best_strategy_grid.csv"), index=False)

    # 3) Phase diagram plot: modular wins region
    # We'll plot for modular_cost (y) vs p_transport (x)
    pivot = df_best.pivot(index="modular_cost", columns="p_transport", values="winner")
    # Encode: global=0, tie=0.5, modular=1
    encode = {"global": 0.0, "tie": 0.5, "modular": 1.0}
    Z = pivot.replace(encode).values.astype(float)
    x = pivot.columns.values.astype(float)
    y = pivot.index.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("p_transport (fraction of transport/selection demands)")
    ax.set_ylabel("Modularity cost (fitness penalty per act)")
    ax.set_title("Phase diagram: modular vs single-knob (winner)")
    # legend-like ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["global wins", "tie", "modular wins"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_diagram_modular_vs_global.png"), dpi=cfg.dpi)
    plt.close(fig)

    # 4) Best fitness vs p_transport for a few modular_cost slices
    slices = [0.0, 0.05, 0.10, 0.15]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for c in slices:
        df_slice = df_best[np.isclose(df_best["modular_cost"], c)].sort_values("p_transport")
        ax.plot(df_slice["p_transport"], df_slice["best_global_fitness"], linestyle="--", marker="o", label=f"global (c={c:g})")
        ax.plot(df_slice["p_transport"], df_slice["best_modular_fitness"], linestyle="-", marker="s", label=f"modular (c={c:g})")
    ax.set_xlabel("p_transport")
    ax.set_ylabel("Best achievable expected fitness")
    ax.set_title("Best fitness vs environment mixture (selected cost slices)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_fitness_vs_ptransport.png"), dpi=cfg.dpi)
    plt.close(fig)

    # 5) Replicator-mutation demo at one point
    pT0 = cfg.demo_p_transport
    c0 = cfg.demo_modular_cost

    df_demo = df_strat.copy()
    df_demo["fitness"] = df_demo.apply(lambda r: fitness_for_strategy(r, lookup, pT0, c0), axis=1)

    fitness = df_demo["fitness"].values.astype(float)
    hist = softmax_replicator(
        fitness=fitness,
        generations=cfg.demo_generations,
        beta=cfg.demo_beta,
        mu=cfg.demo_mu,
    )

    # Aggregate by class
    is_mod = (df_demo["class"].values == "modular")
    mod_mass = hist[:, is_mod].sum(axis=1)
    glob_mass = hist[:, ~is_mod].sum(axis=1)

    # Also track top-3 strategies by equilibrium mass
    eq = hist[-1]
    top_idx = np.argsort(eq)[::-1][:3]
    top_labels = df_demo.iloc[top_idx]["strategy"].tolist()

    # Save demo CSV
    demo_rows = []
    for t in range(hist.shape[0]):
        row = {
            "generation": int(t),
            "p_transport": float(pT0),
            "modular_cost": float(c0),
            "mass_modular": float(mod_mass[t]),
            "mass_global": float(glob_mass[t]),
        }
        for k, idx in enumerate(top_idx):
            row[f"top{k+1}_strategy"] = str(df_demo.iloc[idx]["strategy"])
            row[f"top{k+1}_mass"] = float(hist[t, idx])
        demo_rows.append(row)
    df_demo_out = pd.DataFrame.from_records(demo_rows)
    df_demo_out.to_csv(os.path.join(out_dir, "replicator_dynamics_example.csv"), index=False)

    # Plot demo
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(df_demo_out["generation"], df_demo_out["mass_modular"], label="sum(modular strategies)")
    ax.plot(df_demo_out["generation"], df_demo_out["mass_global"], label="sum(global strategies)")
    # overlay top 3
    for k in range(3):
        ax.plot(df_demo_out["generation"], df_demo_out[f"top{k+1}_mass"], linestyle="--", label=f"top{k+1}: {top_labels[k]}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population mass")
    ax.set_title(f"Replicator-mutation demo (p_transport={pT0:g}, modular_cost={c0:g})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "replicator_dynamics_example.png"), dpi=cfg.dpi)
    plt.close(fig)

    # 6) Metadata
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "noise_grid": noise_grid,
        "notes": [
            "This step is an architecture-level evolutionary pressure model, not a claim about a specific substrate.",
            "Transport performance proxy is normalized Liouvillian gap from Step 3 (R12-CROWN gap sweep).",
            "Memory performance proxy is bounded recall fraction from the Step 3 recurrent-microcircuit sweep.",
            "Modularity cost represents extra regulation/gating complexity; selection favors modularity when both tasks matter and cost is not too high."
        ],
        "demo_top3_strategies_at_equilibrium": top_labels,
        "demo_equilibrium_modular_mass": float(mod_mass[-1]),
        "demo_equilibrium_global_mass": float(glob_mass[-1]),
    }
    with open(os.path.join(out_dir, "metadata_step4.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs written to:", out_dir)
    print("Demo equilibrium modular mass:", meta["demo_equilibrium_modular_mass"])
    print("Demo top-3 strategies:", top_labels)


if __name__ == "__main__":
    main()
