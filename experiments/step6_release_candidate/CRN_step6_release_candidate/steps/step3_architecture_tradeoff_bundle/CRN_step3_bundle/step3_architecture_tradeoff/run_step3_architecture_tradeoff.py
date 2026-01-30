#!/usr/bin/env python3
"""
Step 3: Architecture-level trade-off demo (transport vs recurrent memory).

This script:
1) Loads transport metric g(kappa) from a provided CSV (from R12-CROWN outputs).
2) Runs the recurrent-microcircuit (SBM) transport proxy across a T_env grid.
3) Computes a bounded memory metric and a normalized transport metric on a shared grid.
4) Demonstrates that a modular (locally tuned) architecture can dominate any single global knob.

Outputs are written to ./outputs/.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt


# -----------------------------
# SBM cortex transport proxy
# -----------------------------

@dataclass(frozen=True)
class MemoryConfig:
    N: int = 150
    cluster_sizes: Tuple[int, int, int] = (50, 50, 50)
    p_in: float = 0.30
    p_out: float = 0.02
    epsilon: float = 3.0
    t_end: float = 100.0
    n_trials: int = 20
    seed: int = 1
    T_env_grid: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, math.inf)


def generate_sbm_adjacency(
    rng: np.random.Generator,
    N: int,
    cluster_sizes: Tuple[int, ...],
    p_in: float,
    p_out: float,
) -> tuple[csr_matrix, np.ndarray]:
    clusters = np.concatenate([np.full(sz, cid, dtype=int) for cid, sz in enumerate(cluster_sizes)])
    if clusters.size != N:
        raise ValueError("cluster_sizes do not sum to N")

    rows: List[int] = []
    cols: List[int] = []
    for i in range(N):
        ci = clusters[i]
        for j in range(i + 1, N):
            p = p_in if clusters[j] == ci else p_out
            if rng.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])

    data = np.ones(len(rows), dtype=float)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, clusters


def metropolis_generator(A: csr_matrix, E: np.ndarray, T_env: float) -> csr_matrix:
    """Continuous-time generator Q (columns sum to 0) for Metropolis-like hopping."""
    N = A.shape[0]
    A_coo = A.tocoo()
    rows = A_coo.row
    cols = A_coo.col

    dE = E[rows] - E[cols]
    if math.isinf(T_env):
        rates = np.ones_like(dE, dtype=float)
    else:
        rates = np.exp(-np.maximum(dE, 0.0) / T_env)

    Q = csr_matrix((rates, (rows, cols)), shape=(N, N))
    out_rate = np.array(Q.sum(axis=0)).ravel()
    Q = Q + csr_matrix((-out_rate, (np.arange(N), np.arange(N))), shape=(N, N))
    return Q


def run_memory_sweep(cfg: MemoryConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    records = []

    for trial in range(cfg.n_trials):
        A, clusters = generate_sbm_adjacency(
            rng=rng, N=cfg.N, cluster_sizes=cfg.cluster_sizes, p_in=cfg.p_in, p_out=cfg.p_out
        )
        E = rng.uniform(-cfg.epsilon, cfg.epsilon, size=cfg.N)

        C0 = np.where(clusters == 0)[0]
        S = rng.choice(C0, size=int(0.2 * len(C0)), replace=False)
        Tidx = np.setdiff1d(C0, S)
        Didx = np.where(clusters != 0)[0]

        P0 = np.zeros(cfg.N, dtype=float)
        P0[S] = 1.0 / len(S)

        for T_env in cfg.T_env_grid:
            Q = metropolis_generator(A, E, T_env=T_env)
            P_end = expm_multiply(Q * cfg.t_end, P0)

            recall = float(P_end[Tidx].sum())
            leakage = float(P_end[Didx].sum())
            sel = recall / leakage if leakage > 0 else float("inf")
            frac = recall / (recall + leakage) if (recall + leakage) > 0 else 0.0
            util = recall - leakage

            records.append({
                "trial": trial,
                "T_env": ("inf" if math.isinf(T_env) else float(T_env)),
                "recall": recall,
                "leakage": leakage,
                "selectivity": sel,
                "recall_frac": frac,
                "recall_minus_leakage": util,
                **asdict(cfg),
            })

    return pd.DataFrame.from_records(records)


def summarize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["T_env_num"] = df2["T_env"].replace("inf", np.inf).astype(float)

    rows = []
    for T_env, g in df2.groupby("T_env_num"):
        n = len(g)
        rec_mean = g["recall"].mean()
        leak_mean = g["leakage"].mean()
        rec_std = g["recall"].std(ddof=1)
        leak_std = g["leakage"].std(ddof=1)
        rec_ci = 1.96 * (rec_std / math.sqrt(n)) if n > 1 else 0.0
        leak_ci = 1.96 * (leak_std / math.sqrt(n)) if n > 1 else 0.0
        sel_ratio = rec_mean / leak_mean if leak_mean > 0 else float("inf")
        frac_mean = g["recall_frac"].mean()
        util_mean = g["recall_minus_leakage"].mean()

        rows.append({
            "T_env": ("inf" if math.isinf(T_env) else float(T_env)),
            "n": n,
            "recall_mean": rec_mean,
            "recall_ci95": rec_ci,
            "leakage_mean": leak_mean,
            "leakage_ci95": leak_ci,
            "selectivity_ratio_of_means": sel_ratio,
            "recall_frac_mean": frac_mean,
            "recall_minus_leakage_mean": util_mean,
        })

    out = pd.DataFrame(rows)
    # Sort with inf last
    out["T_sort"] = out["T_env"].replace("inf", np.inf).astype(float).replace(np.inf, 1e9)
    out = out.sort_values("T_sort").drop(columns=["T_sort"])
    return out


# -----------------------------
# Transport metric handling
# -----------------------------

def load_transport_gap(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"kappa", "gap"}.issubset(df.columns):
        raise ValueError("transport CSV must contain columns: kappa, gap")
    return df


def interp_gap_logspace(df_gap: pd.DataFrame, kappa_values: Sequence[float]) -> np.ndarray:
    k = df_gap["kappa"].values.astype(float)
    g = df_gap["gap"].values.astype(float)
    logk = np.log10(k)

    out = np.zeros((len(kappa_values),), dtype=float)
    for i, kv in enumerate(kappa_values):
        out[i] = float(np.interp(np.log10(kv), logk, g))
    return out


# -----------------------------
# Main driver
# -----------------------------

def main() -> None:
    root = os.path.dirname(__file__)
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # 1) Transport (CRN) curve
    transport_csv = os.path.join(out_dir, "transport_gap_sweep.csv")
    if not os.path.exists(transport_csv):
        raise FileNotFoundError(
            "Expected transport_gap_sweep.csv in outputs/. "
            "Copy it from R12-CROWN outputs (gap_sweep.csv) before running."
        )
    df_gap = load_transport_gap(transport_csv)
    max_gap = float(df_gap["gap"].max())

    # 2) Memory sweep
    mem_cfg = MemoryConfig()
    df_trials = run_memory_sweep(mem_cfg)
    df_trials.to_csv(os.path.join(out_dir, "memory_trials.csv"), index=False)
    df_summary = summarize_memory(df_trials)
    df_summary.to_csv(os.path.join(out_dir, "memory_summary.csv"), index=False)

    # 3) Combine on a shared grid (exclude inf for transport)
    T_vals = [t for t in mem_cfg.T_env_grid if not math.isinf(t)]
    gap_interp = interp_gap_logspace(df_gap, T_vals)

    mem_map = df_summary.copy()
    mem_map["T_env_num"] = mem_map["T_env"].replace("inf", np.inf).astype(float)
    mem_map = mem_map.set_index("T_env_num")

    rows = []
    for T_env, gap in zip(T_vals, gap_interp):
        frac = float(mem_map.loc[T_env, "recall_frac_mean"])
        rows.append({
            "noise": float(T_env),
            "transport_gap": float(gap),
            "transport_norm": float(gap / max_gap),
            "memory_frac": float(frac),
        })
    df_comb = pd.DataFrame(rows)
    df_comb.to_csv(os.path.join(out_dir, "combined_tradeoff.csv"), index=False)

    # 4) Fitness demo: global knob vs modular tuning
    wT = 0.5
    wM = 0.5
    df_comb["fitness_global"] = wT * df_comb["transport_norm"] + wM * df_comb["memory_frac"]
    best_global_idx = int(df_comb["fitness_global"].idxmax())
    best_global_noise = float(df_comb.loc[best_global_idx, "noise"])
    best_global_fit = float(df_comb.loc[best_global_idx, "fitness_global"])

    best_transport_idx = int(df_comb["transport_norm"].idxmax())
    best_memory_idx = int(df_comb["memory_frac"].idxmax())
    best_transport_noise = float(df_comb.loc[best_transport_idx, "noise"])
    best_memory_noise = float(df_comb.loc[best_memory_idx, "noise"])
    modular_fit = float(wT * df_comb["transport_norm"].max() + wM * df_comb["memory_frac"].max())

    # 5) Weight sweep sensitivity
    weight_rows = []
    for wT_s in np.linspace(0, 1, 11):
        wM_s = 1.0 - wT_s
        fit_global = wT_s * df_comb["transport_norm"] + wM_s * df_comb["memory_frac"]
        best = float(fit_global.max())
        best_noise = float(df_comb.loc[int(fit_global.idxmax()), "noise"])
        mod = float(wT_s * df_comb["transport_norm"].max() + wM_s * df_comb["memory_frac"].max())
        weight_rows.append({
            "w_transport": float(wT_s),
            "w_memory": float(wM_s),
            "best_global_fitness": best,
            "best_global_noise": best_noise,
            "modular_fitness": mod,
            "modular_gain": (mod / best) if best > 0 else float("nan"),
        })
    df_weights = pd.DataFrame(weight_rows)
    df_weights.to_csv(os.path.join(out_dir, "fitness_weight_sweep.csv"), index=False)

    # 6) Plots
    # Memory recall/leakage
    df_plot = df_summary.copy()
    df_plot["T_env_num"] = df_plot["T_env"].replace("inf", np.inf).astype(float)
    df_plot["T_plot"] = df_plot["T_env_num"].replace(np.inf, 100.0)  # for log scale
    df_plot = df_plot.sort_values("T_plot")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(df_plot["T_plot"], df_plot["recall_mean"], yerr=df_plot["recall_ci95"], fmt="o-", label="Recall (target mass)")
    ax.errorbar(df_plot["T_plot"], df_plot["leakage_mean"], yerr=df_plot["leakage_ci95"], fmt="s-", label="Leakage (distractors mass)")
    ax.set_xscale("log")
    ax.set_xlabel("T_env (log; inf plotted at 100)")
    ax.set_ylabel("Probability mass at t_end")
    ax.set_title("Memory module (SBM proxy): Recall vs Leakage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_recall_leakage_vs_Tenv.png"), dpi=200)
    plt.close(fig)

    # Memory bounded metric
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_plot["T_plot"], df_plot["recall_frac_mean"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("T_env (log; inf plotted at 100)")
    ax.set_ylabel("Recall / (Recall + Leakage)")
    ax.set_title("Memory module: bounded selectivity metric vs T_env")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_frac_vs_Tenv.png"), dpi=200)
    plt.close(fig)

    # Transport gap
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_gap["kappa"], df_gap["gap"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("kappa (dephasing)")
    ax.set_ylabel("Liouvillian gap g(kappa)")
    ax.set_title("Transport module (CRN GKSL): ENAQT-like optimum in gap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transport_gap_vs_kappa.png"), dpi=200)
    plt.close(fig)

    # Pareto scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df_comb["transport_norm"], df_comb["memory_frac"])
    for _, r in df_comb.iterrows():
        ax.annotate(f"{r['noise']}", (r["transport_norm"], r["memory_frac"]), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Transport performance (normalized gap)")
    ax.set_ylabel("Memory performance (recall/(recall+leakage))")
    ax.set_title("Trade-off under a single global noise/permeability knob")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_tradeoff_scatter.png"), dpi=200)
    plt.close(fig)

    # Fitness vs noise (equal weights)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_comb["noise"], df_comb["fitness_global"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Global noise/permeability knob (T_env ~ kappa)")
    ax.set_ylabel("Fitness = 0.5*transport + 0.5*memory")
    ax.set_title("Single-knob architecture: fitness vs noise")
    ax.axhline(modular_fit, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fitness_vs_noise_equal_weights.png"), dpi=200)
    plt.close(fig)

    # Modular gain vs weight
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_weights["w_transport"], df_weights["modular_gain"], marker="o")
    ax.set_xlabel("Weight on transport objective (w_T)")
    ax.set_ylabel("Modular / best-global fitness ratio")
    ax.set_title("Architecture benefit of separating noise regimes (sensitivity)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "modular_gain_vs_weight.png"), dpi=200)
    plt.close(fig)

    # Metadata
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - t0,
        "memory_config": asdict(mem_cfg),
        "fitness_demo": {
            "w_transport": wT,
            "w_memory": wM,
            "best_global_noise": best_global_noise,
            "best_global_fitness": best_global_fit,
            "best_transport_noise": best_transport_noise,
            "best_memory_noise": best_memory_noise,
            "modular_fitness": modular_fit,
            "modular_gain_over_best_global": modular_fit / best_global_fit if best_global_fit > 0 else None,
        },
        "notes": [
            "Memory module is a linear transport proxy (no nonlinear attractor dynamics).",
            "Transport module uses R12-CROWN gap_sweep (Liouvillian gap as proxy).",
            "Global-knob vs modular comparison is an illustration of architectural selection pressure.",
        ],
    }
    with open(os.path.join(out_dir, "metadata_step3.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs written to:", out_dir)
    print("Runtime (sec):", meta["runtime_sec"])
    print("Best global noise:", best_global_noise, "fitness:", best_global_fit)
    print("Modular fitness:", modular_fit, "gain:", meta["fitness_demo"]["modular_gain_over_best_global"])


if __name__ == "__main__":
    main()
