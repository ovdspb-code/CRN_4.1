#!/usr/bin/env python3
"""
SBM cortex transport toy experiment (Step 2)

What it does
------------
1) Builds a stochastic block model (SBM) graph that imitates 3 recurrent cortical
   assemblies (3 clusters of size 50; N=150 by default):
     - within-cluster edge probability p_in
     - cross-cluster edge probability p_out

2) Assigns a node-local energy/disorder landscape:
     E_i ~ Uniform[-epsilon, +epsilon]

3) Runs a continuous-time Markov master equation dP/dt = Q P starting from a
   partial cue (20% of C0 activated uniformly), and evaluates:

   Recall    = sum_{i in Target}    P_i(t_end)
   Leakage   = sum_{i in Distractors} P_i(t_end)
   Selectivity (reported) = Recall/Leakage (ratio of means across trials)

Regimes
-------
A) "Classical low-T":   Metropolis-like rates with T_env = 0.1
B) "Classical T=1":     Metropolis-like rates with T_env = 1.0
C) "Max-permeability":  T_env -> inf (energies ignored; effectively topological diffusion)

Important caveats (reviewer-facing)
-----------------------------------
- This is a *linear transport proxy*, not a full associative-memory model.
  Pattern completion in cortex/hippocampus is typically nonlinear (recurrent
  excitation + inhibition + thresholds). Here we only test a minimal question:
  how fast does mass leak through sparse inter-assembly edges under different
  transport permeability assumptions?

- Treating "CRN" as T_env -> inf is an intentionally extreme permeability limit.
  It matches the earlier simplified intuition ("barriers turned off"), but it is
  NOT equivalent to the full CRN/GKSL wave-layer model used elsewhere.

Outputs
-------
Writes trial-level CSV + summary CSV + two PNG plots + metadata.json into ./outputs/
by default.

Reproducibility
---------------
All randomness is driven by a single base seed; using the same code + seed
regenerates identical results.

"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    N: int = 150
    cluster_sizes: Tuple[int, int, int] = (50, 50, 50)
    p_in: float = 0.30
    p_out: float = 0.02
    epsilon: float = 3.0
    t_end: float = 100.0
    n_trials: int = 100
    seed: int = 1
    Ts: Tuple[float, float, float] = (0.1, 1.0, math.inf)


def generate_sbm_adjacency(
    rng: np.random.Generator,
    N: int,
    cluster_sizes: Tuple[int, ...],
    p_in: float,
    p_out: float,
) -> tuple[csr_matrix, np.ndarray]:
    clusters = np.concatenate([
        np.full(sz, cid, dtype=int) for cid, sz in enumerate(cluster_sizes)
    ])
    if clusters.size != N:
        raise ValueError(f"cluster_sizes sum to {clusters.size}, but N={N}")

    rows = []
    cols = []
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


def run(cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    records = []
    t0 = time.time()

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

        for T_env in cfg.Ts:
            Q = metropolis_generator(A, E, T_env=T_env)
            P_end = expm_multiply(Q * cfg.t_end, P0)

            recall = float(P_end[Tidx].sum())
            leakage = float(P_end[Didx].sum())
            sel = recall / leakage if leakage > 0 else float("inf")

            records.append({
                "trial": trial,
                "T_env": ("inf" if math.isinf(T_env) else float(T_env)),
                "recall": recall,
                "leakage": leakage,
                "selectivity": sel,
                **asdict(cfg),
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(out_dir, "cortex_transport_trials.csv"), index=False)

    # Summary table: mean ± 95% CI, and selectivity as ratio-of-means
    df2 = df.copy()
    df2["T_env_num"] = df2["T_env"].replace("inf", np.inf).astype(float)
    summary_rows = []
    for T_env, g in df2.groupby("T_env_num"):
        n = len(g)
        rec_mean = g["recall"].mean()
        leak_mean = g["leakage"].mean()
        rec_std = g["recall"].std(ddof=1)
        leak_std = g["leakage"].std(ddof=1)
        rec_ci95 = 1.96 * (rec_std / math.sqrt(n)) if n > 1 else 0.0
        leak_ci95 = 1.96 * (leak_std / math.sqrt(n)) if n > 1 else 0.0
        sel_ratio = rec_mean / leak_mean if leak_mean > 0 else float("inf")
        summary_rows.append({
            "T_env": ("inf" if math.isinf(T_env) else float(T_env)),
            "n": n,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "recall_ci95": rec_ci95,
            "leakage_mean": leak_mean,
            "leakage_std": leak_std,
            "leakage_ci95": leak_ci95,
            "selectivity_ratio_of_means": sel_ratio,
        })
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(by="T_env", key=lambda s: s.replace("inf", np.inf))
    summary.to_csv(os.path.join(out_dir, "cortex_transport_summary.csv"), index=False)

    # Plots
    labels = [str(x) for x in summary["T_env"]]
    x = np.arange(len(labels))

    # Recall vs Leakage
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, summary["recall_mean"], width, yerr=summary["recall_ci95"], capsize=4, label="Recall (target mass)")
    ax.bar(x + width/2, summary["leakage_mean"], width, yerr=summary["leakage_ci95"], capsize=4, label="Leakage (distractors mass)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("T_env")
    ax.set_ylabel("Probability mass at t_end")
    ax.set_title("SBM cortex transport: Recall vs Leakage (mean ± 95% CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cortex_transport_recall_leakage.png"), dpi=200)
    plt.close(fig)

    # Selectivity
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, summary["selectivity_ratio_of_means"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("T_env")
    ax.set_ylabel("Recall/Leakage (ratio of means)")
    ax.set_title("SBM cortex transport: Selectivity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cortex_transport_selectivity.png"), dpi=200)
    plt.close(fig)

    # Metadata
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - t0,
        "config": asdict(cfg),
        "notes": [
            "Linear continuous-time Markov transport proxy.",
            "CRN is approximated here by the maximal-permeability limit T_env -> inf (energies ignored).",
        ],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote outputs to: {out_dir}")
    print(f"Runtime: {meta['runtime_sec']:.2f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"), help="Output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--t_end", type=float, default=100.0)
    p.add_argument("--p_in", type=float, default=0.30)
    p.add_argument("--p_out", type=float, default=0.02)
    p.add_argument("--epsilon", type=float, default=3.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        n_trials=args.trials,
        t_end=args.t_end,
        p_in=args.p_in,
        p_out=args.p_out,
        epsilon=args.epsilon,
    )
    run(cfg, out_dir=args.out)
