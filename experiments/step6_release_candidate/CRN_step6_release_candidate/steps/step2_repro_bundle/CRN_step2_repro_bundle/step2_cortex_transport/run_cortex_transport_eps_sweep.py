#!/usr/bin/env python3
"""
Mouse cortex proxy (SBM) disorder (epsilon) sweep at fixed T_env.

This script is a small extension of Step2 `run_cortex_transport.py`:
it sweeps diagonal disorder amplitude epsilon while keeping the temperature
parameter T_env fixed (default T_env=0.1), and reports selectivity as
Recall/Leakage using a ratio-of-means estimator across trials.

Defaults match the numbers reported in Core Figure 2C / Supplementary Table S10a:
- N=150 (3 blocks of 50)
- p_in=0.30, p_out=0.02
- T_env=0.1
- eps_values = 0,1,2,3,4,5,8
- n_trials=100
- seed=1
- t_end=100
- bootstrap_B=5000 (for SelRoM CI)

Outputs:
- cortex_eps_sweep_trials.csv
- cortex_eps_sweep_summary.csv
- metadata.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class Config:
    N: int = 150
    cluster_sizes: Tuple[int, int, int] = (50, 50, 50)
    p_in: float = 0.30
    p_out: float = 0.02
    T_env: float = 0.1
    eps_values: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0)
    t_end: float = 100.0
    n_trials: int = 100
    seed: int = 1
    bootstrap_B: int = 5000


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


def bootstrap_ratio_of_means(recall: np.ndarray, leakage: np.ndarray, B: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = recall.size
    idx = rng.integers(0, n, size=(B, n))
    rec_means = recall[idx].mean(axis=1)
    leak_means = leakage[idx].mean(axis=1)
    ratios = rec_means / leak_means
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return float(lo), float(hi)


def run(cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    records = []
    t0 = time.time()

    for trial in range(cfg.n_trials):
        A, clusters = generate_sbm_adjacency(
            rng=rng, N=cfg.N, cluster_sizes=cfg.cluster_sizes, p_in=cfg.p_in, p_out=cfg.p_out
        )

        # Base disorder; scaled per epsilon so the same microstate is reused across eps-values
        E0 = rng.uniform(-1.0, 1.0, size=cfg.N)

        C0 = np.where(clusters == 0)[0]
        S = rng.choice(C0, size=int(0.2 * len(C0)), replace=False)
        Tidx = np.setdiff1d(C0, S)
        Didx = np.where(clusters != 0)[0]

        P0 = np.zeros(cfg.N, dtype=float)
        P0[S] = 1.0 / len(S)

        for eps in cfg.eps_values:
            E = E0 * float(eps)
            Q = metropolis_generator(A, E, T_env=float(cfg.T_env))
            P_end = expm_multiply(Q * cfg.t_end, P0)

            recall = float(P_end[Tidx].sum())
            leakage = float(P_end[Didx].sum())
            sel = recall / leakage if leakage > 0 else float("inf")

            records.append({
                "trial": trial,
                "epsilon": float(eps),
                "T_env": float(cfg.T_env),
                "recall": recall,
                "leakage": leakage,
                "selectivity": sel,
                **asdict(cfg),
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(out_dir, "cortex_eps_sweep_trials.csv"), index=False)

    # Summary: means + CIs + ratio-of-means selectivity
    summary_rows = []
    for eps, g in df.groupby("epsilon"):
        recall = g["recall"].to_numpy()
        leakage = g["leakage"].to_numpy()
        n = recall.size

        rec_mean = float(recall.mean())
        leak_mean = float(leakage.mean())
        rec_std = float(recall.std(ddof=1))
        leak_std = float(leakage.std(ddof=1))

        rec_ci95 = 1.96 * (rec_std / math.sqrt(n)) if n > 1 else 0.0
        leak_ci95 = 1.96 * (leak_std / math.sqrt(n)) if n > 1 else 0.0

        sel_rom = rec_mean / leak_mean if leak_mean > 0 else float("inf")
        sel_lo, sel_hi = bootstrap_ratio_of_means(recall, leakage, B=cfg.bootstrap_B, seed=123 + int(eps))

        summary_rows.append({
            "epsilon": float(eps),
            "n_trials": int(n),
            "recall_mean": rec_mean,
            "recall_ci95": float(rec_ci95),
            "leakage_mean": leak_mean,
            "leakage_ci95": float(leak_ci95),
            "SelRoM": float(sel_rom),
            "SelRoM_ci95_low": sel_lo,
            "SelRoM_ci95_high": sel_hi,
        })

    summary = pd.DataFrame(summary_rows).sort_values("epsilon")
    summary.to_csv(os.path.join(out_dir, "cortex_eps_sweep_summary.csv"), index=False)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - t0,
        "config": asdict(cfg),
        "notes": [
            "Linear continuous-time Markov transport proxy (not a nonlinear attractor memory model).",
            "Adjacency and base disorder E0 are resampled per trial; E0 is scaled per epsilon within each trial.",
            "SelRoM is the ratio-of-means estimator: mean(recall)/mean(leakage).",
        ],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote outputs to: {out_dir}")
    print(f"Runtime: {meta['runtime_sec']:.2f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./outputs_eps_sweep", help="Output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--t_end", type=float, default=100.0)
    p.add_argument("--p_in", type=float, default=0.30)
    p.add_argument("--p_out", type=float, default=0.02)
    p.add_argument("--T_env", type=float, default=0.1)
    p.add_argument("--eps", default="0,1,2,3,4,5,8", help="Comma-separated epsilon values")
    p.add_argument("--bootstrap_B", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eps_values = tuple(float(x.strip()) for x in args.eps.split(",") if x.strip())
    cfg = Config(
        seed=args.seed,
        n_trials=args.trials,
        t_end=args.t_end,
        p_in=args.p_in,
        p_out=args.p_out,
        T_env=args.T_env,
        eps_values=eps_values,
        bootstrap_B=args.bootstrap_B,
    )
    run(cfg, out_dir=args.out)
