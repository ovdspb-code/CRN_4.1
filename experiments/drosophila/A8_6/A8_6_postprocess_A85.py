#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A8.6 — Postprocess A8.5 trialwise outputs into publication-ready metrics + plots.

Inputs
------
One or more CSV files produced by A8.5:
  A8_5_arch_trialwise.csv

This script will merge inputs (useful when you run separate kappas in separate folders).

Outputs
-------
(out_dir)
  - A8_6_cell_metrics.csv
      per (variant, epsilon, kappa): ratio-of-means, mean-of-ratios, quantiles, good-run probability, etc.
  - A8_6_replicate_metrics.csv
      per replicate: original -> per trial; rewired/lesion -> per surrogate (means across trials)
  - A8_6_pairwise_tests_epsilon3.csv
      pairwise p-values at epsilon=3 (good-run Fisher exact; Selectivity_end Mann-Whitney on log)
  - PNG plots (if matplotlib is available)
      A8_6_Selectivity_ratioOfMeans_vs_epsilon_kappa*.png
      A8_6_goodrun_prob_vs_epsilon_kappa*.png
      A8_6_coverage_end_vs_epsilon_kappa*.png

Notes
-----
- "Selectivity_end_mean" in A8.5 summary is mean-of-ratios and can be heavy-tailed.
  For main figures we recommend ratio-of-means: mean(P_T)/mean(P_D).
- Good-run is defined as:
      good = (Selectivity_end > sel_thresh) AND (P_sink_T_end > pT_min)

Dependencies: Python 3, numpy. Optional: scipy (for stats), matplotlib (for plots).
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_inputs(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        z = 1.959963984540054  # ~N(0,1) 97.5%

    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


@dataclass
class RunRow:
    variant: str
    surrogate_id: int
    epsilon: float
    trial: int
    energy_seed: int
    kappa: float
    pT: float
    pD: float
    sel: float
    cov: float


def read_trialwise_csv(path: str) -> List[RunRow]:
    rows: List[RunRow] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"variant","surrogate_id","epsilon","trial","energy_seed","kappa","P_sink_T_end","P_sink_D_end","Selectivity_end","coverage_end"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"[A8.6] CSV missing columns {sorted(missing)}: {path}")
        for d in r:
            rows.append(RunRow(
                variant=str(d["variant"]),
                surrogate_id=int(float(d["surrogate_id"])),
                epsilon=_safe_float(d["epsilon"]),
                trial=int(float(d["trial"])),
                energy_seed=int(float(d["energy_seed"])),
                kappa=_safe_float(d["kappa"]),
                pT=_safe_float(d["P_sink_T_end"]),
                pD=_safe_float(d["P_sink_D_end"]),
                sel=_safe_float(d["Selectivity_end"]),
                cov=_safe_float(d["coverage_end"]),
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="Comma-separated list of A8_5_arch_trialwise.csv paths. "
                         "Example: 'A8_5_OUT/A8_5_arch_trialwise.csv,A8_5_OUT_k1/A8_5_arch_trialwise.csv'")
    ap.add_argument("--out_dir", required=True, help="Output directory for A8.6")
    ap.add_argument("--pT_min", type=float, default=0.005, help="Threshold on P_sink_T_end for good-run")
    ap.add_argument("--sel_thresh", type=float, default=2.0, help="Threshold on Selectivity_end for good-run")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for confidence intervals")
    ap.add_argument("--no_plots", action="store_true", help="Skip generating PNG plots")
    args = ap.parse_args()

    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(args.out_dir)))
    paths = _parse_inputs(args.inputs)
    if not paths:
        raise SystemExit("[A8.6] No inputs provided.")
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"[A8.6] Input not found: {p}")

    # Read + merge
    all_rows: List[RunRow] = []
    for p in paths:
        all_rows.extend(read_trialwise_csv(p))

    # Deduplicate by unique key (variant, surrogate_id, epsilon, trial, kappa)
    uniq: Dict[Tuple[str,int,float,int,float], RunRow] = {}
    for rr in all_rows:
        key = (rr.variant, rr.surrogate_id, float(rr.epsilon), rr.trial, float(rr.kappa))
        uniq[key] = rr
    rows = list(uniq.values())
    rows.sort(key=lambda x: (x.variant, x.surrogate_id, x.kappa, x.epsilon, x.trial))

    # Build runwise buckets by cell (variant, epsilon, kappa)
    cell_runs: Dict[Tuple[str,float,float], List[RunRow]] = defaultdict(list)
    for rr in rows:
        cell_runs[(rr.variant, float(rr.epsilon), float(rr.kappa))].append(rr)

    # Compute per-cell metrics (pooled over runs)
    cell_metrics = []
    for (variant, eps, kappa), rs in sorted(cell_runs.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        pT = np.array([r.pT for r in rs], dtype=float)
        pD = np.array([r.pD for r in rs], dtype=float)
        sel = np.array([r.sel for r in rs], dtype=float)
        cov = np.array([r.cov for r in rs], dtype=float)

        # pooled ratio-of-means
        pT_mean = float(np.mean(pT))
        pD_mean = float(np.mean(pD))
        sel_ratio_of_means = float(pT_mean / (pD_mean + 1e-12))

        # mean-of-ratios (already computed per run)
        sel_mean = float(np.mean(sel))
        sel_median = float(np.median(sel))
        q25, q75 = [float(x) for x in np.quantile(sel, [0.25, 0.75])]
        q10, q90 = [float(x) for x in np.quantile(sel, [0.10, 0.90])]

        cov_mean = float(np.mean(cov))

        good = (sel > float(args.sel_thresh)) & (pT > float(args.pT_min))
        k_good = int(np.sum(good))
        n = int(len(rs))
        p_good = float(k_good / n) if n > 0 else float("nan")
        ci_lo, ci_hi = _wilson_ci(k_good, n, alpha=float(args.alpha))

        cell_metrics.append([
            variant, eps, kappa, n,
            pT_mean, pD_mean, cov_mean,
            sel_ratio_of_means,
            sel_mean, sel_median, q25, q75, q10, q90,
            p_good, ci_lo, ci_hi,
        ])

    cell_path = os.path.join(out_dir, "A8_6_cell_metrics.csv")
    with open(cell_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant","epsilon","kappa","n_runs",
            "P_sink_T_end_mean","P_sink_D_end_mean","coverage_end_mean",
            "Selectivity_ratioOfMeans",
            "Selectivity_meanOfRatios","Selectivity_median","Selectivity_q25","Selectivity_q75","Selectivity_q10","Selectivity_q90",
            "P_good","P_good_CI_lo","P_good_CI_hi",
        ])
        w.writerows(cell_metrics)

    # Replicate-level metrics for error bars:
    # - original: replicate = trial (one run per trial for each eps,kappa)
    # - rewired/lesion: replicate = surrogate_id (aggregate across trials)
    rep_metrics = []
    rep_buckets: Dict[Tuple[str,int,float,float], List[RunRow]] = defaultdict(list)  # (variant, rep_id, eps, kappa)
    for rr in rows:
        if rr.variant == "original":
            rep_id = rr.trial
        else:
            rep_id = rr.surrogate_id
        rep_buckets[(rr.variant, int(rep_id), float(rr.epsilon), float(rr.kappa))].append(rr)

    for (variant, rep_id, eps, kappa), rs in sorted(rep_buckets.items(), key=lambda x: (x[0][0], x[0][3], x[0][2], x[0][1])):
        pT = np.array([r.pT for r in rs], dtype=float)
        pD = np.array([r.pD for r in rs], dtype=float)
        sel = np.array([r.sel for r in rs], dtype=float)

        pT_mean = float(np.mean(pT))
        pD_mean = float(np.mean(pD))
        ratio = float(pT_mean / (pD_mean + 1e-12))

        mean_sel = float(np.mean(sel))
        good = (sel > float(args.sel_thresh)) & (pT > float(args.pT_min))
        p_good = float(np.mean(good)) if len(good) else float("nan")

        rep_metrics.append([variant, rep_id, eps, kappa, len(rs), pT_mean, pD_mean, ratio, mean_sel, p_good])

    rep_path = os.path.join(out_dir, "A8_6_replicate_metrics.csv")
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant","replicate_id","epsilon","kappa","n_runs_in_rep",
            "P_sink_T_end_mean","P_sink_D_end_mean",
            "Selectivity_ratioOfMeans_rep","Selectivity_meanOfRatios_rep",
            "P_good_rep",
        ])
        w.writerows(rep_metrics)

    # Pairwise tests at epsilon=3 (per kappa)
    # - Good-run: Fisher exact
    # - Selectivity_end: Mann-Whitney U on log(sel)
    try:
        from scipy.stats import fisher_exact, mannwhitneyu
        have_scipy = True
    except Exception:
        have_scipy = False

    tests_rows = []
    if have_scipy:
        kappas = sorted(set(float(r.kappa) for r in rows))
        variants = sorted(set(r.variant for r in rows))
        pairs = []
        for i in range(len(variants)):
            for j in range(i+1, len(variants)):
                pairs.append((variants[i], variants[j]))

        for kappa in kappas:
            # runwise at epsilon=3
            for v1, v2 in pairs:
                rs1 = [r for r in rows if (r.variant == v1 and abs(r.kappa - kappa) < 1e-12 and abs(r.epsilon - 3.0) < 1e-12)]
                rs2 = [r for r in rows if (r.variant == v2 and abs(r.kappa - kappa) < 1e-12 and abs(r.epsilon - 3.0) < 1e-12)]
                if len(rs1) == 0 or len(rs2) == 0:
                    continue

                good1 = [(r.sel > args.sel_thresh) and (r.pT > args.pT_min) for r in rs1]
                good2 = [(r.sel > args.sel_thresh) and (r.pT > args.pT_min) for r in rs2]
                a = int(sum(good1))
                b = int(len(good1) - a)
                c = int(sum(good2))
                d = int(len(good2) - c)
                # Fisher exact
                try:
                    OR, p_fish = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                except Exception:
                    OR, p_fish = (float("nan"), float("nan"))

                # Mann-Whitney on log(sel)
                x1 = np.log(np.array([max(r.sel, 1e-12) for r in rs1], dtype=float))
                x2 = np.log(np.array([max(r.sel, 1e-12) for r in rs2], dtype=float))
                try:
                    U, p_mw = mannwhitneyu(x1, x2, alternative="two-sided")
                except Exception:
                    U, p_mw = (float("nan"), float("nan"))

                tests_rows.append([
                    3.0, kappa, v1, v2,
                    len(rs1), len(rs2),
                    a, b, c, d,
                    float(OR), float(p_fish),
                    float(p_mw),
                ])

    tests_path = os.path.join(out_dir, "A8_6_pairwise_tests_epsilon3.csv")
    with open(tests_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epsilon","kappa","variant_A","variant_B",
            "nA","nB",
            "good_A","notgood_A","good_B","notgood_B",
            "odds_ratio_Fisher","p_Fisher_goodrun",
            "p_MannWhitney_logSelectivity",
        ])
        w.writerows(tests_rows)

    # Plots
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            # read replicate metrics back (easy for error bars)
            # build dict: (variant, eps, kappa) -> list of replicate ratios
            rep_ratio = defaultdict(list)
            rep_good = defaultdict(list)
            rep_cov = defaultdict(list)

            # For coverage we need per replicate mean coverage; compute from rows
            # original replicate = trial, so one run, cov is cov; others aggregate over trials
            rep_cov_buckets = defaultdict(list)
            for rr in rows:
                if rr.variant == "original":
                    rep_id = rr.trial
                else:
                    rep_id = rr.surrogate_id
                rep_cov_buckets[(rr.variant, rep_id, float(rr.epsilon), float(rr.kappa))].append(rr.cov)

            for (variant, rep_id, eps, kappa), covs in rep_cov_buckets.items():
                key = (variant, eps, kappa)
                rep_cov[key].append(float(np.mean(covs)))

            # ratio and good from rep_metrics list we already have in memory
            for rm in rep_metrics:
                variant, rep_id, eps, kappa = rm[0], rm[1], float(rm[2]), float(rm[3])
                key = (variant, eps, kappa)
                rep_ratio[key].append(float(rm[7]))  # Selectivity_ratioOfMeans_rep
                rep_good[key].append(float(rm[9]))   # P_good_rep

            kappas = sorted(set(float(r.kappa) for r in rows))
            eps_values = sorted(set(float(r.epsilon) for r in rows))
            variants = sorted(set(r.variant for r in rows))

            def _plot_metric(metric_dict, ylabel, title_prefix, fname_prefix):
                for kappa in kappas:
                    plt.figure()
                    for variant in variants:
                        xs, ys, es = [], [], []
                        for eps in eps_values:
                            key = (variant, eps, kappa)
                            vals = metric_dict.get(key, [])
                            if not vals:
                                continue
                            xs.append(eps)
                            ys.append(float(np.mean(vals)))
                            # SEM (across replicates)
                            if len(vals) > 1:
                                es.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))))
                            else:
                                es.append(0.0)
                        if xs:
                            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=variant)
                    plt.xlabel("epsilon")
                    plt.ylabel(ylabel)
                    plt.title(f"{title_prefix} (kappa={kappa})")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    out_png = os.path.join(out_dir, f"{fname_prefix}_kappa{kappa}.png".replace(".","p"))
                    plt.savefig(out_png, dpi=180)
                    plt.close()

            _plot_metric(rep_ratio, "Selectivity_ratioOfMeans (replicate mean)", "A8.6: Selectivity (ratio-of-means)", "A8_6_Selectivity_ratioOfMeans_vs_epsilon")
            _plot_metric(rep_good, "P_good (replicate mean)", "A8.6: P_good", "A8_6_goodrun_prob_vs_epsilon")
            _plot_metric(rep_cov, "coverage_end (replicate mean)", "A8.6: coverage_end", "A8_6_coverage_end_vs_epsilon")

        except Exception as e:
            print(f"[A8.6] Plotting skipped: {e}")

    print("[A8.6] DONE")
    print(f"  Wrote: {cell_path}")
    print(f"  Wrote: {rep_path}")
    print(f"  Wrote: {tests_path}")
    print(f"  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
