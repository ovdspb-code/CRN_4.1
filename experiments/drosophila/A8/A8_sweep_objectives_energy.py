#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A8: Sweep objective parameters (lambda, chi) on top of A7 outputs.

Inputs (a7_dir):
  - A7_gksl_sweep.csv
  - A7_baselines.csv
  - A7_meta.json

Outputs (out_dir):
  - A8_objective_sweep.csv  (one row per (lambda, chi, epsilon))
  - A8_meta.json            (settings + input paths)

This script is deliberately lightweight: stdlib + numpy only.
Matplotlib is optional (not required).
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _as_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _mean_std_median(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr)), float(np.nanmedian(arr))


def compute_objectives(pT: float, pD: float, T_max: float, lam: float, chi: float) -> Dict[str, float]:
    sel = float(pT / (pD + 1e-12))
    cov = float(pT + pD)
    U = float(pT - lam * pD)
    cost = float(cov + chi)
    ipc = float(U / cost) if cost > 0 else float("nan")
    return {
        "P_sink_T_end": float(pT),
        "P_sink_D_end": float(pD),
        "Selectivity_end": sel,
        "coverage_end": cov,
        "Utility": U,
        "InfoPerCost": ipc,
        "throughput_total_end": cov / T_max if T_max > 0 else float("nan"),
        "throughput_correct_end": pT / T_max if T_max > 0 else float("nan"),
        "throughput_net_end": (pT - pD) / T_max if T_max > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7_dir", required=True, help="Folder with A7_gksl_sweep.csv, A7_baselines.csv, A7_meta.json")
    ap.add_argument("--out_dir", required=True, help="Output folder for A8 sweep artifacts")
    ap.add_argument("--lambda_grid", default="1.0", help="Comma-separated lambda values")
    ap.add_argument("--chi_grid", default="0.01", help="Comma-separated chi values")
    ap.add_argument("--pT_min", type=float, default=0.005, help="Constraint for bestSel: P_sink_T_end >= pT_min (uses MEAN pT)")
    ap.add_argument("--baseline_T_envs", default="0.1,1.0", help="Thermal baselines to include (comma-separated)")
    args = ap.parse_args()

    a7_dir = os.path.abspath(os.path.expanduser(args.a7_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    gksl_path = os.path.join(a7_dir, "A7_gksl_sweep.csv")
    base_path = os.path.join(a7_dir, "A7_baselines.csv")
    meta_path = os.path.join(a7_dir, "A7_meta.json")

    gksl = _read_csv(gksl_path)
    base = _read_csv(base_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    T_max = float(meta.get("sim", {}).get("T_max", 10.0))
    mode = meta.get("mode", None)

    lam_list = [float(x) for x in args.lambda_grid.split(",") if x.strip()]
    chi_list = [float(x) for x in args.chi_grid.split(",") if x.strip()]
    T_envs = [str(float(x)) for x in args.baseline_T_envs.split(",") if x.strip()]

    # ---- aggregate raw data
    # GKSL: (epsilon, kappa) -> list of (pT, pD) per trial
    g_by = defaultdict(list)
    eps_set = set()
    kappa_set = set()
    for r in gksl:
        eps = _as_float(r.get("epsilon"))
        kappa = _as_float(r.get("kappa"))
        pT = _as_float(r.get("P_sink_T_end"))
        pD = _as_float(r.get("P_sink_D_end"))
        if np.isnan(eps) or np.isnan(kappa):
            continue
        eps_set.add(eps)
        kappa_set.add(kappa)
        g_by[(eps, kappa)].append((pT, pD))

    eps_list = sorted(eps_set)
    kappa_list = sorted(kappa_set)

    # Baselines: (epsilon, model, T_env_str) -> list of (pT,pD)
    b_by = defaultdict(list)
    for r in base:
        eps = _as_float(r.get("epsilon"))
        model2 = str(r.get("model", ""))
        T_env_raw = r.get("T_env", "")
        T_env_str = "" if (T_env_raw is None or str(T_env_raw).strip() == "" or str(T_env_raw).strip().lower() == "nan") else str(float(T_env_raw))
        pT = _as_float(r.get("P_sink_T_end"))
        pD = _as_float(r.get("P_sink_D_end"))
        if np.isnan(eps):
            continue
        b_by[(eps, model2, T_env_str)].append((pT, pD))

    # ---- sweep objectives
    rows = []
    for lam in lam_list:
        for chi in chi_list:
            for eps in eps_list:
                # GKSL curve over kappa (mean-of-trials first)
                curve = []
                for kappa in kappa_list:
                    pts = g_by.get((eps, kappa), [])
                    pT_mean, _, _ = _mean_std_median([p[0] for p in pts])
                    pD_mean, _, _ = _mean_std_median([p[1] for p in pts])
                    obj_mean = compute_objectives(pT_mean, pD_mean, T_max=T_max, lam=lam, chi=chi)
                    curve.append((kappa, obj_mean, pts))

                # best-by objectives (on MEAN curve)
                bestIPC = None
                bestU = None
                bestSel = None
                for kappa, obj_mean, _pts in curve:
                    if bestIPC is None or obj_mean["InfoPerCost"] > bestIPC["InfoPerCost"]:
                        bestIPC = {"kappa": kappa, **obj_mean}
                    if bestU is None or obj_mean["Utility"] > bestU["Utility"]:
                        bestU = {"kappa": kappa, **obj_mean}
                    if obj_mean["P_sink_T_end"] >= float(args.pT_min):
                        if bestSel is None or obj_mean["Selectivity_end"] > bestSel["Selectivity_end"]:
                            bestSel = {"kappa": kappa, **obj_mean}

                # Trialwise stats at chosen kappa (for uncertainty bars)
                def _trial_stats_for(kappa_star: float) -> Dict[str, float]:
                    pts = g_by.get((eps, kappa_star), [])
                    pT_list = [p[0] for p in pts]
                    pD_list = [p[1] for p in pts]
                    ipc_list = []
                    U_list = []
                    sel_list = []
                    cov_list = []
                    for pT, pD in pts:
                        obj = compute_objectives(pT, pD, T_max=T_max, lam=lam, chi=chi)
                        ipc_list.append(obj["InfoPerCost"])
                        U_list.append(obj["Utility"])
                        sel_list.append(obj["Selectivity_end"])
                        cov_list.append(obj["coverage_end"])
                    ipc_mean, ipc_std, ipc_med = _mean_std_median(ipc_list)
                    U_mean, U_std, U_med = _mean_std_median(U_list)
                    sel_mean, sel_std, sel_med = _mean_std_median(sel_list)
                    cov_mean, cov_std, cov_med = _mean_std_median(cov_list)
                    return {
                        "trial_n": float(len(pts)),
                        "InfoPerCost_trial_mean": ipc_mean,
                        "InfoPerCost_trial_std": ipc_std,
                        "InfoPerCost_trial_median": ipc_med,
                        "Utility_trial_mean": U_mean,
                        "Utility_trial_std": U_std,
                        "Utility_trial_median": U_med,
                        "Selectivity_end_trial_mean": sel_mean,
                        "Selectivity_end_trial_std": sel_std,
                        "Selectivity_end_trial_median": sel_med,
                        "coverage_end_trial_mean": cov_mean,
                        "coverage_end_trial_std": cov_std,
                        "coverage_end_trial_median": cov_med,
                    }

                bestIPC_trial = _trial_stats_for(float(bestIPC["kappa"]))
                bestU_trial = _trial_stats_for(float(bestU["kappa"]))

                # Baselines: CRW and CRW_thermal (selected T_envs)
                def _baseline_mean_and_trial(model2: str, T_env_str: str) -> Tuple[Dict[str, float], Dict[str, float]]:
                    pts = b_by.get((eps, model2, T_env_str), [])
                    pT_mean, _, _ = _mean_std_median([p[0] for p in pts])
                    pD_mean, _, _ = _mean_std_median([p[1] for p in pts])
                    obj_mean = compute_objectives(pT_mean, pD_mean, T_max=T_max, lam=lam, chi=chi)

                    ipc_list = []
                    U_list = []
                    sel_list = []
                    cov_list = []
                    for pT, pD in pts:
                        obj = compute_objectives(pT, pD, T_max=T_max, lam=lam, chi=chi)
                        ipc_list.append(obj["InfoPerCost"])
                        U_list.append(obj["Utility"])
                        sel_list.append(obj["Selectivity_end"])
                        cov_list.append(obj["coverage_end"])
                    ipc_mean, ipc_std, ipc_med = _mean_std_median(ipc_list)
                    U_mean, U_std, U_med = _mean_std_median(U_list)
                    sel_mean, sel_std, sel_med = _mean_std_median(sel_list)
                    cov_mean, cov_std, cov_med = _mean_std_median(cov_list)

                    obj_trial = {
                        "trial_n": float(len(pts)),
                        "InfoPerCost_trial_mean": ipc_mean,
                        "InfoPerCost_trial_std": ipc_std,
                        "InfoPerCost_trial_median": ipc_med,
                        "Utility_trial_mean": U_mean,
                        "Utility_trial_std": U_std,
                        "Utility_trial_median": U_med,
                        "Selectivity_end_trial_mean": sel_mean,
                        "Selectivity_end_trial_std": sel_std,
                        "Selectivity_end_trial_median": sel_med,
                        "coverage_end_trial_mean": cov_mean,
                        "coverage_end_trial_std": cov_std,
                        "coverage_end_trial_median": cov_med,
                    }
                    return obj_mean, obj_trial

                crw_mean, crw_trial = _baseline_mean_and_trial("CRW", "")

                thermal_means = {}
                thermal_trials = {}
                for T_env_str in T_envs:
                    m, t = _baseline_mean_and_trial("CRW_thermal", T_env_str)
                    thermal_means[T_env_str] = m
                    thermal_trials[T_env_str] = t

                row = {
                    "mode": mode,
                    "epsilon": eps,
                    "T_max": T_max,
                    "lambda": lam,
                    "chi": chi,
                    "pT_min": float(args.pT_min),
                }

                # pack GKSL bests (mean-based)
                def _pack(prefix: str, rec: Dict[str, float]) -> None:
                    for k in ["kappa", "P_sink_T_end", "P_sink_D_end", "Selectivity_end", "coverage_end", "Utility", "InfoPerCost",
                              "throughput_total_end", "throughput_correct_end", "throughput_net_end"]:
                        if k in rec:
                            row[f"{prefix}_{k}"] = rec[k]

                def _pack_trial(prefix: str, rec: Dict[str, float]) -> None:
                    for k, v in rec.items():
                        row[f"{prefix}_{k}"] = v

                _pack("GKSL_bestIPC", bestIPC)
                _pack_trial("GKSL_bestIPC", bestIPC_trial)
                _pack("GKSL_bestU", bestU)
                _pack_trial("GKSL_bestU", bestU_trial)
                if bestSel is not None:
                    _pack("GKSL_bestSel", bestSel)
                else:
                    row["GKSL_bestSel_note"] = f"No kappa satisfies mean P_sink_T_end >= {args.pT_min:g}."

                # pack baselines
                _pack("CRW", crw_mean)
                _pack_trial("CRW", crw_trial)

                for T_env_str, rec_mean in thermal_means.items():
                    tag = f"CRW_thermal_T{T_env_str}"
                    _pack(tag, rec_mean)
                    _pack_trial(tag, thermal_trials[T_env_str])

                rows.append(row)

    # ---- write CSV
    out_csv = os.path.join(out_dir, "A8_objective_sweep.csv")
    header = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_meta = os.path.join(out_dir, "A8_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "inputs": {
                "A7_gksl_sweep.csv": gksl_path,
                "A7_baselines.csv": base_path,
                "A7_meta.json": meta_path,
            },
            "settings": {
                "lambda_grid": lam_list,
                "chi_grid": chi_list,
                "pT_min": float(args.pT_min),
                "baseline_T_envs": T_envs,
            },
            "outputs": {"A8_objective_sweep.csv": out_csv},
        }, f, ensure_ascii=False, indent=2)

    print("[A8] DONE")
    print(f"  - {out_csv}")
    print(f"  - {out_meta}")


if __name__ == "__main__":
    main()
