import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _as_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _objectives_from_pT_pD(pT: float, pD: float, T_max: float, lam: float, chi: float) -> Dict[str, float]:
    sel = pT / (pD + 1e-12)
    cov = pT + pD
    prec = pT / cov if cov > 0 else float("nan")
    U = pT - lam * pD
    cost = cov + chi
    ipc = (U / cost) if cost > 0 else float("nan")
    return {
        "Selectivity_end": sel,
        "coverage_end": cov,
        "precision_end": prec,
        "Utility": U,
        "throughput_total_end": cov / T_max,
        "throughput_correct_end": pT / T_max,
        "throughput_net_end": (pT - pD) / T_max,
        "InfoPerCost": ipc,
    }


def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) if arr.size > 1 else 0.0)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 0, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the MEAN of `values` (1D array).
    Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = v.size
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(v, size=n, replace=True)
        means[i] = np.mean(samp)
    alpha = (1.0 - ci) / 2.0
    return float(np.mean(v)), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _paired_bootstrap_ci(diff_values: np.ndarray, n_boot: int = 20000, seed: int = 0, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Paired bootstrap CI for the MEAN of paired differences.
    diff_values is a 1D array of (g_i - b_i).
    """
    return _bootstrap_ci(np.array(diff_values, dtype=float), n_boot=n_boot, seed=seed, ci=ci)


def _bootstrap_best_by_objective(
    pT: np.ndarray,
    pD: np.ndarray,
    kappas: np.ndarray,
    objective: str,
    T_max: float,
    lam: float,
    chi: float,
    pT_min: float,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """
    pT, pD: arrays shaped (n_trials, n_kappa)
    Returns CI for the 'best' objective across kappa (global selection per bootstrap sample).
    We compute objectives from mean(pT), mean(pD) within each bootstrap resample.
    """
    rng = np.random.default_rng(seed)
    n_trials, n_kappa = pT.shape

    best_vals = np.empty(n_boot, dtype=float)
    best_kappa = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.choice(np.arange(n_trials), size=n_trials, replace=True)
        # mean across resampled trials
        pT_m = np.mean(pT[idx, :], axis=0)
        pD_m = np.mean(pD[idx, :], axis=0)

        # compute objective curve
        obj_curve = np.empty(n_kappa, dtype=float)
        obj_curve[:] = np.nan
        for j in range(n_kappa):
            obj = _objectives_from_pT_pD(float(pT_m[j]), float(pD_m[j]), T_max=T_max, lam=lam, chi=chi)
            if objective == "Selectivity_end" and float(pT_m[j]) < pT_min:
                obj_curve[j] = np.nan
            else:
                obj_curve[j] = float(obj[objective])

        # pick best
        j_best = int(np.nanargmax(obj_curve)) if np.any(~np.isnan(obj_curve)) else -1
        if j_best < 0:
            best_vals[b] = np.nan
            best_kappa[b] = np.nan
        else:
            best_vals[b] = obj_curve[j_best]
            best_kappa[b] = float(kappas[j_best])

    # CI for best_vals
    v = best_vals[~np.isnan(best_vals)]
    if v.size == 0:
        return {"best_mean": float("nan"), "best_ci_lo": float("nan"), "best_ci_hi": float("nan"), "kappa_mode": float("nan")}
    alpha = 0.025
    out = {
        "best_mean": float(np.mean(v)),
        "best_ci_lo": float(np.quantile(v, alpha)),
        "best_ci_hi": float(np.quantile(v, 1 - alpha)),
    }
    # kappa mode (most frequent)
    k = best_kappa[~np.isnan(best_kappa)]
    if k.size == 0:
        out["kappa_mode"] = float("nan")
    else:
        # round to stable buckets for mode
        buckets = {}
        for kk in k:
            key = f"{kk:.6g}"
            buckets[key] = buckets.get(key, 0) + 1
        mode_key = max(buckets.items(), key=lambda kv: kv[1])[0]
        out["kappa_mode"] = float(mode_key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7_dir", required=True, help="Folder with A7_gksl_sweep.csv / A7_baselines.csv / A7_meta.json")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--lambda_", type=float, default=1.0, help="lambda for Utility / InfoPerCost")
    ap.add_argument("--chi", type=float, default=0.01, help="chi in InfoPerCost denominator (cost = coverage + chi)")
    ap.add_argument("--pT_min", type=float, default=0.005, help="constraint for Selectivity_end best (mean pT >= pT_min)")
    ap.add_argument("--baseline_T_env", type=float, default=0.1, help="which CRW_thermal temperature to use for comparisons")
    ap.add_argument("--n_boot", type=int, default=20000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    a7_dir = os.path.abspath(os.path.expanduser(args.a7_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    g_path = os.path.join(a7_dir, "A7_gksl_sweep.csv")
    b_path = os.path.join(a7_dir, "A7_baselines.csv")
    m_path = os.path.join(a7_dir, "A7_meta.json")

    g_rows = _read_csv(g_path)
    b_rows = _read_csv(b_path)
    meta = json.load(open(m_path, "r", encoding="utf-8")) if os.path.exists(m_path) else {}

    T_max = float(meta.get("sim", {}).get("T_max", 10.0))
    lam = float(args.lambda_)
    chi = float(args.chi)
    pT_min = float(args.pT_min)
    T_env = float(args.baseline_T_env)

    # --- GKSL: build arrays per epsilon ---
    eps_set = sorted({ _as_float(r["epsilon"]) for r in g_rows })
    kappa_set = sorted({ _as_float(r["kappa"]) for r in g_rows })
    trial_set = sorted({ int(float(r["trial"])) for r in g_rows })

    kappas = np.array(kappa_set, dtype=float)
    n_kappa = kappas.size
    n_trials = len(trial_set)

    # index maps
    kappa_to_j = { float(k): j for j, k in enumerate(kappas) }
    trial_to_i = { int(t): i for i, t in enumerate(trial_set) }

    # store per epsilon: pT[i,j], pD[i,j]
    g_store = {}  # eps -> (pT, pD)
    mode = None
    for eps in eps_set:
        pT = np.full((n_trials, n_kappa), np.nan, dtype=float)
        pD = np.full((n_trials, n_kappa), np.nan, dtype=float)
        for r in g_rows:
            eps_r = _as_float(r["epsilon"])
            if abs(eps_r - eps) > 1e-12:
                continue
            mode = r.get("mode", mode)
            i = trial_to_i[int(float(r["trial"]))]
            j = kappa_to_j[float(_as_float(r["kappa"]))]
            pT[i, j] = _as_float(r["P_sink_T_end"])
            pD[i, j] = _as_float(r["P_sink_D_end"])
        g_store[eps] = (pT, pD)

    # --- Baselines: extract CRW + CRW_thermal at chosen T_env ---
    # baseline per epsilon & trial
    b_store = defaultdict(lambda: {"CRW": (np.nan, np.nan), "CRW_thermal": (np.nan, np.nan)})
    for r in b_rows:
        eps = _as_float(r["epsilon"])
        trial = int(float(r["trial"]))
        model = r["model"]
        t_env = _as_float(r.get("T_env", float("nan")))
        pT = _as_float(r["P_sink_T_end"])
        pD = _as_float(r["P_sink_D_end"])
        if model == "CRW":
            b_store[(eps, trial)]["CRW"] = (pT, pD)
        elif model == "CRW_thermal" and abs(t_env - T_env) < 1e-12:
            b_store[(eps, trial)]["CRW_thermal"] = (pT, pD)

    # --- (1) GKSL curve stats table: mean±std across trials for each (eps,kappa) ---
    curve_rows = []
    for eps in eps_set:
        pT, pD = g_store[eps]
        for j, kappa in enumerate(kappas):
            # per-trial metrics
            vals = []
            for i in range(n_trials):
                obj = _objectives_from_pT_pD(float(pT[i, j]), float(pD[i, j]), T_max=T_max, lam=lam, chi=chi)
                vals.append(obj)
            # aggregate
            row = {"mode": mode, "epsilon": eps, "kappa": float(kappa), "n_trials": n_trials}
            # add pT/pD too
            row["P_sink_T_end_mean"], row["P_sink_T_end_std"] = _mean_std(pT[:, j])
            row["P_sink_D_end_mean"], row["P_sink_D_end_std"] = _mean_std(pD[:, j])
            for key in ["Selectivity_end", "coverage_end", "precision_end", "Utility", "InfoPerCost", "throughput_total_end", "throughput_correct_end", "throughput_net_end"]:
                arr = np.array([v[key] for v in vals], dtype=float)
                row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(arr)
            curve_rows.append(row)

    curve_csv = os.path.join(out_dir, "A8_gksl_curve_stats.csv")
    header = sorted({k for r in curve_rows for k in r.keys()})
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)

    # --- (2) Baseline stats ---
    base_rows_out = []
    for eps in eps_set:
        for model_name in ["CRW", "CRW_thermal"]:
            pT_list = []
            pD_list = []
            for t in trial_set:
                pT, pD = b_store[(eps, t)][model_name]
                pT_list.append(pT)
                pD_list.append(pD)
            pT_arr = np.array(pT_list, dtype=float)
            pD_arr = np.array(pD_list, dtype=float)
            # per-trial objectives
            obj_list = [_objectives_from_pT_pD(float(pT_arr[i]), float(pD_arr[i]), T_max=T_max, lam=lam, chi=chi) for i in range(n_trials)]
            row = {
                "mode": mode,
                "epsilon": eps,
                "model": model_name,
                "T_env": (T_env if model_name == "CRW_thermal" else ""),
                "n_trials": n_trials,
            }
            row["P_sink_T_end_mean"], row["P_sink_T_end_std"] = _mean_std(pT_arr)
            row["P_sink_D_end_mean"], row["P_sink_D_end_std"] = _mean_std(pD_arr)
            for key in ["Selectivity_end", "coverage_end", "precision_end", "Utility", "InfoPerCost", "throughput_total_end", "throughput_correct_end", "throughput_net_end"]:
                arr = np.array([o[key] for o in obj_list], dtype=float)
                row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(arr)
            base_rows_out.append(row)

    base_csv = os.path.join(out_dir, "A8_baseline_stats.csv")
    header = sorted({k for r in base_rows_out for k in r.keys()})
    with open(base_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in base_rows_out:
            w.writerow(r)

    # --- (3) Best-by-objective with bootstrap CI (selection uncertainty included) ---
    best_rows = []
    for eps in eps_set:
        pT, pD = g_store[eps]
        # best by mean InfoPerCost (ratio-of-means within sample; same as A7)
        mean_pT = np.mean(pT, axis=0)
        mean_pD = np.mean(pD, axis=0)
        obj_curve = np.array([_objectives_from_pT_pD(float(mean_pT[j]), float(mean_pD[j]), T_max=T_max, lam=lam, chi=chi)["InfoPerCost"] for j in range(n_kappa)], dtype=float)
        j_best = int(np.nanargmax(obj_curve))
        best_kappa_ipc = float(kappas[j_best])

        # bootstrap best by objective (InfoPerCost, Utility, Selectivity_end constrained)
        boot_ipc = _bootstrap_best_by_objective(pT, pD, kappas, "InfoPerCost", T_max, lam, chi, pT_min, args.n_boot, seed=args.seed + 10 + int(1000 * eps))
        boot_U   = _bootstrap_best_by_objective(pT, pD, kappas, "Utility", T_max, lam, chi, pT_min, args.n_boot, seed=args.seed + 20 + int(1000 * eps))
        boot_Sel = _bootstrap_best_by_objective(pT, pD, kappas, "Selectivity_end", T_max, lam, chi, pT_min, args.n_boot, seed=args.seed + 30 + int(1000 * eps))

        best_rows.append({
            "mode": mode,
            "epsilon": eps,
            "lambda": lam,
            "chi": chi,
            "pT_min": pT_min,
            "T_max": T_max,
            "baseline_T_env": T_env,
            "bestIPC_kappa_ratioOfMeans": best_kappa_ipc,
            "bestIPC_InfoPerCost_boot_mean": boot_ipc["best_mean"],
            "bestIPC_InfoPerCost_ci_lo": boot_ipc["best_ci_lo"],
            "bestIPC_InfoPerCost_ci_hi": boot_ipc["best_ci_hi"],
            "bestIPC_kappa_mode_boot": boot_ipc["kappa_mode"],
            "bestU_Utility_boot_mean": boot_U["best_mean"],
            "bestU_Utility_ci_lo": boot_U["best_ci_lo"],
            "bestU_Utility_ci_hi": boot_U["best_ci_hi"],
            "bestU_kappa_mode_boot": boot_U["kappa_mode"],
            "bestSel_Selectivity_boot_mean": boot_Sel["best_mean"],
            "bestSel_Selectivity_ci_lo": boot_Sel["best_ci_lo"],
            "bestSel_Selectivity_ci_hi": boot_Sel["best_ci_hi"],
            "bestSel_kappa_mode_boot": boot_Sel["kappa_mode"],
        })

    best_csv = os.path.join(out_dir, "A8_best_by_objective_bootstrap.csv")
    header = sorted({k for r in best_rows for k in r.keys()})
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in best_rows:
            w.writerow(r)

    # --- (4) Paired effects vs baseline thermal on InfoPerCost and Utility at kappa_mode (bootstrap) ---
    # We compare GKSL at the "ratio-of-means best IPC kappa" (to match A7) against CRW_thermal (paired by trial).
    effects_rows = []
    for eps in eps_set:
        pT, pD = g_store[eps]
        # choose kappa by ratio-of-means best IPC (A7-compatible)
        mean_pT = np.mean(pT, axis=0)
        mean_pD = np.mean(pD, axis=0)
        ipc_curve = np.array([_objectives_from_pT_pD(float(mean_pT[j]), float(mean_pD[j]), T_max=T_max, lam=lam, chi=chi)["InfoPerCost"] for j in range(n_kappa)], dtype=float)
        j_best = int(np.nanargmax(ipc_curve))
        k_best = float(kappas[j_best])

        # per-trial GKSL objectives at that kappa
        g_ipc = []
        g_U = []
        for i in range(n_trials):
            obj = _objectives_from_pT_pD(float(pT[i, j_best]), float(pD[i, j_best]), T_max=T_max, lam=lam, chi=chi)
            g_ipc.append(obj["InfoPerCost"])
            g_U.append(obj["Utility"])
        g_ipc = np.array(g_ipc, dtype=float)
        g_U = np.array(g_U, dtype=float)

        # per-trial baseline thermal objectives
        b_ipc = []
        b_U = []
        for t in trial_set:
            pT_b, pD_b = b_store[(eps, t)]["CRW_thermal"]
            obj = _objectives_from_pT_pD(float(pT_b), float(pD_b), T_max=T_max, lam=lam, chi=chi)
            b_ipc.append(obj["InfoPerCost"])
            b_U.append(obj["Utility"])
        b_ipc = np.array(b_ipc, dtype=float)
        b_U = np.array(b_U, dtype=float)

        diff_ipc = g_ipc - b_ipc
        diff_U = g_U - b_U

        ipc_mean, ipc_lo, ipc_hi = _paired_bootstrap_ci(diff_ipc, n_boot=args.n_boot, seed=args.seed + 101 + int(1000*eps))
        U_mean, U_lo, U_hi = _paired_bootstrap_ci(diff_U, n_boot=args.n_boot, seed=args.seed + 202 + int(1000*eps))

        effects_rows.append({
            "mode": mode,
            "epsilon": eps,
            "kappa_bestIPC_ratioOfMeans": k_best,
            "delta_InfoPerCost_mean": ipc_mean,
            "delta_InfoPerCost_ci_lo": ipc_lo,
            "delta_InfoPerCost_ci_hi": ipc_hi,
            "delta_Utility_mean": U_mean,
            "delta_Utility_ci_lo": U_lo,
            "delta_Utility_ci_hi": U_hi,
        })

    effects_csv = os.path.join(out_dir, "A8_effects_vs_thermal.csv")
    header = sorted({k for r in effects_rows for k in r.keys()})
    with open(effects_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in effects_rows:
            w.writerow(r)

    # --- Plots (optional) ---
    if plt is not None:
        # Read best table into arrays
        eps_vals = np.array([r["epsilon"] for r in best_rows], dtype=float)

        ipc_mean = np.array([r["bestIPC_InfoPerCost_boot_mean"] for r in best_rows], dtype=float)
        ipc_lo = np.array([r["bestIPC_InfoPerCost_ci_lo"] for r in best_rows], dtype=float)
        ipc_hi = np.array([r["bestIPC_InfoPerCost_ci_hi"] for r in best_rows], dtype=float)

        U_mean = np.array([r["bestU_Utility_boot_mean"] for r in best_rows], dtype=float)
        U_lo = np.array([r["bestU_Utility_ci_lo"] for r in best_rows], dtype=float)
        U_hi = np.array([r["bestU_Utility_ci_hi"] for r in best_rows], dtype=float)

        # Baseline thermal mean+CI for InfoPerCost and Utility
        # (bootstrap mean across trials, no selection)
        base_ipc_mean = []
        base_ipc_lo = []
        base_ipc_hi = []
        base_U_mean = []
        base_U_lo = []
        base_U_hi = []

        for eps in eps_vals:
            # collect per-trial baseline objectives
            vals_ipc = []
            vals_U = []
            for t in trial_set:
                pT_b, pD_b = b_store[(float(eps), int(t))]["CRW_thermal"]
                obj = _objectives_from_pT_pD(float(pT_b), float(pD_b), T_max=T_max, lam=lam, chi=chi)
                vals_ipc.append(obj["InfoPerCost"])
                vals_U.append(obj["Utility"])
            m, lo, hi = _bootstrap_ci(np.array(vals_ipc, dtype=float), n_boot=args.n_boot, seed=args.seed + 301 + int(1000*eps))
            base_ipc_mean.append(m); base_ipc_lo.append(lo); base_ipc_hi.append(hi)
            m, lo, hi = _bootstrap_ci(np.array(vals_U, dtype=float), n_boot=args.n_boot, seed=args.seed + 401 + int(1000*eps))
            base_U_mean.append(m); base_U_lo.append(lo); base_U_hi.append(hi)

        base_ipc_mean = np.array(base_ipc_mean, dtype=float)
        base_ipc_lo = np.array(base_ipc_lo, dtype=float)
        base_ipc_hi = np.array(base_ipc_hi, dtype=float)

        base_U_mean = np.array(base_U_mean, dtype=float)
        base_U_lo = np.array(base_U_lo, dtype=float)
        base_U_hi = np.array(base_U_hi, dtype=float)

        # Plot IPC
        plt.figure()
        plt.errorbar(eps_vals, ipc_mean, yerr=[np.maximum(ipc_mean - ipc_lo, 0), np.maximum(ipc_hi - ipc_mean, 0)], fmt="o-", capsize=3, label="GKSL best-by-InfoPerCost (boot)")
        plt.errorbar(eps_vals, base_ipc_mean, yerr=[np.maximum(base_ipc_mean - base_ipc_lo, 0), np.maximum(base_ipc_hi - base_ipc_mean, 0)], fmt="o--", capsize=3, label=f"CRW_thermal (T_env={T_env:g})")
        plt.xlabel("epsilon")
        plt.ylabel("InfoPerCost (bootstrap CI)")
        plt.title(f"A8: InfoPerCost vs epsilon (lambda={lam:g}, chi={chi:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A8_InfoPerCost_vs_epsilon_withCI.png"), dpi=200)
        plt.close()

        # Plot Utility
        plt.figure()
        plt.errorbar(eps_vals, U_mean, yerr=[np.maximum(U_mean - U_lo, 0), np.maximum(U_hi - U_mean, 0)], fmt="o-", capsize=3, label="GKSL best-by-Utility (boot)")
        plt.errorbar(eps_vals, base_U_mean, yerr=[np.maximum(base_U_mean - base_U_lo, 0), np.maximum(base_U_hi - base_U_mean, 0)], fmt="o--", capsize=3, label=f"CRW_thermal (T_env={T_env:g})")
        plt.xlabel("epsilon")
        plt.ylabel("Utility (bootstrap CI)")
        plt.title(f"A8: Utility vs epsilon (lambda={lam:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A8_Utility_vs_epsilon_withCI.png"), dpi=200)
        plt.close()

    # --- metadata ---
    out_meta = os.path.join(out_dir, "A8_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "inputs": {"A7_gksl_sweep.csv": g_path, "A7_baselines.csv": b_path, "A7_meta.json": m_path},
            "settings": {
                "lambda": lam, "chi": chi, "pT_min": pT_min, "baseline_T_env": T_env,
                "n_boot": int(args.n_boot), "seed": int(args.seed)
            },
            "outputs": {
                "A8_gksl_curve_stats.csv": curve_csv,
                "A8_baseline_stats.csv": base_csv,
                "A8_best_by_objective_bootstrap.csv": best_csv,
                "A8_effects_vs_thermal.csv": effects_csv,
            }
        }, f, ensure_ascii=False, indent=2)

    print("[A8.1] DONE")
    print(f"  - {curve_csv}")
    print(f"  - {base_csv}")
    print(f"  - {best_csv}")
    print(f"  - {effects_csv}")
    if plt is not None:
        print("  - plots: A8_InfoPerCost_vs_epsilon_withCI.png, A8_Utility_vs_epsilon_withCI.png")


if __name__ == "__main__":
    main()
