import argparse
import csv
import json
import math
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
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _group_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _compute_objectives(pT: float, pD: float, T_max: float, lam: float, chi: float) -> Dict[str, float]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7_dir", required=True, help="Folder with A7_gksl_sweep.csv and A7_baselines.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for A7 analysis artifacts")
    ap.add_argument("--lambda_", type=float, default=1.0, help="lambda used in Utility and InfoPerCost")
    ap.add_argument("--chi", type=float, default=0.01, help="chi used in InfoPerCost")
    ap.add_argument("--pT_min", type=float, default=0.005, help="constraint for Selectivity_end (P_sink_T_end >= pT_min)")
    args = ap.parse_args()

    a7_dir = os.path.abspath(os.path.expanduser(args.a7_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    gksl_path = os.path.join(a7_dir, "A7_gksl_sweep.csv")
    base_path = os.path.join(a7_dir, "A7_baselines.csv")
    meta_path = os.path.join(a7_dir, "A7_meta.json")

    gksl = _read_csv(gksl_path)
    base = _read_csv(base_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if os.path.exists(meta_path) else {}

    T_max = float(meta.get("sim", {}).get("T_max", 10.0))

    lam = float(args.lambda_)
    chi = float(args.chi)

    # Aggregate GKSL: (epsilon, kappa) -> list of (pT, pD)
    g_by = defaultdict(list)
    eps_set = set()
    kappa_set = set()
    mode = None
    for r in gksl:
        eps = _as_float(r["epsilon"])
        kappa = _as_float(r["kappa"])
        pT = _as_float(r["P_sink_T_end"])
        pD = _as_float(r["P_sink_D_end"])
        eps_set.add(eps); kappa_set.add(kappa)
        mode = r.get("mode", mode)
        g_by[(eps, kappa)].append((pT, pD))

    eps_list = sorted(eps_set)
    kappa_list = sorted(kappa_set)

    # Aggregate baselines: (epsilon, model, T_env) -> list of (pT,pD)
    b_by = defaultdict(list)
    for r in base:
        eps = _as_float(r["epsilon"])
        model = r["model"]
        T_env = r.get("T_env", "")
        pT = _as_float(r["P_sink_T_end"])
        pD = _as_float(r["P_sink_D_end"])
        b_by[(eps, model, T_env)].append((pT, pD))

    # Build mean tables
    rows_summary = []
    for eps in eps_list:
        # GKSL mean curves
        curve = []
        for kappa in kappa_list:
            pts = g_by.get((eps, kappa), [])
            pT_mean, _ = _group_mean_std([p[0] for p in pts])
            pD_mean, _ = _group_mean_std([p[1] for p in pts])
            obj = _compute_objectives(pT_mean, pD_mean, T_max=T_max, lam=lam, chi=chi)
            curve.append((kappa, pT_mean, pD_mean, obj))

        # best_by_Utility (mean)
        bestU = None
        for kappa, pT, pD, obj in curve:
            if bestU is None or obj["Utility"] > bestU["Utility"]:
                bestU = {"kappa": kappa, "P_sink_T_end": pT, "P_sink_D_end": pD, **obj}

        # best_by_InfoPerCost (mean)
        bestIPC = None
        for kappa, pT, pD, obj in curve:
            if bestIPC is None or obj["InfoPerCost"] > bestIPC["InfoPerCost"]:
                bestIPC = {"kappa": kappa, "P_sink_T_end": pT, "P_sink_D_end": pD, **obj}

        # best_by_Selectivity_end constrained (mean)
        bestSel = None
        for kappa, pT, pD, obj in curve:
            if pT < float(args.pT_min):
                continue
            if bestSel is None or obj["Selectivity_end"] > bestSel["Selectivity_end"]:
                bestSel = {"kappa": kappa, "P_sink_T_end": pT, "P_sink_D_end": pD, **obj}
        if bestSel is None:
            bestSel = {"kappa": float("nan"), "note": f"No kappa satisfies P_sink_T_end >= {args.pT_min:g}."}

        # baselines
        baseline_rows = []
        for (eps2, model2, T_env2), pts in b_by.items():
            if abs(eps2 - eps) > 1e-12:
                continue
            pT_mean, _ = _group_mean_std([p[0] for p in pts])
            pD_mean, _ = _group_mean_std([p[1] for p in pts])
            obj = _compute_objectives(pT_mean, pD_mean, T_max=T_max, lam=lam, chi=chi)
            baseline_rows.append((model2, T_env2, pT_mean, pD_mean, obj))

        # Save a compact summary row (for easiest reading)
        def _pack(prefix: str, rec: dict) -> Dict[str, float]:
            out = {}
            for k in ["kappa", "P_sink_T_end", "P_sink_D_end", "Selectivity_end", "coverage_end", "Utility", "InfoPerCost"]:
                if k in rec:
                    out[f"{prefix}_{k}"] = rec[k]
            return out

        row = {"mode": mode, "epsilon": eps, "lambda": lam, "chi": chi, "pT_min": float(args.pT_min)}
        row.update(_pack("GKSL_bestU", bestU))
        row.update(_pack("GKSL_bestIPC", bestIPC))
        row.update(_pack("GKSL_bestSel", bestSel if isinstance(bestSel, dict) else {}))

        # Append baseline fields (may be multiple T_envs; we include a few canonical names)
        for model2, T_env2, pT_mean, pD_mean, obj in baseline_rows:
            tag = f"{model2}" + (f"_T{T_env2}" if str(T_env2).strip() != "" else "")
            row[f"{tag}_P_sink_T_end"] = pT_mean
            row[f"{tag}_P_sink_D_end"] = pD_mean
            row[f"{tag}_Selectivity_end"] = obj["Selectivity_end"]
            row[f"{tag}_Utility"] = obj["Utility"]
            row[f"{tag}_InfoPerCost"] = obj["InfoPerCost"]
        rows_summary.append(row)

    # Write summary CSV
    out_csv = os.path.join(out_dir, "A7_summary_by_epsilon.csv")
    # dynamic header
    header = sorted({k for r in rows_summary for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_summary:
            w.writerow(r)

    # Optionally plot mean Selectivity_end curves
    if plt is not None and len(eps_list) > 0 and len(kappa_list) > 0:
        # Selectivity vs kappa by epsilon
        plt.figure()
        plt.xscale("log")
        for eps in eps_list:
            sel_means = []
            for kappa in kappa_list:
                pts = g_by.get((eps, kappa), [])
                pT_mean, _ = _group_mean_std([p[0] for p in pts])
                pD_mean, _ = _group_mean_std([p[1] for p in pts])
                sel = pT_mean / (pD_mean + 1e-12)
                sel_means.append(sel)
            plt.plot(kappa_list, sel_means, marker="o", label=f"ε={eps:g}")
        plt.xlabel("kappa")
        plt.ylabel("Selectivity_end (mean)")
        plt.title("GKSL: Selectivity_end vs kappa (by epsilon)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A7_selectivity_vs_kappa_by_epsilon.png"), dpi=200)
        plt.close()

        # Coverage vs kappa by epsilon (log y)
        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        for eps in eps_list:
            cov_means = []
            for kappa in kappa_list:
                pts = g_by.get((eps, kappa), [])
                pT_mean, _ = _group_mean_std([p[0] for p in pts])
                pD_mean, _ = _group_mean_std([p[1] for p in pts])
                cov_means.append(pT_mean + pD_mean)
            plt.plot(kappa_list, cov_means, marker="o", label=f"ε={eps:g}")
        plt.xlabel("kappa")
        plt.ylabel("coverage_end (mean)")
        plt.title("GKSL: coverage_end vs kappa (by epsilon)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A7_coverage_vs_kappa_by_epsilon.png"), dpi=200)
        plt.close()

        # Utility vs epsilon for best-by-Utility kappa (mean)
        eps_vals = []
        util_vals = []
        ipc_vals = []
        for r in rows_summary:
            eps_vals.append(float(r["epsilon"]))
            util_vals.append(float(r.get("GKSL_bestU_Utility", float("nan"))))
            ipc_vals.append(float(r.get("GKSL_bestIPC_InfoPerCost", float("nan"))))
        plt.figure()
        plt.plot(eps_vals, util_vals, marker="o")
        plt.xlabel("epsilon")
        plt.ylabel("Utility (mean best-by-Utility kappa)")
        plt.title(f"GKSL: Utility vs epsilon (lambda={lam:g})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A7_bestU_utility_vs_epsilon.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(eps_vals, ipc_vals, marker="o")
        plt.xlabel("epsilon")
        plt.ylabel("InfoPerCost (mean best-by-InfoPerCost kappa)")
        plt.title(f"GKSL: InfoPerCost vs epsilon (lambda={lam:g}, chi={chi:g})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A7_bestIPC_infopercost_vs_epsilon.png"), dpi=200)
        plt.close()

    out_json = os.path.join(out_dir, "A7_analysis_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "inputs": {"A7_gksl_sweep.csv": gksl_path, "A7_baselines.csv": base_path, "A7_meta.json": meta_path},
            "settings": {"lambda": lam, "chi": chi, "pT_min": float(args.pT_min)},
            "outputs": {"summary_csv": out_csv},
        }, f, ensure_ascii=False, indent=2)

    print("[A7_ANALYZE] DONE")
    print(f"  - {out_csv}")
    if plt is not None:
        print(f"  - plots: A7_selectivity_vs_kappa_by_epsilon.png, A7_coverage_vs_kappa_by_epsilon.png, ...")


if __name__ == "__main__":
    main()
