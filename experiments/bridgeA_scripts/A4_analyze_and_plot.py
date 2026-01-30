#!/usr/bin/env python3
"""
A4_analyze_and_plot.py
======================

Minimal, publication-friendly analysis of Bridge-A real-graph benchmark outputs.

Inputs (default names; you can override via CLI):
- A3_summary.json
- A3_kappa_sweep_GKSL.csv
- A3_timeseries_CRW.csv
- A3_timeseries_GKSL_best.csv

Outputs (written to --out_dir, default A4_outputs/):
- A4_kappa_sweep_enriched.csv
- A4_GKSL_kappa_vs_selectivity.png
- A4_GKSL_kappa_vs_Psink.png
- A4_timeseries_target.png
- A4_timeseries_distractor.png
- A4_quick_note.md

Dependencies: numpy, matplotlib (pandas NOT required).
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def plot_kappa_selectivity(kappa: np.ndarray, sel: np.ndarray, outpath: str) -> None:
    fig = plt.figure()
    plt.plot(kappa, sel, marker="o")
    plt.xscale("log")
    plt.xlabel("kappa")
    plt.ylabel("Selectivity_end")
    plt.title("GKSL κ-sweep: Selectivity_end")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_kappa_psink(kappa: np.ndarray, pT: np.ndarray, pD: np.ndarray, outpath: str) -> None:
    fig = plt.figure()
    plt.plot(kappa, pT, marker="o", label="P_sink_T_end")
    plt.plot(kappa, pD, marker="o", label="P_sink_D_end")
    plt.xscale("log")
    plt.xlabel("kappa")
    plt.ylabel("Probability at T_end")
    plt.title("GKSL κ-sweep: P_sink_*_end")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(t_crw, y_crw, t_gk, y_gk, ylabel: str, title: str, outpath: str) -> None:
    fig = plt.figure()
    plt.plot(t_crw, y_crw, label=f"CRW {ylabel}")
    plt.plot(t_gk, y_gk, label=f"GKSL(best) {ylabel}")
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", default="A3_summary.json")
    ap.add_argument("--kappa_csv", default="A3_kappa_sweep_GKSL.csv")
    ap.add_argument("--ts_crw_csv", default="A3_timeseries_CRW.csv")
    ap.add_argument("--ts_gksl_csv", default="A3_timeseries_GKSL_best.csv")
    ap.add_argument("--out_dir", default="A4_outputs")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    with open(args.summary_json, "r") as f:
        summary = json.load(f)

    # ---- kappa sweep ----
    krows = read_csv_dicts(args.kappa_csv)
    enriched: List[Dict[str, Any]] = []
    for r in krows:
        k = to_float(r.get("kappa"))
        pT = to_float(r.get("P_sink_T_end"))
        pD = to_float(r.get("P_sink_D_end"))
        sel = to_float(r.get("Selectivity_end"))
        p_total = pT + pD
        delta = pT - pD
        frac = pT / p_total if p_total > 0 else float("nan")
        enriched.append({
            **r,
            "P_total_end": p_total,
            "Delta_TminusD": delta,
            "FracTarget": frac
        })

    out_table = os.path.join(args.out_dir, "A4_kappa_sweep_enriched.csv")
    fieldnames = list(enriched[0].keys()) if enriched else ["kappa","P_sink_T_end","P_sink_D_end","Selectivity_end","P_total_end","Delta_TminusD","FracTarget"]
    write_csv(out_table, fieldnames, enriched)

    kappa = np.array([to_float(r.get("kappa")) for r in krows], dtype=float)
    sel = np.array([to_float(r.get("Selectivity_end")) for r in krows], dtype=float)
    pT = np.array([to_float(r.get("P_sink_T_end")) for r in krows], dtype=float)
    pD = np.array([to_float(r.get("P_sink_D_end")) for r in krows], dtype=float)

    plot_kappa_selectivity(kappa, sel, os.path.join(args.out_dir, "A4_GKSL_kappa_vs_selectivity.png"))
    plot_kappa_psink(kappa, pT, pD, os.path.join(args.out_dir, "A4_GKSL_kappa_vs_Psink.png"))

    # ---- timeseries ----
    crw = read_csv_dicts(args.ts_crw_csv)
    gk = read_csv_dicts(args.ts_gksl_csv)

    t_crw = np.array([to_float(r.get("t")) for r in crw], dtype=float)
    t_gk = np.array([to_float(r.get("t")) for r in gk], dtype=float)

    pT_crw = np.array([to_float(r.get("P_sink_T")) for r in crw], dtype=float)
    pD_crw = np.array([to_float(r.get("P_sink_D")) for r in crw], dtype=float)
    pT_gk = np.array([to_float(r.get("P_sink_T")) for r in gk], dtype=float)
    pD_gk = np.array([to_float(r.get("P_sink_D")) for r in gk], dtype=float)

    plot_timeseries(t_crw, pT_crw, t_gk, pT_gk, "P_sink_T", "Timeseries: Target sink probability", os.path.join(args.out_dir, "A4_timeseries_target.png"))
    plot_timeseries(t_crw, pD_crw, t_gk, pD_gk, "P_sink_D", "Timeseries: Distractor sink probability", os.path.join(args.out_dir, "A4_timeseries_distractor.png"))

    # ---- note ----
    note = []
    note.append("# A4: Real graph benchmark quick analysis (auto-generated)")
    note.append("")
    note.append("## Run summary (from A3_summary.json)")
    note.append(f"- mode: {summary.get('mode')}")
    if "active_subgraph" in summary:
        note.append(f"- active_subgraph: N_nodes={summary['active_subgraph'].get('N_nodes')}, N_edges={summary['active_subgraph'].get('N_edges')}")
    if "sim" in summary:
        note.append(f"- sim: gamma={summary['sim'].get('gamma')}, eta_sink={summary['sim'].get('eta_sink')}, T_max={summary['sim'].get('T_max')}, dt={summary['sim'].get('dt','(see config)')}")
    note.append("")
    if "classical_CRW" in summary:
        note.append("### Classical_CRW @ T_end")
        note.append(f"- P_sink_T_end = {summary['classical_CRW'].get('P_sink_T_end')}")
        note.append(f"- P_sink_D_end = {summary['classical_CRW'].get('P_sink_D_end')}")
        note.append(f"- Selectivity_end = {summary['classical_CRW'].get('Selectivity_end')}")
        note.append("")
    if "gksl_GKSL" in summary and "best_by_Selectivity_end" in summary["gksl_GKSL"]:
        best = summary["gksl_GKSL"]["best_by_Selectivity_end"]
        note.append("### GKSL best_by_Selectivity_end @ T_end")
        note.append(f"- kappa = {best.get('kappa')}")
        note.append(f"- P_sink_T_end = {best.get('P_sink_T_end')}")
        note.append(f"- P_sink_D_end = {best.get('P_sink_D_end')}")
        note.append(f"- Selectivity_end = {best.get('Selectivity_end')}")
        note.append("")
    note.append("## Output files")
    note.append("- A4_kappa_sweep_enriched.csv")
    note.append("- A4_GKSL_kappa_vs_selectivity.png")
    note.append("- A4_GKSL_kappa_vs_Psink.png")
    note.append("- A4_timeseries_target.png")
    note.append("- A4_timeseries_distractor.png")
    with open(os.path.join(args.out_dir, "A4_quick_note.md"), "w") as f:
        f.write("\n".join(note))

    print(f"[A4] Wrote outputs to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
