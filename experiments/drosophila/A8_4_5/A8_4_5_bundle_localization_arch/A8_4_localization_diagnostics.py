#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A8.4 — Localization diagnostics on the REAL Drosophila larva connectome graph.

What this script does (high level):
  1) Loads your REAL connectome graph.graphml + the frozen benchmark config (bench_config_*.json)
  2) Reconstructs the GKSL Hamiltonian node-subspace used in A7:
       H = -gamma * L_sym(p_uv) + diag(E)
     where E_i ~ U[-epsilon, +epsilon] with the SAME per-trial seeds as A7.
  3) Computes Anderson-style localization proxies on eigenmodes of H:
       PR_k = 1 / sum_i |psi_{i,k}|^4
     and compares modes overlapping Target vs Distractor sets.
  4) Joins those localization numbers with the *already computed* GKSL endpoints from A7
     (Selectivity_end, P_sink_T_end, P_sink_D_end) at a chosen kappa.

Outputs:
  - A8_4_localization_trialwise.csv
  - A8_4_localization_by_epsilon.csv
  - A8_4_deltaPR_vs_selectivity.png
  - A8_4_deltaPR_vs_epsilon.png

This is intentionally stdlib + numpy + scipy + networkx + matplotlib only.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _as_str_list(xs) -> List[str]:
    return [str(x) for x in xs]


def _normalize_outgoing_probs(G: nx.DiGraph, nodes: List[str], weight_attr: str) -> Dict[Tuple[str, str], float]:
    """Directed row-normalized transition probabilities p(u->v)=w/sum_out_w(u), restricted to node list."""
    node_set = set(map(str, nodes))
    p_uv: Dict[Tuple[str, str], float] = {}
    for u in nodes:
        u = str(u)
        if u not in node_set:
            continue
        out_edges = []
        wsum = 0.0
        for _, v, data in G.out_edges(u, data=True):
            v = str(v)
            if v not in node_set:
                continue
            w = float(data.get(weight_attr, 1.0))
            if w <= 0:
                continue
            out_edges.append((v, w))
            wsum += w
        if wsum <= 0:
            continue
        for v, w in out_edges:
            p_uv[(u, v)] = w / wsum
    return p_uv


def _symmetrized_laplacian_from_directed_probs(nodes: List[str], p_uv: Dict[Tuple[str, str], float]) -> csr_matrix:
    """Symmetric Laplacian L = D - A from directed p(u->v).

    a_ij = 0.5 * (p(i->j) + p(j->i)), implemented by accumulating p/2 for each directed edge.
    """
    node_list = [str(x) for x in nodes]
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    pairs: Dict[Tuple[int, int], float] = {}
    for (u, v), p in p_uv.items():
        if p <= 0:
            continue
        iu = idx.get(str(u))
        iv = idx.get(str(v))
        if iu is None or iv is None or iu == iv:
            continue
        key = (min(iu, iv), max(iu, iv))
        pairs[key] = pairs.get(key, 0.0) + 0.5 * float(p)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    deg = np.zeros(n, dtype=float)

    for (i, j), aij in pairs.items():
        if aij <= 0:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([aij, aij])
        deg[i] += aij
        deg[j] += aij

    A = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    L = diags(deg, 0, shape=(n, n), dtype=float) - A
    return L


def _load_a7_gksl_map(a7_gksl_csv: str) -> Dict[Tuple[float, int, float], Dict[str, float]]:
    """Map (epsilon, trial, kappa) -> endpoints."""
    out: Dict[Tuple[float, int, float], Dict[str, float]] = {}
    with open(a7_gksl_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eps = float(row["epsilon"])
            trial = int(row["trial"])
            kappa = float(row["kappa"])
            out[(eps, trial, kappa)] = {
                "P_sink_T_end": float(row["P_sink_T_end"]),
                "P_sink_D_end": float(row["P_sink_D_end"]),
                "Selectivity_end": float(row["Selectivity_end"]),
                "coverage_end": float(row["coverage_end"]),
            }
    return out


def _ipr_and_pr(evecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given eigenvectors matrix V (n x n), columns are eigenvectors.

    IPR_k = sum_i |psi_i|^4 ; PR_k = 1/IPR_k
    """
    # amplitudes squared
    amp2 = evecs * evecs
    ipr = np.sum(amp2 * amp2, axis=0)
    pr = 1.0 / np.maximum(ipr, 1e-30)
    return ipr, pr


def _overlap(evecs: np.ndarray, idx_set: np.ndarray) -> np.ndarray:
    """Overlap O_k = sum_{i in idx_set} |psi_i|^2."""
    amp2 = evecs * evecs
    return np.sum(amp2[idx_set, :], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7_dir", required=True, help="Path to folder that contains A7_meta.json and A7_gksl_sweep.csv")
    ap.add_argument("--out_dir", required=True, help="Where to write A8.4 outputs")
    ap.add_argument("--kappa", default="0.001", help="Which kappa to join from A7_gksl_sweep.csv (default: 0.001)")
    ap.add_argument("--topK", type=int, default=5, help="Top-K eigenmodes (by overlap) used for Target/Distractor PR summaries")
    ap.add_argument("--graphml", default="", help="Optional override for graph.graphml path")
    ap.add_argument("--config", default="", help="Optional override for bench_config_*.json path")
    args = ap.parse_args()

    a7_dir = os.path.abspath(os.path.expanduser(args.a7_dir))
    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(args.out_dir)))

    kappa_join = float(args.kappa)
    topK = int(args.topK)
    topK = max(1, min(topK, 50))

    meta_path = os.path.join(a7_dir, "A7_meta.json")
    gksl_path = os.path.join(a7_dir, "A7_gksl_sweep.csv")
    if not os.path.exists(meta_path):
        raise SystemExit(f"A7_meta.json not found: {meta_path}")
    if not os.path.exists(gksl_path):
        raise SystemExit(f"A7_gksl_sweep.csv not found: {gksl_path}")

    meta = _read_json(meta_path)

    config_path = args.config.strip() or meta.get("config", "")
    graphml_path = args.graphml.strip() or meta.get("graphml", "")
    if not config_path:
        raise SystemExit("Cannot resolve bench config path. Provide --config /path/to/bench_config_*.json")
    if not graphml_path:
        raise SystemExit("Cannot resolve graphml path. Provide --graphml /path/to/graph.graphml")

    config_path = os.path.abspath(os.path.expanduser(config_path))
    graphml_path = os.path.abspath(os.path.expanduser(graphml_path))

    if not os.path.exists(config_path):
        raise SystemExit(f"bench_config not found: {config_path}")
    if not os.path.exists(graphml_path):
        raise SystemExit(f"graphml not found: {graphml_path}")

    cfg = _read_json(config_path)

    weight_attr = cfg.get("weight_attr", "weight")
    source_nodes = _as_str_list(cfg["source_nodes"])
    target_nodes = _as_str_list(cfg["target_nodes"])
    distractor_nodes = _as_str_list(cfg["distractor_nodes"])
    active_nodes = _as_str_list(cfg["active_nodes"])

    sim = cfg.get("sim_defaults", {})
    gamma = float(sim.get("gamma", 1.0))

    disorder = meta.get("disorder", {})
    epsilons = [float(x) for x in disorder.get("epsilons", [])]
    n_trials = int(disorder.get("n_trials", 10))
    base_seed = int(disorder.get("base_seed", 42))

    # Load graph
    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    node_set = set(active_nodes)
    # IMPORTANT: match A7 ordering: iterate in the order of G.nodes()
    H_nodes = [str(n) for n in G.nodes() if str(n) in node_set]
    if len(H_nodes) == 0:
        raise SystemExit("Active subgraph is empty. Check active_nodes in bench_config.")
    Gs = G.subgraph(H_nodes).copy()

    # Directed probs and symmetric Laplacian
    p_uv = _normalize_outgoing_probs(Gs, H_nodes, weight_attr)
    L_sym = _symmetrized_laplacian_from_directed_probs(H_nodes, p_uv)

    # Base Hamiltonian (without disorder)
    H0 = (-gamma) * L_sym  # sparse, real symmetric

    # Index maps
    idx = {nm: i for i, nm in enumerate(H_nodes)}
    tgt_idx = np.array([idx[nm] for nm in target_nodes if nm in idx], dtype=int)
    dst_idx = np.array([idx[nm] for nm in distractor_nodes if nm in idx], dtype=int)
    if len(tgt_idx) == 0 or len(dst_idx) == 0:
        raise SystemExit("Target/Distractor nodes are not inside active subgraph; check config.")

    # Load GKSL endpoints from A7
    gksl_map = _load_a7_gksl_map(gksl_path)

    trialwise_csv = os.path.join(out_dir, "A8_4_localization_trialwise.csv")
    by_eps_csv = os.path.join(out_dir, "A8_4_localization_by_epsilon.csv")

    rows = []

    n = len(H_nodes)

    for eps in epsilons:
        for trial in range(n_trials):
            energy_seed = int(base_seed + 1000 * trial + int(round(eps * 100)))
            rng = np.random.RandomState(energy_seed)
            E = rng.uniform(-eps, eps, size=n).astype(float)

            # H = H0 + diag(E)
            H = (H0 + diags(E, 0, shape=(n, n), dtype=float)).toarray()

            # eigenmodes
            evals, evecs = np.linalg.eigh(H)
            ipr, pr = _ipr_and_pr(evecs)

            ov_T = _overlap(evecs, tgt_idx)
            ov_D = _overlap(evecs, dst_idx)

            # Top-K modes by overlap
            kT = min(topK, n)
            kD = min(topK, n)
            topT = np.argsort(-ov_T)[:kT]
            topD = np.argsort(-ov_D)[:kD]

            PR_all_mean = float(np.mean(pr))
            PR_T_topK = float(np.mean(pr[topT]))
            PR_D_topK = float(np.mean(pr[topD]))
            delta_PR = PR_T_topK - PR_D_topK
            ratio_PR = float(PR_T_topK / max(PR_D_topK, 1e-30))

            # Join with GKSL endpoints at chosen kappa
            key = (eps, trial, kappa_join)
            if key not in gksl_map:
                # allow float tolerance by searching
                found = None
                for (e2, tr2, k2), v in gksl_map.items():
                    if abs(e2 - eps) < 1e-12 and tr2 == trial and abs(k2 - kappa_join) < 1e-12:
                        found = v
                        break
                if found is None:
                    raise SystemExit(f"Cannot find A7 GKSL row for epsilon={eps}, trial={trial}, kappa={kappa_join}.")
                g = found
            else:
                g = gksl_map[key]

            rows.append([
                eps,
                trial,
                energy_seed,
                kappa_join,
                g["P_sink_T_end"],
                g["P_sink_D_end"],
                g["Selectivity_end"],
                g["coverage_end"],
                PR_all_mean,
                PR_T_topK,
                PR_D_topK,
                float(delta_PR),
                float(ratio_PR),
            ])

    # Write trialwise CSV
    with open(trialwise_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epsilon",
            "trial",
            "energy_seed",
            "kappa",
            "P_sink_T_end",
            "P_sink_D_end",
            "Selectivity_end",
            "coverage_end",
            "PR_all_mean",
            "PR_T_topK_mean",
            "PR_D_topK_mean",
            "delta_PR_TminusD",
            "ratio_PR_T_over_D",
        ])
        w.writerows(rows)

    # Aggregate by epsilon
    by_eps: Dict[float, List[List[float]]] = {}
    for r in rows:
        eps = float(r[0])
        by_eps.setdefault(eps, []).append(r)

    agg_rows = []
    for eps, rs in sorted(by_eps.items(), key=lambda x: x[0]):
        arr = np.array(rs, dtype=float)
        # columns in arr correspond to rows list above
        sel = arr[:, 6]
        dpr = arr[:, 11]
        pr_ratio = arr[:, 12]
        agg_rows.append([
            eps,
            len(rs),
            float(np.mean(sel)),
            float(np.std(sel, ddof=1) if len(rs) > 1 else 0.0),
            float(np.mean(dpr)),
            float(np.std(dpr, ddof=1) if len(rs) > 1 else 0.0),
            float(np.mean(pr_ratio)),
            float(np.std(pr_ratio, ddof=1) if len(rs) > 1 else 0.0),
        ])

    with open(by_eps_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epsilon",
            "n_trials",
            "Selectivity_end_mean",
            "Selectivity_end_std",
            "delta_PR_mean",
            "delta_PR_std",
            "ratio_PR_mean",
            "ratio_PR_std",
        ])
        w.writerows(agg_rows)

    # Plots
    try:
        import matplotlib.pyplot as plt

        # scatter delta_PR vs Selectivity_end
        eps_vals = np.array([r[0] for r in rows], dtype=float)
        sel_vals = np.array([r[6] for r in rows], dtype=float)
        dpr_vals = np.array([r[11] for r in rows], dtype=float)

        plt.figure()
        sc = plt.scatter(dpr_vals, sel_vals, c=eps_vals)
        plt.xlabel("delta_PR_TminusD")
        plt.ylabel("Selectivity_end")
        plt.title(f"A8.4: Localization vs Selectivity (kappa={kappa_join})")
        plt.colorbar(sc, label="epsilon")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A8_4_deltaPR_vs_selectivity.png"), dpi=160)
        plt.close()

        # delta_PR vs epsilon (mean ± std)
        eps_grid = np.array([r[0] for r in agg_rows], dtype=float)
        dpr_m = np.array([r[4] for r in agg_rows], dtype=float)
        dpr_s = np.array([r[5] for r in agg_rows], dtype=float)

        plt.figure()
        plt.errorbar(eps_grid, dpr_m, yerr=dpr_s, fmt="-o")
        plt.xlabel("epsilon")
        plt.ylabel("delta_PR_mean")
        plt.title(f"A8.4: Differential localization vs disorder (kappa={kappa_join})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A8_4_deltaPR_vs_epsilon.png"), dpi=160)
        plt.close()

    except Exception as e:
        print(f"[A8.4] Plotting skipped due to error: {e}")

    print("[A8.4] DONE")
    print(f"  Wrote: {trialwise_csv}")
    print(f"  Wrote: {by_eps_csv}")
    print(f"  PNGs in: {out_dir}")


if __name__ == "__main__":
    main()
