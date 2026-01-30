#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A8.5 — Architecture dependence (controlled in‑silico manipulations of the REAL connectome).

Goal
----
Test whether the disorder‑enhanced Selectivity (the epsilon~3 effect) depends on the
specific mushroom‑body wiring, rather than only on generic degree statistics.

We generate *derived* graphs from the REAL connectome (not synthetic replacements):
  1) original: no change
  2) rewired_type: shuffle destinations *within each (src_type, dst_type) edge block* (preserves
     edge‑type counts and per‑node outgoing counts per block; destroys specific wiring)
  3) lesion_KC_MBON: randomly drop a fraction of KC->MBON edges (a crude in‑silico "simplification")

For each variant we run a reduced GKSL endpoint benchmark (no full sweeps by default):
  - epsilons: 0,3,5 (default)
  - kappas: 0.001 and 1.0 (default)
  - n_trials: 5 (default)

Outputs
-------
  - A8_5_arch_trialwise.csv
  - A8_5_arch_summary.csv
  - A8_5_selectivity_vs_epsilon.png

Notes
-----
- This uses the same Hamiltonian core as A7/A3: H = -gamma*L_sym + diag(E)
- Energies use the A7 seeding scheme so results are comparable.
- This is heavier than A8.4. Start with defaults; increase n_trials when you are confident.

"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags, identity, kron
from scipy.sparse.linalg import expm_multiply


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _as_str_list(xs) -> List[str]:
    return [str(x) for x in xs]


def _vec_index(i: int, j: int, dim: int) -> int:
    return i + dim * j


def _normalize_outgoing_probs(G: nx.DiGraph, nodes: List[str], weight_attr: str) -> Dict[Tuple[str, str], float]:
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


def _build_liouvillian_gksl_two_sinks(
    node_list: List[str],
    H0_nodes: csr_matrix,
    target_nodes: List[str],
    distractor_nodes: List[str],
    gamma: float,
    kappa: float,
    eta_sink: float,
    energies: Optional[Dict[str, float]] = None,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int]]:
    """Copy of the A7 GKSL builder, but accepts precomputed H0_nodes = -gamma*L_sym."""

    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    sink_T = n
    sink_D = n + 1
    dim = n + 2

    targets = [str(x) for x in target_nodes if str(x) in idx]
    distractors = [str(x) for x in distractor_nodes if str(x) in idx]

    # Hamiltonian on node subspace
    H_nodes = H0_nodes
    if energies is not None:
        E = np.array([float(energies.get(nm, 0.0)) for nm in node_list], dtype=float)
        H_nodes = (H_nodes + diags(E, 0, shape=(n, n), dtype=float)).tocsr()

    # Embed into full dim x dim Hamiltonian (sinks decoupled)
    H = coo_matrix((dim, dim), dtype=float).tocsr()
    H[:n, :n] = H_nodes

    I = identity(dim, format="csr", dtype=complex)

    # Commutator: -i (I⊗H - H^T⊗I)
    H_c = H.astype(complex)
    L_comm = (-1j) * (kron(I, H_c, format="csr") - kron(H_c.transpose(), I, format="csr"))

    # Dephasing: -kappa on off-diagonals, 0 on diagonal.
    dim2 = dim * dim
    diag = np.zeros(dim2, dtype=float)
    if kappa != 0.0:
        diag[:] = -float(kappa)
        for i in range(dim):
            diag[i + dim * i] = 0.0
    L_deph = diags(diag, 0, shape=(dim2, dim2), format="csr", dtype=complex)

    def dissipator_for_L(op: csr_matrix) -> csr_matrix:
        Lc = op.astype(complex)
        A = (Lc.getH() @ Lc).tocsr()  # L†L
        term1 = kron(Lc.conjugate(), Lc, format="csr")
        term2 = 0.5 * (
            kron(identity(dim, format="csr", dtype=complex), A, format="csr")
            + kron(A.transpose(), identity(dim, format="csr", dtype=complex), format="csr")
        )
        return term1 - term2

    L_sink = csr_matrix((dim2, dim2), dtype=complex)
    for t in targets:
        t_idx = idx[t]
        op = coo_matrix(([np.sqrt(eta_sink)], ([sink_T], [t_idx])), shape=(dim, dim)).tocsr()
        L_sink = L_sink + dissipator_for_L(op)
    for d in distractors:
        d_idx = idx[d]
        op = coo_matrix(([np.sqrt(eta_sink)], ([sink_D], [d_idx])), shape=(dim, dim)).tocsr()
        L_sink = L_sink + dissipator_for_L(op)

    Lsuper = (L_comm + L_deph + L_sink).tocsr()
    vec0 = np.zeros(dim2, dtype=complex)

    return Lsuper, vec0, {"sink_T": sink_T, "sink_D": sink_D, **idx}


def _metrics_from_endpoints(pT: float, pD: float) -> Tuple[float, float]:
    sel = float(pT / (pD + 1e-12))
    cov = float(pT + pD)
    return sel, cov


def _rewire_destinations_by_edge_type(
    Gs: nx.DiGraph,
    node_type_attr: str,
    weight_attr: str,
    rng: np.random.RandomState,
    disallow_self_loops: bool = True,
) -> nx.DiGraph:
    """Shuffle destinations within each (src_type, dst_type) block. Sources and weights stay attached."""

    # Extract edges with types
    blocks: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = {}
    for u, v, data in Gs.edges(data=True):
        u = str(u); v = str(v)
        w = float(data.get(weight_attr, 1.0))
        if w <= 0:
            continue
        tu = str(Gs.nodes[u].get(node_type_attr, "NA"))
        tv = str(Gs.nodes[v].get(node_type_attr, "NA"))
        blocks.setdefault((tu, tv), []).append((u, v, w))

    Gnew = nx.DiGraph()
    # Copy nodes with attributes
    for n, attrs in Gs.nodes(data=True):
        Gnew.add_node(str(n), **attrs)

    for (tu, tv), edges in blocks.items():
        dests = [v for (_, v, _) in edges]
        rng.shuffle(dests)

        for (u, _old_v, w), new_v in zip(edges, dests):
            if disallow_self_loops and (new_v == u):
                # fallback: keep old destination
                new_v = _old_v
            if Gnew.has_edge(u, new_v):
                Gnew[u][new_v][weight_attr] = float(Gnew[u][new_v].get(weight_attr, 0.0)) + w
            else:
                Gnew.add_edge(u, new_v, **{weight_attr: w})

    return Gnew


def _lesion_edges_by_type(
    Gs: nx.DiGraph,
    node_type_attr: str,
    weight_attr: str,
    src_type: str,
    dst_type: str,
    drop_fraction: float,
    rng: np.random.RandomState,
) -> nx.DiGraph:
    """Randomly drop a fraction of edges in a specific (src_type -> dst_type) block."""

    drop_fraction = float(np.clip(drop_fraction, 0.0, 1.0))

    Gnew = nx.DiGraph()
    for n, attrs in Gs.nodes(data=True):
        Gnew.add_node(str(n), **attrs)

    for u, v, data in Gs.edges(data=True):
        u = str(u); v = str(v)
        w = float(data.get(weight_attr, 1.0))
        if w <= 0:
            continue
        tu = str(Gs.nodes[u].get(node_type_attr, "NA"))
        tv = str(Gs.nodes[v].get(node_type_attr, "NA"))
        if tu == src_type and tv == dst_type:
            if rng.rand() < drop_fraction:
                continue
        # keep edge
        if Gnew.has_edge(u, v):
            Gnew[u][v][weight_attr] = float(Gnew[u][v].get(weight_attr, 0.0)) + w
        else:
            Gnew.add_edge(u, v, **{weight_attr: w})

    return Gnew


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7_dir", required=True, help="Path to A7_outputs folder (contains A7_meta.json)")
    ap.add_argument("--out_dir", required=True, help="Output folder for A8.5")
    ap.add_argument("--epsilons", default="0,3,5", help="Comma list, e.g. '0,3,5'")
    ap.add_argument("--kappas", default="0.001,1.0", help="Comma list, e.g. '0.001,1.0'")
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--n_surrogates", type=int, default=3, help="How many independent rewired_type graphs to sample")
    ap.add_argument("--lesion_drop", type=float, default=0.5, help="Fraction of KC->MBON edges to drop in lesion variant")
    ap.add_argument("--graphml", default="", help="Override graph.graphml path")
    ap.add_argument("--config", default="", help="Override bench_config_*.json path")
    args = ap.parse_args()

    a7_dir = os.path.abspath(os.path.expanduser(args.a7_dir))
    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(args.out_dir)))

    meta_path = os.path.join(a7_dir, "A7_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"A7_meta.json not found: {meta_path}")
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
    node_type_attr = cfg.get("node_type_attr", "celltype")
    weight_attr = cfg.get("weight_attr", "weight")

    source_nodes = _as_str_list(cfg["source_nodes"])
    target_nodes = _as_str_list(cfg["target_nodes"])
    distractor_nodes = _as_str_list(cfg["distractor_nodes"])
    active_nodes = _as_str_list(cfg["active_nodes"])

    sim = cfg.get("sim_defaults", {})
    gamma = float(sim.get("gamma", 1.0))
    eta_sink = float(sim.get("eta_sink", 1.0))
    T_max = float(sim.get("T_max", 10.0))

    disorder = meta.get("disorder", {})
    base_seed = int(disorder.get("base_seed", 42))

    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    kappas = [float(x.strip()) for x in args.kappas.split(",") if x.strip()]
    n_trials = int(args.n_trials)

    # Load graph and active subgraph (A7 ordering)
    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    node_set = set(active_nodes)
    H_nodes = [str(n) for n in G.nodes() if str(n) in node_set]
    if len(H_nodes) == 0:
        raise SystemExit("Active subgraph is empty. Check active_nodes in bench_config.")

    Gs0 = G.subgraph(H_nodes).copy()

    # Prepare variants
    variants: List[Tuple[str, int, nx.DiGraph]] = []

    variants.append(("original", 0, Gs0))

    # rewired_type: multiple surrogates
    n_sur = max(1, int(args.n_surrogates))
    for sid in range(1, n_sur + 1):
        rng = np.random.RandomState(base_seed + 9000 + sid)
        Gs_rew = _rewire_destinations_by_edge_type(Gs0, node_type_attr, weight_attr, rng)
        variants.append(("rewired_type", sid, Gs_rew))

    # lesion: multiple surrogates too (different random drops)
    for sid in range(1, n_sur + 1):
        rng = np.random.RandomState(base_seed + 12000 + sid)
        Gs_les = _lesion_edges_by_type(
            Gs0, node_type_attr, weight_attr,
            src_type="KCs", dst_type="MBONs",
            drop_fraction=float(args.lesion_drop),
            rng=rng,
        )
        variants.append(("lesion_KC_MBON", sid, Gs_les))

    trialwise_path = os.path.join(out_dir, "A8_5_arch_trialwise.csv")
    summary_path = os.path.join(out_dir, "A8_5_arch_summary.csv")

    # Run
    trial_rows = []

    for variant_name, sid, Gs in variants:
        # Build p_uv and base Hamiltonian once per variant
        p_uv = _normalize_outgoing_probs(Gs, H_nodes, weight_attr)
        L_sym = _symmetrized_laplacian_from_directed_probs(H_nodes, p_uv)
        H0_nodes = (-gamma) * L_sym

        for eps in epsilons:
            for trial in range(n_trials):
                energy_seed = int(base_seed + 1000 * trial + int(round(eps * 100)))
                rngE = np.random.RandomState(energy_seed)
                energies = {node: float(rngE.uniform(-eps, eps)) for node in H_nodes}

                for kappa in kappas:
                    Lsuper, vec0, idx_map = _build_liouvillian_gksl_two_sinks(
                        node_list=H_nodes,
                        H0_nodes=H0_nodes,
                        target_nodes=target_nodes,
                        distractor_nodes=distractor_nodes,
                        gamma=gamma,
                        kappa=float(kappa),
                        eta_sink=eta_sink,
                        energies=energies,
                    )
                    dim = len(H_nodes) + 2
                    sinkT = idx_map["sink_T"]
                    sinkD = idx_map["sink_D"]

                    # init rho0 as diagonal mixture over sources
                    vec0[:] = 0.0
                    src = [s for s in source_nodes if s in idx_map]
                    if len(src) == 0:
                        raise SystemExit("No source nodes found in active subgraph for this config.")
                    p0 = 1.0 / len(src)
                    for s in src:
                        i = idx_map[s]
                        vec0[_vec_index(i, i, dim)] = p0

                    vec_end = expm_multiply(Lsuper * T_max, vec0)
                    pT = float(np.real(vec_end[_vec_index(sinkT, sinkT, dim)]))
                    pD = float(np.real(vec_end[_vec_index(sinkD, sinkD, dim)]))
                    sel, cov = _metrics_from_endpoints(pT, pD)

                    trial_rows.append([
                        variant_name,
                        sid,
                        eps,
                        trial,
                        energy_seed,
                        float(kappa),
                        pT,
                        pD,
                        sel,
                        cov,
                        int(Gs.number_of_edges()),
                    ])

    # Write trialwise
    with open(trialwise_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant",
            "surrogate_id",
            "epsilon",
            "trial",
            "energy_seed",
            "kappa",
            "P_sink_T_end",
            "P_sink_D_end",
            "Selectivity_end",
            "coverage_end",
            "N_edges",
        ])
        w.writerows(trial_rows)

    # Aggregate summary
    # key: (variant, sid, epsilon, kappa)
    buckets: Dict[Tuple[str, int, float, float], List[List[float]]] = {}
    for r in trial_rows:
        key = (str(r[0]), int(r[1]), float(r[2]), float(r[5]))
        buckets.setdefault(key, []).append(r)

    sum_rows = []
    for (variant, sid, eps, kappa), rs in sorted(
        buckets.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]),
    ):
        arr = np.array([r[2:] for r in rs], dtype=float)
        # columns now: eps=0, trial=1, seed=2, kappa=3, pT=4, pD=5, sel=6, cov=7, edges=8
        pT = arr[:, 4]
        pD = arr[:, 5]
        sel = arr[:, 6]
        cov = arr[:, 7]
        sum_rows.append([
            variant,
            sid,
            eps,
            kappa,
            len(rs),
            float(np.mean(pT)),
            float(np.mean(pD)),
            float(np.mean(sel)),
            float(np.std(sel, ddof=1) if len(rs) > 1 else 0.0),
            float(np.mean(cov)),
        ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "variant",
            "surrogate_id",
            "epsilon",
            "kappa",
            "n_trials",
            "P_sink_T_end_mean",
            "P_sink_D_end_mean",
            "Selectivity_end_mean",
            "Selectivity_end_std",
            "coverage_end_mean",
        ])
        w.writerows(sum_rows)

    # Quick plot: selectivity vs epsilon at the first kappa in list (usually coherent 0.001)
    try:
        import matplotlib.pyplot as plt

        kappa0 = kappas[0]

        # aggregate over surrogates for each variant
        # key (variant, epsilon) -> list of means (over trials) for each surrogate
        agg2: Dict[Tuple[str, float], List[float]] = {}
        for r in sum_rows:
            variant = r[0]
            eps = float(r[2])
            kappa = float(r[3])
            sel_mean = float(r[7])
            if abs(kappa - kappa0) > 1e-12:
                continue
            agg2.setdefault((variant, eps), []).append(sel_mean)

        plt.figure()
        for variant in sorted(set(v for (v, _e) in agg2.keys())):
            xs = []
            ys = []
            for eps in sorted(set(e for (_v, e) in agg2.keys() if _v == variant)):
                xs.append(eps)
                ys.append(float(np.mean(agg2[(variant, eps)])))
            plt.plot(xs, ys, marker="o", label=variant)

        plt.xlabel("epsilon")
        plt.ylabel("Selectivity_end_mean")
        plt.title(f"A8.5: Architecture dependence (kappa={kappa0})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "A8_5_selectivity_vs_epsilon.png"), dpi=160)
        plt.close()

    except Exception as e:
        print(f"[A8.5] Plotting skipped due to error: {e}")

    print("[A8.5] DONE")
    print(f"  Trialwise: {trialwise_path}")
    print(f"  Summary:   {summary_path}")
    print(f"  PNGs in:   {out_dir}")


if __name__ == "__main__":
    main()
