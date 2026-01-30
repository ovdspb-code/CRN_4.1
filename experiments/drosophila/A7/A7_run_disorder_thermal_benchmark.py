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


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _vec_index(i: int, j: int, dim: int) -> int:
    return i + dim * j


def _normalize_outgoing_probs(
    G: nx.DiGraph,
    nodes: List[str],
    weight_attr: str,
) -> Dict[Tuple[str, str], float]:
    """
    Directed row-normalized transition probabilities:
      p(u->v) = w(u->v) / sum_out_w(u)
    restricted to the provided node list.

    Returns dict keyed by (u,v) as strings.
    """
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


def _symmetrized_laplacian_from_directed_probs(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
) -> csr_matrix:
    """
    Symmetric coupling:
        a_ij = 0.5 * (p(i->j) + p(j->i))  implemented by accumulating directed p/2 for each directed edge
    L = D - A (symmetric).
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
        a = 0.5 * float(p)
        key = (min(iu, iv), max(iu, iv))
        pairs[key] = pairs.get(key, 0.0) + a

    rows = []
    cols = []
    vals = []
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


def _build_classical_generator_with_two_sinks(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
    source_nodes: List[str],
    target_nodes: List[str],
    distractor_nodes: List[str],
    gamma: float,
    eta_sink: float,
    energies: Optional[Dict[str, float]] = None,
    T_env: Optional[float] = None,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int]]:
    """
    Continuous-time Markov chain generator Q such that dp/dt = Q @ p
    with columns summing to 0.

    If energies and T_env are provided, applies a Metropolis-like uphill suppression:
      accept(u->v) = exp(-(E_v - E_u)/T_env) if E_v > E_u else 1.
    """

    node_list = [str(x) for x in nodes]
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    sink_T = n
    sink_D = n + 1
    dim = n + 2

    targets = set(map(str, target_nodes))
    distractors = set(map(str, distractor_nodes))

    do_thermal = (energies is not None) and (T_env is not None) and (T_env > 0)

    rows = []
    cols = []
    vals = []
    out_rate = np.zeros(n, dtype=float)

    for (u, v), p in p_uv.items():
        u = str(u); v = str(v)
        if u not in idx or v not in idx:
            continue
        if p <= 0:
            continue
        j = idx[u]
        i = idx[v]
        r = gamma * float(p)
        if do_thermal:
            Eu = float(energies.get(u, 0.0))
            Ev = float(energies.get(v, 0.0))
            dE = Ev - Eu
            if dE > 0:
                r = r * float(np.exp(-dE / float(T_env)))
        if r <= 0:
            continue
        rows.append(i); cols.append(j); vals.append(r)
        out_rate[j] += r

    # Sink couplings
    for u in node_list:
        j = idx[u]
        if u in targets:
            rows.append(sink_T); cols.append(j); vals.append(float(eta_sink))
            out_rate[j] += float(eta_sink)
        if u in distractors:
            rows.append(sink_D); cols.append(j); vals.append(float(eta_sink))
            out_rate[j] += float(eta_sink)

    # Diagonal (negative outflow)
    for u in node_list:
        j = idx[u]
        if out_rate[j] != 0.0:
            rows.append(j); cols.append(j); vals.append(-out_rate[j])

    Q = coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()

    # Initial distribution
    p0 = np.zeros(dim, dtype=float)
    src = [str(x) for x in source_nodes if str(x) in idx]
    if len(src) == 0:
        raise RuntimeError("No source nodes found inside active subgraph; check config.active_nodes / source_nodes.")
    p0[[idx[s] for s in src]] = 1.0 / len(src)

    return Q, p0, {"sink_T": sink_T, "sink_D": sink_D, **idx}


def _build_liouvillian_gksl_two_sinks_with_energies(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
    target_nodes: List[str],
    distractor_nodes: List[str],
    gamma: float,
    kappa: float,
    eta_sink: float,
    energies: Optional[Dict[str, float]] = None,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int]]:
    """
    GKSL/Lindblad Liouvillian acting on vec(rho) with 2 sink states.

    H = -gamma * L_sym + diag(E)   on node subspace only (sinks are decoupled)
    """
    node_list = [str(x) for x in nodes]
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    sink_T = n
    sink_D = n + 1
    dim = n + 2

    targets = [str(x) for x in target_nodes if str(x) in idx]
    distractors = [str(x) for x in distractor_nodes if str(x) in idx]

    # Hamiltonian on node subspace
    L_sym = _symmetrized_laplacian_from_directed_probs(node_list, p_uv)
    H_nodes = (-gamma) * L_sym  # real symmetric

    # Add diagonal disorder if provided
    if energies is not None:
        E = np.array([float(energies.get(nm, 0.0)) for nm in node_list], dtype=float)
        H_nodes = (H_nodes + diags(E, 0, shape=(n, n), dtype=float)).tocsr()

    # Embed into full dim x dim Hamiltonian (sinks decoupled)
    H = coo_matrix((dim, dim), dtype=float).tocsr()
    H[:n, :n] = H_nodes

    I = identity(dim, format="csr", dtype=complex)

    # Commutator superoperator: -i (I ⊗ H - H^T ⊗ I)
    H_c = H.astype(complex)
    L_comm = (-1j) * (kron(I, H_c, format="csr") - kron(H_c.transpose(), I, format="csr"))

    # Dephasing superoperator: -kappa on off-diagonals, 0 on diagonal.
    dim2 = dim * dim
    diag = np.zeros(dim2, dtype=float)
    if kappa != 0.0:
        diag[:] = -float(kappa)
        for i in range(dim):
            diag[i + dim * i] = 0.0
    L_deph = diags(diag, 0, shape=(dim2, dim2), format="csr", dtype=complex)

    def dissipator_for_L(op: csr_matrix) -> csr_matrix:
        # D(ρ) = L ρ L† - 1/2 {L†L, ρ}
        Lc = op.astype(complex)
        A = (Lc.getH() @ Lc).tocsr()  # L†L
        term1 = kron(Lc.conjugate(), Lc, format="csr")  # (L*) ⊗ L
        term2 = 0.5 * (kron(identity(dim, format="csr", dtype=complex), A, format="csr") +
                       kron(A.transpose(), identity(dim, format="csr", dtype=complex), format="csr"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to bench_config_*.json (from A2).")
    ap.add_argument("--out_dir", required=True, help="Output folder for A7 artifacts.")
    ap.add_argument("--epsilons", default="0,1,3,5", help="Comma-separated epsilon grid for diagonal disorder.")
    ap.add_argument("--n_trials", type=int, default=5, help="Number of disorder realizations per epsilon.")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed (defaults to config.seed).")
    ap.add_argument("--kappa_grid", default=None, help="Override kappa grid, comma-separated.")
    ap.add_argument("--T_envs", default="0.1,1.0", help="Comma-separated list of temperatures for CRW_thermal.")
    ap.add_argument("--no_crw_topo", action="store_true", help="Skip CRW (topological) baseline.")
    ap.add_argument("--no_gksl", action="store_true", help="Skip GKSL sweep (for debugging).")
    ap.add_argument("--no_crw_thermal", action="store_true", help="Skip CRW_thermal baselines.")
    args = ap.parse_args()

    cfg = _read_json(os.path.abspath(os.path.expanduser(args.config)))
    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(args.out_dir)))

    mode = str(cfg.get("mode", "unknown"))
    graphml = str(cfg["graphml_source"])
    weight_attr = str(cfg.get("weight_attr", "weight"))

    source_nodes = [str(x) for x in cfg["source_nodes"]]
    target_nodes = [str(x) for x in cfg["target_nodes"]]
    distractor_nodes = [str(x) for x in cfg["distractor_nodes"]]
    active_nodes = [str(x) for x in cfg["active_nodes"]]

    sim = cfg.get("sim_defaults", {})
    gamma = float(sim.get("gamma", 1.0))
    eta_sink = float(sim.get("eta_sink", 1.0))
    T_max = float(sim.get("T_max", 10.0))
    kappa_grid = sim.get("kappa_grid", [0.0, 0.1, 1.0, 10.0])

    if args.kappa_grid is not None:
        kappa_grid = [float(x) for x in args.kappa_grid.split(",") if x.strip()]

    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]
    T_envs = [float(x) for x in args.T_envs.split(",") if x.strip()]

    base_seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 0))

    # Load graph
    G = nx.read_graphml(os.path.abspath(os.path.expanduser(graphml)))
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    node_set = set(map(str, active_nodes))
    H_nodes = [str(n) for n in G.nodes() if str(n) in node_set]
    Gs = G.subgraph(H_nodes).copy()

    # Directed probs
    p_uv = _normalize_outgoing_probs(Gs, H_nodes, weight_attr)

    # Prepare outputs
    gksl_csv = os.path.join(out_dir, "A7_gksl_sweep.csv")
    base_csv = os.path.join(out_dir, "A7_baselines.csv")
    seed_csv = os.path.join(out_dir, "A7_energy_seeds.csv")
    meta_json = os.path.join(out_dir, "A7_meta.json")

    with open(gksl_csv, "w", newline="", encoding="utf-8") as f_gksl, \
         open(base_csv, "w", newline="", encoding="utf-8") as f_base, \
         open(seed_csv, "w", newline="", encoding="utf-8") as f_seed:

        wg = csv.writer(f_gksl)
        wb = csv.writer(f_base)
        ws = csv.writer(f_seed)

        wg.writerow(["mode", "epsilon", "trial", "kappa", "P_sink_T_end", "P_sink_D_end", "Selectivity_end", "coverage_end"])
        wb.writerow(["mode", "epsilon", "trial", "model", "T_env", "P_sink_T_end", "P_sink_D_end", "Selectivity_end", "coverage_end"])
        ws.writerow(["mode", "epsilon", "trial", "energy_seed"])

        for eps in epsilons:
            for trial in range(int(args.n_trials)):
                energy_seed = int(base_seed + 1000 * trial + int(round(eps * 100)))
                ws.writerow([mode, eps, trial, energy_seed])

                rng = np.random.RandomState(energy_seed)
                energies = {node: float(rng.uniform(-eps, eps)) for node in H_nodes}

                # Baselines
                if not args.no_crw_topo:
                    Q, p0, idx = _build_classical_generator_with_two_sinks(
                        nodes=H_nodes, p_uv=p_uv,
                        source_nodes=source_nodes, target_nodes=target_nodes, distractor_nodes=distractor_nodes,
                        gamma=gamma, eta_sink=eta_sink,
                        energies=None, T_env=None
                    )
                    p_end = expm_multiply(Q * T_max, p0)
                    pT = float(p_end[idx["sink_T"]]); pD = float(p_end[idx["sink_D"]])
                    sel, cov = _metrics_from_endpoints(pT, pD)
                    wb.writerow([mode, eps, trial, "CRW", "", pT, pD, sel, cov])

                if (not args.no_crw_thermal) and (eps > 0 or True):
                    for T_env in T_envs:
                        Q, p0, idx = _build_classical_generator_with_two_sinks(
                            nodes=H_nodes, p_uv=p_uv,
                            source_nodes=source_nodes, target_nodes=target_nodes, distractor_nodes=distractor_nodes,
                            gamma=gamma, eta_sink=eta_sink,
                            energies=energies, T_env=T_env
                        )
                        p_end = expm_multiply(Q * T_max, p0)
                        pT = float(p_end[idx["sink_T"]]); pD = float(p_end[idx["sink_D"]])
                        sel, cov = _metrics_from_endpoints(pT, pD)
                        wb.writerow([mode, eps, trial, "CRW_thermal", T_env, pT, pD, sel, cov])

                # GKSL sweep
                if not args.no_gksl:
                    for kappa in kappa_grid:
                        kappa = float(kappa)
                        Lsuper, vec0, idx_q = _build_liouvillian_gksl_two_sinks_with_energies(
                            nodes=H_nodes, p_uv=p_uv,
                            target_nodes=target_nodes, distractor_nodes=distractor_nodes,
                            gamma=gamma, kappa=kappa, eta_sink=eta_sink,
                            energies=energies
                        )
                        dim = len(H_nodes) + 2
                        sinkT_i = idx_q["sink_T"]; sinkD_i = idx_q["sink_D"]

                        # init diag mixture over sources
                        vec0[:] = 0.0
                        src = [s for s in source_nodes if s in idx_q]
                        if len(src) == 0:
                            raise RuntimeError("No source nodes found inside active subgraph for GKSL build.")
                        p_init = 1.0 / len(src)
                        for s in src:
                            i = idx_q[s]
                            vec0[_vec_index(i, i, dim)] = p_init

                        vec_end = expm_multiply(Lsuper * T_max, vec0)
                        pT = float(np.real(vec_end[_vec_index(sinkT_i, sinkT_i, dim)]))
                        pD = float(np.real(vec_end[_vec_index(sinkD_i, sinkD_i, dim)]))
                        sel, cov = _metrics_from_endpoints(pT, pD)
                        wg.writerow([mode, eps, trial, kappa, pT, pD, sel, cov])

    meta = {
        "mode": mode,
        "config": os.path.abspath(os.path.expanduser(args.config)),
        "graphml": os.path.abspath(os.path.expanduser(graphml)),
        "active_subgraph": {"N_nodes": len(H_nodes), "N_edges": int(Gs.number_of_edges())},
        "sim": {"gamma": gamma, "eta_sink": eta_sink, "T_max": T_max, "kappa_grid": list(map(float, kappa_grid))},
        "disorder": {"kind": "diagonal_uniform", "epsilons": epsilons, "n_trials": int(args.n_trials), "base_seed": base_seed},
        "thermal": {"T_envs": T_envs, "metropolis": True},
    }
    _write_json(meta_json, meta)

    print("[A7] DONE")
    print(f"  - {gksl_csv}")
    print(f"  - {base_csv}")
    print(f"  - {seed_csv}")
    print(f"  - {meta_json}")


if __name__ == "__main__":
    main()
