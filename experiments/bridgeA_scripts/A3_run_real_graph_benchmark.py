
import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags, identity, kron
from scipy.sparse.linalg import expm_multiply


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _normalize_outgoing_probs(
    G: nx.DiGraph,
    nodes: List[str],
    weight_attr: str,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Returns:
      p[(u,v)] = w(u->v)/sum_out_w(u)  (0 if u has 0 outgoing weight within nodes)
      sum_out_w[u] = sum of outgoing weights within nodes
    """
    node_set = set(nodes)

    sum_out = {u: 0.0 for u in nodes}
    # collect weights (sum multi-edges if any)
    w_uv: Dict[Tuple[str, str], float] = {}

    for u, v, data in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue
        w = float(data.get(weight_attr, 1.0))
        if w <= 0:
            continue
        key = (str(u), str(v))
        w_uv[key] = w_uv.get(key, 0.0) + w
        sum_out[str(u)] += w

    p = {}
    for (u, v), w in w_uv.items():
        denom = sum_out[u]
        p[(u, v)] = (w / denom) if denom > 0 else 0.0

    return p, sum_out


def _build_classical_generator_with_two_sinks(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
    source_nodes: List[str],
    target_nodes: List[str],
    distractor_nodes: List[str],
    gamma: float,
    eta_sink: float,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int]]:
    """
    Continuous-time Markov chain generator Q such that dp/dt = Q @ p
    with columns summing to 0 (for probability conservation).

    We add two absorbing sinks:
      sink_T (correct)
      sink_D (distractor)

    Absorption rates:
      node in target_nodes    -> sink_T at rate eta_sink
      node in distractor_nodes-> sink_D at rate eta_sink
    """
    node_list = [str(x) for x in nodes]
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    sink_T = n
    sink_D = n + 1
    dim = n + 2

    targets = set(map(str, target_nodes))
    distractors = set(map(str, distractor_nodes))

    rows = []
    cols = []
    vals = []

    # Internal transitions (from u to v)
    out_rate = np.zeros(n, dtype=float)

    for (u, v), p in p_uv.items():
        if u not in idx or v not in idx:
            continue
        if p <= 0:
            continue
        j = idx[u]  # source column
        i = idx[v]  # destination row
        r = gamma * p
        rows.append(i)
        cols.append(j)
        vals.append(r)
        out_rate[j] += r

    # Sink couplings
    for u in node_list:
        j = idx[u]
        if u in targets:
            rows.append(sink_T)
            cols.append(j)
            vals.append(eta_sink)
            out_rate[j] += eta_sink
        if u in distractors:
            rows.append(sink_D)
            cols.append(j)
            vals.append(eta_sink)
            out_rate[j] += eta_sink

    # Diagonal (negative outflow)
    for u in node_list:
        j = idx[u]
        if out_rate[j] != 0.0:
            rows.append(j)
            cols.append(j)
            vals.append(-out_rate[j])

    Q = coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()

    # Initial distribution (column vector)
    p0 = np.zeros(dim, dtype=float)
    src = [str(x) for x in source_nodes if str(x) in idx]
    if len(src) == 0:
        raise RuntimeError("No source nodes found inside active subgraph; check config.active_nodes / source_nodes.")
    p0[[idx[s] for s in src]] = 1.0 / len(src)

    return Q, p0, {"sink_T": sink_T, "sink_D": sink_D, **idx}


def _symmetrized_laplacian_from_directed_probs(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
) -> csr_matrix:
    """
    Build a *symmetric* weighted Laplacian from directed row-normalized probabilities.

    We define symmetric coupling:
        a_ij = 0.5 * (p(i->j) + p(j->i))
    Then Laplacian L = D - A (symmetric, PSD).

    This is a pragmatic choice: it preserves feedforward edges (if only i->j exists, a_ij = p(i->j)/2)
    while keeping weights O(1), preventing gigantic time-scale issues from raw synapse counts.
    """
    node_list = [str(x) for x in nodes]
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    # collect symmetric edges
    # We only build i<j entries then mirror.
    pairs = {}
    for (u, v), p in p_uv.items():
        if p <= 0:
            continue
        iu = idx.get(str(u))
        iv = idx.get(str(v))
        if iu is None or iv is None or iu == iv:
            continue
        a = 0.5 * p
        key = (min(iu, iv), max(iu, iv))
        pairs[key] = pairs.get(key, 0.0) + a  # accumulate

    # add reverse direction contributions
    # This is already handled because p_uv contains directed; both directions will add.

    # Build symmetric adjacency
    rows = []
    cols = []
    vals = []
    deg = np.zeros(n, dtype=float)

    for (i, j), aij in pairs.items():
        if aij <= 0:
            continue
        # symmetric entries
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([aij, aij])
        deg[i] += aij
        deg[j] += aij

    A = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    L = diags(deg, 0, shape=(n, n), dtype=float) - A
    return L


def _build_liouvillian_gksl_two_sinks(
    nodes: List[str],
    p_uv: Dict[Tuple[str, str], float],
    target_nodes: List[str],
    distractor_nodes: List[str],
    gamma: float,
    kappa: float,
    eta_sink: float,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int]]:
    """
    GKSL/Lindblad Liouvillian acting on vec(rho) with 2 sink states.

    Hilbert space dimension:
      dim = N_nodes + 2 (sink_T, sink_D)

    Dynamics:
      dρ/dt = -i[H,ρ] + D_deph(κ) + Σ_t D_sink_t(η) + Σ_d D_sink_d(η)

    - H = -γ * L_sym where L_sym is symmetric Laplacian built from directed probabilities.
    - Dephasing: standard local dephasing on node subspace only.
    - Sink jumps: L_t = sqrt(η) |sink_T><t| for targets, and similarly for distractors.

    Returns:
      Lsuper (csr, complex),
      vec_rho0 (complex),
      index map (node->i plus sink indices).
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

    # Embed into full dim x dim Hamiltonian (sinks decoupled)
    H = coo_matrix((dim, dim), dtype=float).tocsr()
    H[:n, :n] = H_nodes

    I = identity(dim, format="csr", dtype=complex)

    # Commutator superoperator: -i (I ⊗ H - H^T ⊗ I)
    H_c = H.astype(complex)
    L_comm = (-1j) * (kron(I, H_c, format="csr") - kron(H_c.transpose(), I, format="csr"))

    # Dephasing superoperator: diagonal with 0 on diagonal positions, -kappa on off-diagonals
    # Apply only on node subspace; sink-sink and sink-node coherences are also dephased? We set:
    # - dephase only among "physical" nodes, but also kill coherences involving node indices.
    # Pragmatic: dephase all off-diagonals except sink-sink (which is irrelevant). This is conservative.
    dim2 = dim * dim
    diag = np.zeros(dim2, dtype=float)

    # mark diagonal positions => 0; everything else => -kappa (except we optionally keep sink-sink diagonal untouched)
    if kappa != 0.0:
        diag[:] = -kappa
        # set true diagonal entries to 0
        for i in range(dim):
            diag[i + dim * i] = 0.0
        # optionally: do not dephase coherences purely within the two sinks (they are decoupled anyway)
        # (off-diagonal sink coherences correspond to i=sink_T,j=sink_D and vice versa)
        # We'll keep them dephased as well (does not matter for populations).
    L_deph = diags(diag, 0, shape=(dim2, dim2), format="csr", dtype=complex)

    # Sink dissipators
    def dissipator_for_L(op: csr_matrix) -> csr_matrix:
        # D(ρ) = L ρ L† - 1/2 {L†L, ρ}
        Lc = op.astype(complex)
        A = (Lc.getH() @ Lc).tocsr()  # L†L
        term1 = kron(Lc.conjugate(), Lc, format="csr")  # (L*) ⊗ L
        term2 = 0.5 * (kron(identity(dim, format="csr", dtype=complex), A, format="csr") +
                       kron(A.transpose(), identity(dim, format="csr", dtype=complex), format="csr"))
        return term1 - term2

    L_sink = csr_matrix((dim2, dim2), dtype=complex)
    # target sinks
    for t in targets:
        t_idx = idx[t]
        # L has one nonzero: (sink_T, t_idx)
        op = coo_matrix(([np.sqrt(eta_sink)], ([sink_T], [t_idx])), shape=(dim, dim)).tocsr()
        L_sink = L_sink + dissipator_for_L(op)
    for d in distractors:
        d_idx = idx[d]
        op = coo_matrix(([np.sqrt(eta_sink)], ([sink_D], [d_idx])), shape=(dim, dim)).tocsr()
        L_sink = L_sink + dissipator_for_L(op)

    Lsuper = (L_comm + L_deph + L_sink).tocsr()

    # Initial rho0 = diagonal mixture over sources will be set by caller; here just prepare zeros
    vec0 = np.zeros(dim2, dtype=complex)

    return Lsuper, vec0, {"sink_T": sink_T, "sink_D": sink_D, **idx}


def _vec_index(i: int, j: int, dim: int) -> int:
    # column-major: index = i + dim*j
    return i + dim * j


def run_benchmark(config_path: str, out_dir: str, traj_points: int = 51) -> None:
    cfg = _read_json(config_path)
    mode = cfg.get("mode", "unknown")
    graphml = cfg["graphml_source"]
    node_type_attr = cfg.get("node_type_attr", "celltype")
    weight_attr = cfg.get("weight_attr", "weight")

    source_nodes = [str(x) for x in cfg["source_nodes"]]
    target_nodes = [str(x) for x in cfg["target_nodes"]]
    distractor_nodes = [str(x) for x in cfg["distractor_nodes"]]
    active_nodes = [str(x) for x in cfg["active_nodes"]]

    sim = cfg.get("sim_defaults", {})
    gamma = float(sim.get("gamma", 1.0))
    eta_sink = float(sim.get("eta_sink", 1.0))
    kappa_grid = sim.get("kappa_grid", [0.0, 0.1, 1.0, 10.0])
    T_max = float(sim.get("T_max", 10.0))

    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(out_dir)))

    # Load graph
    G = nx.read_graphml(os.path.abspath(os.path.expanduser(graphml)))
    # Ensure directed
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # Active subgraph
    node_set = set(map(str, active_nodes))
    H_nodes = [str(n) for n in G.nodes() if str(n) in node_set]
    Gs = G.subgraph(H_nodes).copy()

    # Compute normalized directed probs
    p_uv, sum_out = _normalize_outgoing_probs(Gs, H_nodes, weight_attr)

    # CLASSICAL baseline (CRW with 2 sinks)
    Q, p0, idx_map = _build_classical_generator_with_two_sinks(
        nodes=H_nodes,
        p_uv=p_uv,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        distractor_nodes=distractor_nodes,
        gamma=gamma,
        eta_sink=eta_sink,
    )
    sinkT = idx_map["sink_T"]
    sinkD = idx_map["sink_D"]

    # Time grid for classical (cheap)
    num_steps = max(2, int(np.floor(T_max / 0.05)) + 1)  # fixed small dt for cheap classical traces
    p_t = expm_multiply(Q, p0, start=0.0, stop=T_max, num=num_steps, endpoint=True)
    t_grid = np.linspace(0.0, T_max, num_steps)
    crw_T = np.array([pt[sinkT] for pt in p_t], dtype=float)
    crw_D = np.array([pt[sinkD] for pt in p_t], dtype=float)

    crw_end_T = float(crw_T[-1])
    crw_end_D = float(crw_D[-1])
    crw_sel = float(crw_end_T / (crw_end_D + 1e-12))

    _write_csv(
        os.path.join(out_dir, "A3_timeseries_CRW.csv"),
        ["t", "P_sink_T", "P_sink_D"],
        [[float(t), float(a), float(b)] for t, a, b in zip(t_grid, crw_T, crw_D)],
    )

    # GKSL sweeps (compute only ENDPOINT per kappa; optional one trajectory for best kappa)
    rows = []
    best = None  # (metric, kappa, P_T, P_D)
    for kappa in kappa_grid:
        kappa = float(kappa)
        Lsuper, vec0, idx_q = _build_liouvillian_gksl_two_sinks(
            nodes=H_nodes,
            p_uv=p_uv,
            target_nodes=target_nodes,
            distractor_nodes=distractor_nodes,
            gamma=gamma,
            kappa=kappa,
            eta_sink=eta_sink,
        )
        dim = len(H_nodes) + 2
        sinkT_i = idx_q["sink_T"]
        sinkD_i = idx_q["sink_D"]

        # init diag mixture over sources
        vec0[:] = 0.0
        src = [s for s in source_nodes if s in idx_q]
        if len(src) == 0:
            raise RuntimeError("No source nodes found inside active subgraph for GKSL build.")
        p_init = 1.0 / len(src)
        for s in src:
            i = idx_q[s]
            vec0[_vec_index(i, i, dim)] = p_init

        # End state only
        vec_end = expm_multiply(Lsuper * T_max, vec0)

        P_T_end = float(np.real(vec_end[_vec_index(sinkT_i, sinkT_i, dim)]))
        P_D_end = float(np.real(vec_end[_vec_index(sinkD_i, sinkD_i, dim)]))
        sel = float(P_T_end / (P_D_end + 1e-12))

        rows.append([mode, kappa, P_T_end, P_D_end, sel])

        metric = sel
        cand = (metric, kappa, P_T_end, P_D_end)
        if best is None:
            best = cand
        else:
            # maximize selectivity, tie-break by higher P_T_end
            if cand[0] > best[0] + 1e-12 or (abs(cand[0] - best[0]) <= 1e-12 and cand[2] > best[2]):
                best = cand

    _write_csv(
        os.path.join(out_dir, "A3_kappa_sweep_GKSL.csv"),
        ["mode", "kappa", "P_sink_T_end", "P_sink_D_end", "Selectivity_end"],
        rows,
    )

    # Optional: trajectory for best kappa (coarse)
    best_kappa = float(best[1]) if best is not None else float(kappa_grid[0])
    Lbest, vec0, idx_q = _build_liouvillian_gksl_two_sinks(
        nodes=H_nodes,
        p_uv=p_uv,
        target_nodes=target_nodes,
        distractor_nodes=distractor_nodes,
        gamma=gamma,
        kappa=best_kappa,
        eta_sink=eta_sink,
    )
    dim = len(H_nodes) + 2
    sinkT_i = idx_q["sink_T"]
    sinkD_i = idx_q["sink_D"]

    vec0[:] = 0.0
    src = [s for s in source_nodes if s in idx_q]
    p_init = 1.0 / len(src)
    for s in src:
        i = idx_q[s]
        vec0[_vec_index(i, i, dim)] = p_init

    n_traj = int(traj_points)
    n_traj = max(11, min(n_traj, 101))  # keep it bounded to control memory
    # NOTE: this will allocate (n_traj x dim^2) complex array; bounded by n_traj <= 101.
    traj = expm_multiply(Lbest, vec0, start=0.0, stop=T_max, num=n_traj, endpoint=True)
    t2 = np.linspace(0.0, T_max, n_traj)
    gksl_T = np.array([np.real(v[_vec_index(sinkT_i, sinkT_i, dim)]) for v in traj], dtype=float)
    gksl_D = np.array([np.real(v[_vec_index(sinkD_i, sinkD_i, dim)]) for v in traj], dtype=float)

    _write_csv(
        os.path.join(out_dir, "A3_timeseries_GKSL_best.csv"),
        ["t", "kappa_best", "P_sink_T", "P_sink_D"],
        [[float(t), best_kappa, float(a), float(b)] for t, a, b in zip(t2, gksl_T, gksl_D)],
    )

    # Summary
    summary = {
        "timestamp": _now_iso(),
        "mode": mode,
        "config_path": os.path.abspath(config_path),
        "graphml": os.path.abspath(os.path.expanduser(graphml)),
        "active_subgraph": {
            "N_nodes": len(H_nodes),
            "N_edges": int(Gs.number_of_edges()),
        },
        "classical_CRW": {
            "P_sink_T_end": crw_end_T,
            "P_sink_D_end": crw_end_D,
            "Selectivity_end": crw_sel,
        },
        "gksl_GKSL": {
            "kappa_grid": list(map(float, kappa_grid)),
            "best_by_Selectivity_end": {
                "kappa": best_kappa,
                "P_sink_T_end": float(best[2]),
                "P_sink_D_end": float(best[3]),
                "Selectivity_end": float(best[0]),
            },
        },
        "sim": {
            "gamma": gamma,
            "eta_sink": eta_sink,
            "T_max": T_max,
            "traj_points_best": n_traj,
        },
        "runtime_env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(os.path.join(out_dir, "A3_summary.json"), summary)

    print(f"[A3:{mode}] DONE. Outputs in: {out_dir}")
    print("  - A3_kappa_sweep_GKSL.csv")
    print("  - A3_timeseries_CRW.csv")
    print("  - A3_timeseries_GKSL_best.csv")
    print("  - A3_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to bench_config_*.json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--traj_points", type=int, default=51, help="Trajectory points for best-kappa GKSL trace (<=101).")
    args = ap.parse_args()
    run_benchmark(args.config, args.out, traj_points=args.traj_points)


if __name__ == "__main__":
    main()
