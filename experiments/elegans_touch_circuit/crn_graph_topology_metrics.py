#!/usr/bin/env python3
"""
CRN THEORY — graph topology metrics helper
=========================================

Purpose
-------
Compute topology metrics needed for CRN/CTQW/CRW simulation writeups:

- Effective depth L_eff between a source set S (e.g., sensory neurons) and a target set T (e.g., motor/readout nodes)
- Diameter / average shortest path (compactness)
- Clustering coefficient + a small-worldness index sigma (Humphries–Gurney style; ER baseline)
- Optional per-pair distance table for auditing

Key design choice (to avoid "L confusion")
------------------------------------------
We report depth in *edges* (graph distance) as:
    L_eff_edges := mean_{s in S, t in T} d(s,t)

If you also want the "layer count" (nodes along the path), use:
    L_eff_nodes := L_eff_edges + 1

In Glued Trees benchmarks, the manuscript's L parameter is a *tree depth*,
and the root-to-root shortest-path distance is 2L edges. Here we do not
reuse that symbol; we compute distances directly to avoid ambiguity.

Usage
-----
(1) Built-in preset (C. elegans touch-withdrawal toy circuit used in scripts):
    python crn_graph_topology_metrics.py --preset elegans_touch \
        --sources ALML ALMR AVM PLML PLMR PVM \
        --targets VA1 VB1

(2) JSON output for easy inclusion into the paper/repo:
    python crn_graph_topology_metrics.py --preset elegans_touch \
        --json-out elegans_metrics.json

Dependencies: networkx, numpy (standard in the CRN env).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


# -----------------------------
# Preset: C. elegans touch circuit (matches crn_nematode_honest_test.py)
# -----------------------------
SENSORY = ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM']
INTER = ['AVD', 'PVC', 'AVA', 'AVB']
MOTOR = [
    'VA1', 'VA2', 'VA3', 'VA4', 'DA1', 'DA2', 'DA3',
    'VB1', 'VB2', 'VB3', 'VB4', 'DB1', 'DB2', 'DB3'
]

# Gap junction edges (treated as undirected)
GJ_EDGES = [
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'),
    ('PLML', 'PVC'), ('PLMR', 'PVC'), ('PVM', 'PVC'),
    ('AVD', 'AVA'), ('PVC', 'AVB'),
    ('AVA', 'VA1'), ('AVA', 'DA1'),
    ('AVB', 'VB1'), ('AVB', 'DB1'),
    ('ALML', 'ALMR'), ('PLML', 'PLMR'),
]

# Chemical synapses (in the CRN scripts they are symmetrized as undirected "hopping")
CHEM_EDGES = [
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'),
    ('ALML', 'PVC'), ('ALMR', 'PVC'),
    ('PLML', 'PVC'), ('PLMR', 'PVC'),
    ('PLML', 'AVD'), ('PLMR', 'AVD'),
    ('AVA', 'VA1'), ('AVA', 'VA2'), ('AVA', 'VA3'), ('AVA', 'DA1'), ('AVA', 'DA2'),
    ('AVB', 'VB1'), ('AVB', 'VB2'), ('AVB', 'VB3'), ('AVB', 'DB1'), ('AVB', 'DB2'),
    ('PVC', 'DB1'), ('PVC', 'DB2'),
    ('AVD', 'VA1'), ('AVD', 'VA2'),
]


def build_elegans_touch_circuit(
    *,
    add_motor_chain: bool = True,
    w_gap: float = 2.0,
    w_chem: float = 1.0,
    w_chain: float = 1.0,
) -> nx.Graph:
    """
    Builds the simplified touch withdrawal circuit graph used in the CRN nematode scripts.

    Notes
    -----
    - Uses nx.Graph (undirected) by design, matching the "symmetrized hopping" approximation.
    - If you want a more biologically faithful directed synapse graph, build it separately and
      pass it to the metric functions below.
    """
    G = nx.Graph()
    all_nodes = SENSORY + INTER + MOTOR
    G.add_nodes_from(all_nodes)

    for u, v in GJ_EDGES:
        if G.has_edge(u, v):
            G[u][v]['weight'] = float(G[u][v].get('weight', 0.0) + w_gap)
        else:
            G.add_edge(u, v, weight=float(w_gap))

    for u, v in CHEM_EDGES:
        if G.has_edge(u, v):
            G[u][v]['weight'] = float(G[u][v].get('weight', 0.0) + w_chem)
        else:
            G.add_edge(u, v, weight=float(w_chem))

    if add_motor_chain:
        for i in range(len(MOTOR) - 1):
            u, v = MOTOR[i], MOTOR[i + 1]
            if G.has_edge(u, v):
                G[u][v]['weight'] = float(G[u][v].get('weight', 0.0) + w_chain)
            else:
                G.add_edge(u, v, weight=float(w_chain))

    return G


# -----------------------------
# Metrics
# -----------------------------
@dataclass(frozen=True)
class DepthStats:
    n_pairs: int
    n_reachable: int
    n_unreachable: int
    mean_edges: float
    median_edges: float
    std_edges: float
    min_edges: Optional[int]
    max_edges: Optional[int]
    q10_edges: Optional[float]
    q25_edges: Optional[float]
    q75_edges: Optional[float]
    q90_edges: Optional[float]
    mean_nodes: float  # mean_edges + 1

    # Audit payload (small graphs): distances as a list
    distances_edges: List[int]


def effective_depth(
    G: nx.Graph,
    sources: Sequence[Any],
    targets: Sequence[Any],
    *,
    weight: Optional[str] = None,
) -> DepthStats:
    """
    Compute effective depth statistics between source set S and target set T.

    Parameters
    ----------
    G : nx.Graph (or nx.DiGraph)
        Graph on which distances are computed.
    sources, targets : sequences
        Node labels for the source/target sets.
    weight : str or None
        If provided, compute weighted shortest paths using this edge attribute.
        For "depth" in the biological narrative you usually want unweighted hops,
        so default is None.

    Returns
    -------
    DepthStats
    """
    dists: List[int] = []
    n_pairs = 0
    n_unreachable = 0

    # Pre-compute all-pairs shortest paths from each source for speed.
    for s in sources:
        if s not in G:
            continue
        n_pairs += len([t for t in targets if t in G])

        if weight is None:
            lengths = nx.single_source_shortest_path_length(G, s)
        else:
            lengths = nx.single_source_dijkstra_path_length(G, s, weight=weight)

        for t in targets:
            if t not in G:
                continue
            if t in lengths:
                dists.append(int(lengths[t]))
            else:
                n_unreachable += 1

    n_reachable = len(dists)
    if n_reachable == 0:
        # Degenerate case: return NaNs but keep structure stable
        return DepthStats(
            n_pairs=n_pairs,
            n_reachable=0,
            n_unreachable=n_unreachable,
            mean_edges=float('nan'),
            median_edges=float('nan'),
            std_edges=float('nan'),
            min_edges=None,
            max_edges=None,
            q10_edges=None,
            q25_edges=None,
            q75_edges=None,
            q90_edges=None,
            mean_nodes=float('nan'),
            distances_edges=[],
        )

    arr = np.asarray(dists, dtype=float)
    qs = np.quantile(arr, [0.10, 0.25, 0.75, 0.90])

    return DepthStats(
        n_pairs=n_pairs,
        n_reachable=n_reachable,
        n_unreachable=n_unreachable,
        mean_edges=float(arr.mean()),
        median_edges=float(np.median(arr)),
        std_edges=float(arr.std(ddof=1)) if n_reachable > 1 else 0.0,
        min_edges=int(arr.min()),
        max_edges=int(arr.max()),
        q10_edges=float(qs[0]),
        q25_edges=float(qs[1]),
        q75_edges=float(qs[2]),
        q90_edges=float(qs[3]),
        mean_nodes=float(arr.mean() + 1.0),
        distances_edges=[int(x) for x in dists],
    )


@dataclass(frozen=True)
class SmallWorldStats:
    C: float
    L: float
    C_rand: float
    L_rand: float
    sigma: float
    n_random: int


def _largest_cc_undirected(G: nx.Graph) -> nx.Graph:
    Gu = G.to_undirected() if G.is_directed() else G
    if nx.is_connected(Gu):
        return Gu
    largest = max(nx.connected_components(Gu), key=len)
    return Gu.subgraph(largest).copy()


def small_world_index_sigma(
    G: nx.Graph,
    *,
    n_random: int = 200,
    seed: int = 42,
) -> SmallWorldStats:
    """
    Compute a simple small-worldness index sigma relative to an ER ensemble.

    sigma := (C/C_rand) / (L/L_rand)

    Where:
      C = average clustering coefficient
      L = average shortest path length (on largest connected component)
      C_rand, L_rand = ensemble means over ER(n, p) graphs with the same n and p ~ density

    This is not meant as a definitive network-science claim; it's a sanity check
    supporting 'compact + clustered' language in the manuscript.
    """
    rng = random.Random(seed)

    H = _largest_cc_undirected(G)
    n = H.number_of_nodes()
    m = H.number_of_edges()
    if n < 3:
        return SmallWorldStats(C=float('nan'), L=float('nan'), C_rand=float('nan'), L_rand=float('nan'),
                              sigma=float('nan'), n_random=n_random)

    # Observed metrics
    C = float(nx.average_clustering(H))
    L = float(nx.average_shortest_path_length(H))

    # ER baseline with matching density (or matching expected edges)
    p = (2.0 * m) / (n * (n - 1))

    Cs: List[float] = []
    Ls: List[float] = []

    for _ in range(n_random):
        # networkx uses numpy rng internally; we pass a varying seed for reproducibility without global state
        er_seed = rng.randint(0, 2**32 - 1)
        Gr = nx.erdos_renyi_graph(n, p, seed=er_seed)

        if not nx.is_connected(Gr):
            largest = max(nx.connected_components(Gr), key=len)
            Gr = Gr.subgraph(largest).copy()

        # If graph collapsed too much (rare at these densities), skip it.
        if Gr.number_of_nodes() < 3:
            continue

        Cs.append(float(nx.average_clustering(Gr)))
        Ls.append(float(nx.average_shortest_path_length(Gr)))

    C_rand = float(np.mean(Cs)) if Cs else float('nan')
    L_rand = float(np.mean(Ls)) if Ls else float('nan')

    sigma = float('nan')
    if C_rand > 0 and L_rand > 0:
        sigma = (C / C_rand) / (L / L_rand)

    return SmallWorldStats(C=C, L=L, C_rand=C_rand, L_rand=L_rand, sigma=sigma, n_random=len(Cs))


@dataclass(frozen=True)
class GraphSummary:
    n_nodes: int
    n_edges: int
    density: float
    n_components: int
    largest_component_size: int
    diameter: Optional[int]
    avg_shortest_path: Optional[float]
    avg_clustering: Optional[float]
    small_world: Optional[SmallWorldStats]
    depth: Optional[DepthStats]


def summarize_graph(
    G: nx.Graph,
    *,
    sources: Optional[Sequence[Any]] = None,
    targets: Optional[Sequence[Any]] = None,
    weight: Optional[str] = None,
    compute_small_world: bool = True,
    n_random: int = 200,
    seed: int = 42,
) -> GraphSummary:
    Gu = G.to_undirected() if G.is_directed() else G
    n = Gu.number_of_nodes()
    m = Gu.number_of_edges()
    dens = float(nx.density(Gu)) if n > 1 else 0.0

    comps = list(nx.connected_components(Gu)) if n > 0 else []
    n_comp = len(comps)
    largest = max(comps, key=len) if comps else set()
    H = Gu.subgraph(largest).copy() if largest else Gu

    diam = None
    L = None
    if H.number_of_nodes() >= 2 and nx.is_connected(H):
        diam = int(nx.diameter(H))
        L = float(nx.average_shortest_path_length(H))

    C = float(nx.average_clustering(H)) if H.number_of_nodes() >= 3 else None

    sw = None
    if compute_small_world and H.number_of_nodes() >= 3 and H.number_of_edges() > 0:
        sw = small_world_index_sigma(H, n_random=n_random, seed=seed)

    depth_stats = None
    if sources is not None and targets is not None:
        depth_stats = effective_depth(G, sources, targets, weight=weight)

    return GraphSummary(
        n_nodes=n,
        n_edges=m,
        density=dens,
        n_components=n_comp,
        largest_component_size=H.number_of_nodes(),
        diameter=diam,
        avg_shortest_path=L,
        avg_clustering=C,
        small_world=sw,
        depth=depth_stats,
    )


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRN graph topology metrics helper")
    p.add_argument("--preset", choices=["elegans_touch"], default="elegans_touch",
                   help="Graph preset to analyze.")
    p.add_argument("--sources", nargs="*", default=SENSORY,
                   help="Source nodes (default: preset sensory set).")
    p.add_argument("--targets", nargs="*", default=["VA1", "VB1"],
                   help="Target nodes (default: VA1 VB1).")
    p.add_argument("--weight", default=None,
                   help="Edge attribute to use as distance weight (default: unweighted hops).")
    p.add_argument("--no-small-world", action="store_true",
                   help="Disable ER small-world sigma computation.")
    p.add_argument("--n-random", type=int, default=200,
                   help="Number of ER samples for sigma (default: 200).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for ER baseline (default: 42).")
    p.add_argument("--json-out", default=None,
                   help="Write full summary to JSON file.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.preset == "elegans_touch":
        G = build_elegans_touch_circuit()

    summary = summarize_graph(
        G,
        sources=args.sources,
        targets=args.targets,
        weight=args.weight,
        compute_small_world=(not args.no_small_world),
        n_random=args.n_random,
        seed=args.seed,
    )

    # Human-readable print
    print("=== Graph Summary ===")
    print(f"N nodes: {summary.n_nodes}")
    print(f"M edges: {summary.n_edges}")
    print(f"Density: {summary.density:.4f}")
    print(f"Connected components: {summary.n_components} (largest: {summary.largest_component_size})")
    if summary.diameter is not None:
        print(f"Diameter (largest CC): {summary.diameter}")
    if summary.avg_shortest_path is not None:
        print(f"Avg shortest path (largest CC): {summary.avg_shortest_path:.3f}")
    if summary.avg_clustering is not None:
        print(f"Avg clustering (largest CC): {summary.avg_clustering:.3f}")

    if summary.depth is not None:
        d = summary.depth
        print("\n=== Effective Depth S→T ===")
        print(f"Pairs total: {d.n_pairs}, reachable: {d.n_reachable}, unreachable: {d.n_unreachable}")
        print(f"L_eff_edges (mean): {d.mean_edges:.3f}  | median: {d.median_edges:.1f}  | std: {d.std_edges:.3f}")
        print(f"Range (edges): [{d.min_edges}, {d.max_edges}]")
        print(f"Quantiles (edges): q10={d.q10_edges:.1f}, q25={d.q25_edges:.1f}, q75={d.q75_edges:.1f}, q90={d.q90_edges:.1f}")
        print(f"L_eff_nodes (mean): {d.mean_nodes:.3f}")

    if summary.small_world is not None:
        sw = summary.small_world
        print("\n=== Small-world sigma (ER baseline) ===")
        print(f"C: {sw.C:.3f}, L: {sw.L:.3f}")
        print(f"C_rand: {sw.C_rand:.3f}, L_rand: {sw.L_rand:.3f}  (n={sw.n_random})")
        print(f"sigma: {sw.sigma:.3f}")

    if args.json_out:
        payload: Dict[str, Any] = asdict(summary)

        # dataclasses -> dict ok, but nested dataclasses already converted by asdict
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
