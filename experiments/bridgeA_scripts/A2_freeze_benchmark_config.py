#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2_freeze_benchmark_config.py

Step A.2 (Drosophila larva / Mushroom Body): Freeze a reproducible benchmark config
on top of the real connectome graph produced in Step A.1.

What it does:
  1) Loads GraphML(s) from A.1 outputs (core and/or extended).
  2) Detects node "type" attribute and edge weight attribute.
  3) Computes sanity/topology reports (reachability, path length stats).
  4) Picks canonical Source / Target / Distractor sets deterministically.
  5) Builds an "active subgraph" (nodes on PN→...→MBON paths, hop-limited).
  6) Writes:
        - topology_report_<mode>.json
        - bench_config_<mode>.json
        - active_subgraph_<mode>.graphml
        - active_nodes_<mode>.csv
        - active_edges_<mode>.csv

No pandas required (pure stdlib + networkx + numpy).
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict, deque

import networkx as nx
import numpy as np


KNOWN_TYPE_TOKENS = ("PN", "KC", "MBON", "MBIN", "FBN", "FFN")


def _find_graphml_files(a1_dir: str):
    """Recursively find .graphml files under a directory."""
    out = []
    for root, _, files in os.walk(a1_dir):
        for fn in files:
            if fn.lower().endswith(".graphml"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _guess_mode_from_path(p: str) -> str:
    low = os.path.basename(p).lower()
    if "extended" in low:
        return "extended"
    if "core" in low:
        return "core"
    # Fallback: infer from directory names
    low2 = p.lower()
    if "/extended" in low2 or "\\extended" in low2:
        return "extended"
    if "/core" in low2 or "\\core" in low2:
        return "core"
    return "unknown"


def _detect_node_type_attr(G: nx.Graph):
    """
    Detect node attribute containing the coarse cell type.
    Heuristic: attribute whose values contain tokens PN/KC/MBON etc.
    """
    # Collect candidate keys
    keys = set()
    for _, attrs in G.nodes(data=True):
        for k in attrs.keys():
            keys.add(k)

    # Score keys by how many values look like known types
    best_key = None
    best_score = -1
    for k in keys:
        vals = []
        for _, attrs in G.nodes(data=True):
            v = attrs.get(k, None)
            if v is None:
                continue
            vals.append(str(v))
        if not vals:
            continue
        score = 0
        for v in vals:
            vv = v.upper()
            if any(tok in vv for tok in KNOWN_TYPE_TOKENS):
                score += 1
        # Prefer keys that classify most nodes
        if score > best_score:
            best_score = score
            best_key = k

    return best_key, best_score


def _detect_weight_attr(G: nx.Graph):
    """
    Detect edge attribute that represents weight (synapse count).
    Preference order: 'weight', 'w', 'synapses', 'count', otherwise first numeric attr.
    """
    if G.number_of_edges() == 0:
        return None
    u, v, attrs = next(iter(G.edges(data=True)))
    preferred = ["weight", "w", "synapses", "count", "n_syn", "n_synapses"]
    for k in preferred:
        if k in attrs:
            return k
    # Fallback: first numeric
    for k, val in attrs.items():
        try:
            float(val)
            return k
        except Exception:
            continue
    return None


def _as_float(x, default=1.0):
    try:
        return float(x)
    except Exception:
        return default


def _node_type(G: nx.Graph, n, type_key: str):
    v = G.nodes[n].get(type_key, None)
    if v is None:
        return "UNKNOWN"
    return str(v)


def _split_types(G: nx.DiGraph, type_key: str):
    """
    Return dict of {coarse_type: set(nodes)}. Uses substring matching.
    """
    buckets = defaultdict(set)
    for n in G.nodes():
        t = _node_type(G, n, type_key).upper()
        if "MBON" in t:
            buckets["MBONs"].add(n)
        elif "KC" in t:
            buckets["KCs"].add(n)
        elif "PN" in t:
            buckets["PNs"].add(n)
        elif "MBIN" in t:
            buckets["MBINs"].add(n)
        elif "FBN" in t:
            buckets["MB-FBNs"].add(n)
        elif "FFN" in t:
            buckets["MB-FFNs"].add(n)
        else:
            buckets["OTHER"].add(n)
    return buckets


def _edge_type_counts(G: nx.DiGraph, type_key: str):
    c = Counter()
    for u, v in G.edges():
        tu = _node_type(G, u, type_key)
        tv = _node_type(G, v, type_key)
        c[f"{tu}->{tv}"] += 1
    return dict(c)


def _weighted_out_degree(G: nx.DiGraph, nodes, weight_key: str):
    scores = {}
    for n in nodes:
        s = 0.0
        for _, v, attrs in G.out_edges(n, data=True):
            s += _as_float(attrs.get(weight_key, 1.0), 1.0)
        scores[n] = s
    return scores


def _weighted_in_degree(G: nx.DiGraph, nodes, weight_key: str):
    scores = {}
    for n in nodes:
        s = 0.0
        for u, _, attrs in G.in_edges(n, data=True):
            s += _as_float(attrs.get(weight_key, 1.0), 1.0)
        scores[n] = s
    return scores


def _multi_source_bfs_limited(G: nx.DiGraph, sources, max_hops: int):
    """
    Return dict {node: dist} for nodes within max_hops from any source (directed).
    """
    dist = {}
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_hops:
            continue
        for v in G.successors(u):
            if v in dist:
                continue
            dist[v] = du + 1
            q.append(v)
    return dist


def _path_length_stats(G: nx.DiGraph, sources, targets):
    """
    Compute directed shortest-path lengths from any source to each target.
    """
    # BFS from all sources (unweighted hop-count)
    dist = _multi_source_bfs_limited(G, sources, max_hops=10**9)
    lens = []
    unreachable = 0
    for t in targets:
        if t in dist:
            lens.append(dist[t])
        else:
            unreachable += 1
    if lens:
        arr = np.array(lens, dtype=float)
        stats = {
            "n_targets": int(len(targets)),
            "n_reachable": int(len(lens)),
            "n_unreachable": int(unreachable),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }
    else:
        stats = {
            "n_targets": int(len(targets)),
            "n_reachable": 0,
            "n_unreachable": int(unreachable),
        }
    return stats


def _write_csv_nodes(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_csv_edges(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def freeze_one(graphml_path: str, out_dir: str, k_source: int, k_target: int, max_hops: int, seed: int):
    mode = _guess_mode_from_path(graphml_path)
    print(f"\n[A2] Loading GraphML ({mode}): {graphml_path}")

    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        # GraphML can load as Graph; if so, treat as directed for safety
        G = nx.DiGraph(G)

    type_key, type_score = _detect_node_type_attr(G)
    if type_key is None:
        raise RuntimeError("Could not detect node-type attribute in GraphML nodes.")
    weight_key = _detect_weight_attr(G)
    if weight_key is None:
        raise RuntimeError("Could not detect edge-weight attribute in GraphML edges.")

    # Normalize weight attribute to float for all edges
    for u, v, attrs in G.edges(data=True):
        attrs[weight_key] = _as_float(attrs.get(weight_key, 1.0), 1.0)

    buckets = _split_types(G, type_key)
    pns = sorted(list(buckets.get("PNs", [])))
    kcs = sorted(list(buckets.get("KCs", [])))
    mbons = sorted(list(buckets.get("MBONs", [])))

    node_type_counts = {k: len(v) for k, v in buckets.items() if len(v) > 0}

    # Simple sanity
    isolated = [n for n in G.nodes() if (G.in_degree(n) + G.out_degree(n)) == 0]
    n_isolated = len(isolated)

    # Weighted degree for picking sources/targets
    rng = np.random.default_rng(seed)

    pn_scores = _weighted_out_degree(G, pns, weight_key)
    # take top-k_source by weighted out-degree; tie-break by node id for determinism
    pns_sorted = sorted(pns, key=lambda n: (-pn_scores.get(n, 0.0), str(n)))
    source_nodes = pns_sorted[: min(k_source, len(pns_sorted))]

    mbon_scores = _weighted_in_degree(G, mbons, weight_key)
    mbons_sorted = sorted(mbons, key=lambda n: (-mbon_scores.get(n, 0.0), str(n)))
    target_nodes = mbons_sorted[: min(k_target, len(mbons_sorted))]
    distractor_nodes = mbons_sorted[min(k_target, len(mbons_sorted)) : min(2 * k_target, len(mbons_sorted))]

    # Reachability stats (any PN -> MBON)
    path_stats_all = _path_length_stats(G, sources=pns, targets=mbons)
    path_stats_sel = _path_length_stats(G, sources=source_nodes, targets=target_nodes)

    # Active subgraph: forward-reachable from selected sources within max_hops AND
    # reverse-reachable to selected targets within max_hops.
    forward = _multi_source_bfs_limited(G, source_nodes, max_hops=max_hops)
    Grev = G.reverse(copy=False)
    backward = _multi_source_bfs_limited(Grev, target_nodes, max_hops=max_hops)

    active_nodes = set(forward.keys()).intersection(set(backward.keys()))
    # Always include selected endpoints
    active_nodes.update(source_nodes)
    active_nodes.update(target_nodes)
    active_nodes.update(distractor_nodes)

    H = G.subgraph(active_nodes).copy()

    # Edge-type counts (coarse)
    edge_type_counts = Counter()
    for u, v in H.edges():
        tu = _node_type(H, u, type_key)
        tv = _node_type(H, v, type_key)
        edge_type_counts[f"{tu}->{tv}"] += 1

    # Build reports
    def _node_label(n):
        # Try common label keys; else return node id
        attrs = G.nodes[n]
        for k in ("label", "name", "id", "cell_id"):
            if k in attrs and attrs[k] not in (None, ""):
                return str(attrs[k])
        return str(n)

    selection = {
        "sources": [{"node": str(n), "label": _node_label(n), "w_out": pn_scores.get(n, 0.0)} for n in source_nodes],
        "targets": [{"node": str(n), "label": _node_label(n), "w_in": mbon_scores.get(n, 0.0)} for n in target_nodes],
        "distractors": [{"node": str(n), "label": _node_label(n), "w_in": mbon_scores.get(n, 0.0)} for n in distractor_nodes],
    }

    topo_report = {
        "mode": mode,
        "graphml": os.path.abspath(graphml_path),
        "detected": {"node_type_attr": type_key, "node_type_score": int(type_score), "weight_attr": weight_key},
        "counts_full": {"N_nodes": int(G.number_of_nodes()), "N_edges": int(G.number_of_edges()), "N_isolated": int(n_isolated)},
        "counts_types_full": node_type_counts,
        "path_lengths_all_PN_to_all_MBON": path_stats_all,
        "path_lengths_selected_sources_to_selected_targets": path_stats_sel,
        "selection": selection,
        "active_subgraph": {"N_nodes": int(H.number_of_nodes()), "N_edges": int(H.number_of_edges()), "max_hops": int(max_hops)},
        "edge_type_counts_active": dict(edge_type_counts),
    }

    bench_config = {
        "mode": mode,
        "graphml_source": os.path.abspath(graphml_path),
        "node_type_attr": type_key,
        "weight_attr": weight_key,
        "seed": int(seed),
        "k_source": int(k_source),
        "k_target": int(k_target),
        "max_hops": int(max_hops),
        "source_nodes": [str(n) for n in source_nodes],
        "target_nodes": [str(n) for n in target_nodes],
        "distractor_nodes": [str(n) for n in distractor_nodes],
        "active_nodes": [str(n) for n in sorted(active_nodes, key=str)],
        # Simulation defaults (editable later)
        "sim_defaults": {
            "gamma": 1.0,
            "eta_sink": 1.0,
            "kappa_grid": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
            "T_max": 10.0,
            "dt": 0.05,
            "disorder": {"kind": "none", "epsilon": 0.0, "seed": int(seed)},
        },
        "selection_human_readable": selection,
    }

    os.makedirs(out_dir, exist_ok=True)

    # Write JSONs
    topo_path = os.path.join(out_dir, f"topology_report_{mode}.json")
    cfg_path = os.path.join(out_dir, f"bench_config_{mode}.json")
    with open(topo_path, "w", encoding="utf-8") as f:
        json.dump(topo_report, f, indent=2, ensure_ascii=False)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(bench_config, f, indent=2, ensure_ascii=False)

    # Write active subgraph
    subgraph_path = os.path.join(out_dir, f"active_subgraph_{mode}.graphml")
    nx.write_graphml(H, subgraph_path)

    # Write nodes/edges CSV for convenience
    nodes_rows = []
    for n, attrs in H.nodes(data=True):
        row = {"node": str(n), "cell_type": str(attrs.get(type_key, ""))}
        # store a couple of optional labels if exist
        for k in ("label", "name"):
            if k in attrs:
                row[k] = str(attrs.get(k))
        nodes_rows.append(row)

    edges_rows = []
    for u, v, attrs in H.edges(data=True):
        row = {"u": str(u), "v": str(v), "weight": float(attrs.get(weight_key, 1.0))}
        # also preserve edge type if present
        for k in ("type", "edge_type"):
            if k in attrs:
                row[k] = str(attrs.get(k))
        edges_rows.append(row)

    nodes_csv = os.path.join(out_dir, f"active_nodes_{mode}.csv")
    edges_csv = os.path.join(out_dir, f"active_edges_{mode}.csv")
    if nodes_rows:
        _write_csv_nodes(nodes_csv, nodes_rows)
    if edges_rows:
        _write_csv_edges(edges_csv, edges_rows)

    # Print summary
    print(f"[A2:{mode}] node_type_attr='{type_key}', weight_attr='{weight_key}'")
    print(f"[A2:{mode}] FULL: N={G.number_of_nodes()} E={G.number_of_edges()} (isolated={n_isolated})")
    print(f"[A2:{mode}] ACTIVE: N={H.number_of_nodes()} E={H.number_of_edges()} (max_hops={max_hops})")
    print(f"[A2:{mode}] Sources={len(source_nodes)} Targets={len(target_nodes)} Distractors={len(distractor_nodes)}")
    print(f"[A2:{mode}] Wrote:")
    print(f"  - {topo_path}")
    print(f"  - {cfg_path}")
    print(f"  - {subgraph_path}")
    if nodes_rows:
        print(f"  - {nodes_csv}")
    if edges_rows:
        print(f"  - {edges_csv}")

    return topo_path, cfg_path, subgraph_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1_dir", default=None, help="Folder where Step-A1 wrote GraphML(s). If set, auto-discovers .graphml files.")
    ap.add_argument("--graphml", default=None, nargs="*", help="Explicit GraphML file path(s). Overrides --a1_dir.")
    ap.add_argument("--out_dir", default="A2_outputs", help="Output folder.")
    ap.add_argument("--k_source", type=int, default=10, help="Number of PN source nodes to freeze.")
    ap.add_argument("--k_target", type=int, default=5, help="Number of MBON target nodes to freeze.")
    ap.add_argument("--max_hops", type=int, default=4, help="Hop limit used to define the active subgraph.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic selection.")
    args = ap.parse_args()

    graphml_files = []
    if args.graphml:
        graphml_files = [os.path.abspath(os.path.expanduser(p)) for p in args.graphml]
    elif args.a1_dir:
        a1 = os.path.abspath(os.path.expanduser(args.a1_dir))
        graphml_files = _find_graphml_files(a1)
        if not graphml_files:
            raise RuntimeError(f"No .graphml files found under: {a1}")
    else:
        raise RuntimeError("Provide either --graphml <file(s)> or --a1_dir <folder>.")

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Freeze each graphml found
    for p in graphml_files:
        freeze_one(
            graphml_path=p,
            out_dir=out_dir,
            k_source=args.k_source,
            k_target=args.k_target,
            max_hops=args.max_hops,
            seed=args.seed,
        )

    print("\n[A2] DONE")


if __name__ == "__main__":
    main()
