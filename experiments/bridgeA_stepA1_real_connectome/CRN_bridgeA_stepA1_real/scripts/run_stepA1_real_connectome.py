#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bridge-A / Step A.1 — Build a REAL connectome graph and extract MB subgraph.

This script is designed to be *reviewer-friendly* and reproducible:
- deterministic (no random sampling unless explicitly requested)
- emits JSON/CSV/GraphML artifacts
- performs sanity/QC checks

Supported inputs (recommended):
- Netzschleuder fly_larva: fly_larva.csv.zip OR fly_larva.xml.zst

If you already have a different real connectome export (edges+nodes CSV),
see the helper `--format generic_csv` mode.

NOTE: This script does NOT fetch the dataset automatically by default.
You are expected to download it and pass via --input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np


try:
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -----------------------------
# Utilities
# -----------------------------


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MBRegex:
    pn: str
    kc: str
    mbon: str
    apl: str
    dan: str


# -----------------------------
# Loaders
# -----------------------------


def load_netzschleuder_graphml_zst(path: Path) -> nx.MultiDiGraph:
    """Load Netzschleuder GraphML packed with zstandard (.xml.zst).

    Requires `pip install zstandard`.
    """
    try:
        import zstandard as zstd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Input is .zst but `zstandard` is not installed. Install with: pip install zstandard"
        ) from e

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_xml = td_path / "fly_larva.graphml"

        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as fin, open(out_xml, "wb") as fout:
            dctx.copy_stream(fin, fout)

        # networkx uses lxml/ElementTree internally; GraphML is XML.
        G = nx.read_graphml(out_xml)

    # networkx returns MultiDiGraph for GraphML with parallel edges
    if not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Unexpected graph type from GraphML: {type(G)}")

    if isinstance(G, nx.DiGraph):
        # normalize to MultiDiGraph
        MG = nx.MultiDiGraph(G)
    else:
        MG = G

    return MG


def _read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with zf.open(member, "r") as f:
        # Decode as UTF-8 with fallbacks
        text = (line.decode("utf-8", errors="replace") for line in f)
        reader = csv.DictReader(text)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def load_netzschleuder_csvzip(path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load Netzschleuder CSV zip.

    Netzschleuder's CSV export is typically a ZIP with 2 CSV files:
    - one for vertices
    - one for edges

    We try to detect them heuristically.

    Returns:
        (nodes_rows, edges_rows)
    """
    with zipfile.ZipFile(path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV members found inside zip: {path}")

        tables: Dict[str, List[Dict[str, str]]] = {}
        for m in members:
            tables[m] = _read_csv_from_zip(zf, m)

    # Score tables
    def score_edges(rows: List[Dict[str, str]]) -> int:
        if not rows:
            return -10
        cols = {c.lower().strip() for c in rows[0].keys() if c}
        score = 0
        # typical edge list columns
        if {"source", "target"}.issubset(cols):
            score += 10
        if {"src", "dst"}.issubset(cols):
            score += 8
        if {"from", "to"}.issubset(cols):
            score += 8
        if "count" in cols or "weight" in cols or "synapses" in cols:
            score += 3
        if "etype" in cols:
            score += 1
        # edges should have many rows
        score += min(5, len(rows) // 10000)
        return score

    def score_nodes(rows: List[Dict[str, str]]) -> int:
        if not rows:
            return -10
        cols = {c.lower().strip() for c in rows[0].keys() if c}
        score = 0
        if "cell_type" in cols:
            score += 10
        if "hemisphere" in cols:
            score += 2
        if "annotations" in cols:
            score += 2
        if "vid" in cols or "vids" in cols or "id" in cols:
            score += 2
        # nodes should be fewer than edges
        score += min(3, len(rows) // 1000)
        return score

    best_edge_key = max(tables.keys(), key=lambda k: score_edges(tables[k]))
    best_node_key = max(tables.keys(), key=lambda k: score_nodes(tables[k]))

    edges_rows = tables[best_edge_key]
    nodes_rows = tables[best_node_key]

    if best_edge_key == best_node_key:
        # If detection failed (single table), attempt to split by presence of source/target
        cols = {c.lower().strip() for c in edges_rows[0].keys() if c}
        if {"source", "target"}.issubset(cols):
            # treat as edges; nodes unknown
            raise RuntimeError(
                "CSV zip appears to contain only an edge table; node metadata table not found. "
                "Please provide GraphML (.xml.zst) or a nodes CSV as well."
            )
        raise RuntimeError(
            f"Unable to detect nodes vs edges CSV inside zip. Members={list(tables.keys())}"
        )

    return nodes_rows, edges_rows


# -----------------------------
# Graph construction
# -----------------------------


def build_digraph_from_netzschleuder_tables(
    nodes_rows: List[Dict[str, str]],
    edges_rows: List[Dict[str, str]],
    min_syn: int = 0,
) -> nx.DiGraph:
    """Construct a DiGraph from Netzschleuder CSV tables."""

    # --- Nodes ---
    # Determine node id column
    node_cols = {c.lower(): c for c in (nodes_rows[0].keys() if nodes_rows else []) if c}
    node_id_col = None
    for cand in ["vid", "vids", "id", "node", "node_id", "neuron", "neuron_id"]:
        if cand in node_cols:
            node_id_col = node_cols[cand]
            break
    if node_id_col is None:
        # Fall back to first column
        node_id_col = list(nodes_rows[0].keys())[0]

    G = nx.DiGraph()

    for r in nodes_rows:
        nid = _as_str(r.get(node_id_col))
        if nid == "":
            continue
        attrs = {k: v for k, v in r.items() if k != node_id_col}
        G.add_node(nid, **attrs)

    # --- Edges ---
    edge_cols = {c.lower(): c for c in (edges_rows[0].keys() if edges_rows else []) if c}
    src_col = None
    dst_col = None
    for sc, tc in [("source", "target"), ("src", "dst"), ("from", "to")]:
        if sc in edge_cols and tc in edge_cols:
            src_col = edge_cols[sc]
            dst_col = edge_cols[tc]
            break
    if src_col is None or dst_col is None:
        raise RuntimeError(
            f"Edge table must contain source/target columns. Found: {list(edge_cols.keys())}"
        )

    w_col = None
    for cand in ["count", "weight", "synapses", "w"]:
        if cand in edge_cols:
            w_col = edge_cols[cand]
            break

    etype_col = edge_cols.get("etype")

    missing_nodes = 0
    kept_edges = 0
    for r in edges_rows:
        u = _as_str(r.get(src_col))
        v = _as_str(r.get(dst_col))
        if u == "" or v == "":
            continue

        w = _safe_int(r.get(w_col), default=1) if w_col is not None else 1
        if w < min_syn:
            continue

        if u not in G:
            G.add_node(u)
            missing_nodes += 1
        if v not in G:
            G.add_node(v)
            missing_nodes += 1

        attrs = {}
        if w_col is not None:
            attrs["count"] = w
        if etype_col is not None:
            attrs["etype"] = _as_str(r.get(etype_col))

        # If parallel edges exist, sum counts
        if G.has_edge(u, v):
            prev = G[u][v].get("count", 1)
            G[u][v]["count"] = _safe_int(prev, default=1) + w
            # Keep etype if exists; otherwise ignore
        else:
            G.add_edge(u, v, **attrs)
        kept_edges += 1

    return G


def collapse_multidigraph_to_digraph(MG: nx.MultiDiGraph, min_syn: int = 0) -> nx.DiGraph:
    """Convert MultiDiGraph to DiGraph summing `count` weights."""
    G = nx.DiGraph()
    for n, attrs in MG.nodes(data=True):
        G.add_node(str(n), **{k: _as_str(v) for k, v in attrs.items()})

    for u, v, attrs in MG.edges(data=True):
        uu = str(u)
        vv = str(v)
        w = None
        # Netzschleuder uses `count` for synapse count.
        if "count" in attrs:
            w = _safe_int(attrs.get("count"), default=1)
        elif "weight" in attrs:
            w = _safe_int(attrs.get("weight"), default=1)
        else:
            w = 1

        if w < min_syn:
            continue

        if G.has_edge(uu, vv):
            G[uu][vv]["count"] = _safe_int(G[uu][vv].get("count"), default=0) + w
        else:
            out_attrs = {}
            out_attrs["count"] = w
            if "etype" in attrs:
                out_attrs["etype"] = _as_str(attrs.get("etype"))
            G.add_edge(uu, vv, **out_attrs)

    return G


# -----------------------------
# MB extraction + QC
# -----------------------------


def _node_label_for_match(attrs: Dict[str, Any]) -> str:
    # Prefer cell_type; fallback to annotations
    s = _as_str(attrs.get("cell_type"))
    if s.strip() != "":
        return s
    s = _as_str(attrs.get("annotations"))
    return s


def classify_mb_nodes(G: nx.DiGraph, rx: MBRegex) -> Dict[str, str]:
    """Return mapping node_id -> group label {PN, KC, MBON, APL, DAN, OTHER} for matched nodes only."""
    out: Dict[str, str] = {}
    re_pn = re.compile(rx.pn, flags=re.IGNORECASE)
    re_kc = re.compile(rx.kc, flags=re.IGNORECASE)
    re_mbon = re.compile(rx.mbon, flags=re.IGNORECASE)
    re_apl = re.compile(rx.apl, flags=re.IGNORECASE)
    re_dan = re.compile(rx.dan, flags=re.IGNORECASE)

    for n, attrs in G.nodes(data=True):
        label = _node_label_for_match(attrs)
        if label == "":
            continue

        if re_kc.search(label):
            out[n] = "KC"
        elif re_mbon.search(label):
            out[n] = "MBON"
        elif re_pn.search(label):
            out[n] = "PN"
        elif re_apl.search(label):
            out[n] = "APL"
        elif re_dan.search(label):
            out[n] = "DAN"

    return out


def extract_mb_subgraph(G: nx.DiGraph, node_groups: Dict[str, str]) -> nx.DiGraph:
    nodes = list(node_groups.keys())
    return G.subgraph(nodes).copy()


def qc_mb(G_mb: nx.DiGraph, node_groups: Dict[str, str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    groups = {
        "PN": {n for n, g in node_groups.items() if g == "PN"},
        "KC": {n for n, g in node_groups.items() if g == "KC"},
        "MBON": {n for n, g in node_groups.items() if g == "MBON"},
        "APL": {n for n, g in node_groups.items() if g == "APL"},
        "DAN": {n for n, g in node_groups.items() if g == "DAN"},
    }

    stats["n_nodes"] = G_mb.number_of_nodes()
    stats["n_edges"] = G_mb.number_of_edges()
    stats["group_sizes"] = {k: len(v) for k, v in groups.items()}

    # Edge-type fractions (PN->KC, KC->MBON, etc.)
    def count_edges(src_set: set, dst_set: set) -> Tuple[int, int]:
        n = 0
        wsum = 0
        for u, v, attrs in G_mb.edges(data=True):
            if u in src_set and v in dst_set:
                n += 1
                wsum += _safe_int(attrs.get("count"), default=1)
        return n, wsum

    pn_kc_n, pn_kc_w = count_edges(groups["PN"], groups["KC"])
    kc_mbon_n, kc_mbon_w = count_edges(groups["KC"], groups["MBON"])

    stats["edges_PN_to_KC"] = {"n": pn_kc_n, "synapses": pn_kc_w}
    stats["edges_KC_to_MBON"] = {"n": kc_mbon_n, "synapses": kc_mbon_w}

    # Divergence / convergence metrics
    # For each KC: number of PN inputs, number of MBON outputs
    def indeg_from_set(node: str, src_set: set) -> int:
        c = 0
        for u in G_mb.predecessors(node):
            if u in src_set:
                c += 1
        return c

    def outdeg_to_set(node: str, dst_set: set) -> int:
        c = 0
        for v in G_mb.successors(node):
            if v in dst_set:
                c += 1
        return c

    kc_pn_in = [indeg_from_set(kc, groups["PN"]) for kc in groups["KC"]]
    kc_mbon_out = [outdeg_to_set(kc, groups["MBON"]) for kc in groups["KC"]]

    def summarize_int_list(xs: List[int]) -> Dict[str, Any]:
        if not xs:
            return {"n": 0}
        arr = np.array(xs, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }

    stats["KC_in_from_PN"] = summarize_int_list(kc_pn_in)
    stats["KC_out_to_MBON"] = summarize_int_list(kc_mbon_out)

    # SCC structure (MB is not expected to be strongly recurrent if we only keep PN/KC/MBON)
    try:
        scc_sizes = sorted((len(c) for c in nx.strongly_connected_components(G_mb)), reverse=True)
        stats["scc_sizes_top10"] = scc_sizes[:10]
    except Exception:
        stats["scc_sizes_top10"] = None

    return stats


def plot_degree_hist(G: nx.DiGraph, out_png: Path, title: str = "") -> None:
    if not _HAS_PLT:
        return

    indeg = [d for _, d in G.in_degree()]
    outdeg = [d for _, d in G.out_degree()]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(indeg, bins=30)
    plt.title("in-degree")
    plt.subplot(1, 2, 2)
    plt.hist(outdeg, bins=30)
    plt.title("out-degree")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def export_nodes_csv(G: nx.DiGraph, node_groups: Optional[Dict[str, str]], out_csv: Path) -> None:
    # Collect all attribute keys
    keys: List[str] = ["node_id"]
    if node_groups is not None:
        keys.append("group")

    all_attr_keys = set()
    for _, attrs in G.nodes(data=True):
        all_attr_keys.update(attrs.keys())

    # Keep stable order (common keys first)
    preferred = ["cell_type", "annotations", "hemisphere", "homologue", "cluster"]
    ordered_attrs = [k for k in preferred if k in all_attr_keys] + sorted(
        [k for k in all_attr_keys if k not in preferred]
    )

    keys += ordered_attrs

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for n, attrs in G.nodes(data=True):
            row = {"node_id": str(n)}
            if node_groups is not None:
                row["group"] = node_groups.get(n, "")
            for k in ordered_attrs:
                row[k] = _as_str(attrs.get(k))
            w.writerow(row)


def export_edges_csv(G: nx.DiGraph, out_csv: Path) -> None:
    keys = ["source", "target", "count", "etype"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for u, v, attrs in G.edges(data=True):
            w.writerow(
                {
                    "source": str(u),
                    "target": str(v),
                    "count": _safe_int(attrs.get("count"), default=1),
                    "etype": _as_str(attrs.get("etype")),
                }
            )


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bridge-A Step A.1: build REAL larva connectome graph and extract MB subgraph"
    )
    ap.add_argument("--input", type=str, required=True, help="Path to fly_larva.csv.zip or fly_larva.xml.zst")
    ap.add_argument("--outdir", type=str, default="outputs_real", help="Output directory")
    ap.add_argument("--min_syn", type=int, default=5, help="Minimum synapse count threshold for keeping an edge")

    ap.add_argument(
        "--pn_regex",
        type=str,
        default=r"\\bPN\\b|Projection\\s*Neuron",
        help="Regex to match PN nodes by cell_type/annotations",
    )
    ap.add_argument(
        "--kc_regex",
        type=str,
        default=r"\\bKC\\b|Kenyon",
        help="Regex to match KC nodes by cell_type/annotations",
    )
    ap.add_argument(
        "--mbon_regex",
        type=str,
        default=r"MBON",
        help="Regex to match MBON nodes by cell_type/annotations",
    )
    ap.add_argument(
        "--apl_regex",
        type=str,
        default=r"\\bAPL\\b",
        help="Regex to match APL nodes (optional)",
    )
    ap.add_argument(
        "--dan_regex",
        type=str,
        default=r"\\bDAN\\b|Dopamin",
        help="Regex to match DAN nodes (optional)",
    )

    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    _ensure_dir(outdir)

    rx = MBRegex(
        pn=args.pn_regex,
        kc=args.kc_regex,
        mbon=args.mbon_regex,
        apl=args.apl_regex,
        dan=args.dan_regex,
    )

    # --- Load graph ---
    if in_path.name.lower().endswith(".xml.zst") or in_path.name.lower().endswith(".gml.zst"):
        MG = load_netzschleuder_graphml_zst(in_path)
        G = collapse_multidigraph_to_digraph(MG, min_syn=args.min_syn)
        load_mode = "graphml_zst"
    elif in_path.name.lower().endswith(".csv.zip"):
        nodes_rows, edges_rows = load_netzschleuder_csvzip(in_path)
        G = build_digraph_from_netzschleuder_tables(nodes_rows, edges_rows, min_syn=args.min_syn)
        load_mode = "csvzip"
    elif in_path.name.lower().endswith(".graphml") or in_path.name.lower().endswith(".xml"):
        MG = nx.read_graphml(in_path)
        if isinstance(MG, nx.DiGraph):
            MG = nx.MultiDiGraph(MG)
        G = collapse_multidigraph_to_digraph(MG, min_syn=args.min_syn)
        load_mode = "graphml"
    else:
        raise RuntimeError(f"Unsupported input format: {in_path}")

    # --- Full graph stats ---
    full_stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "min_syn": args.min_syn,
        "load_mode": load_mode,
    }
    (outdir / "fly_larva_full_stats.json").write_text(json.dumps(full_stats, indent=2), encoding="utf-8")

    # --- Extract MB subgraph ---
    node_groups = classify_mb_nodes(G, rx)
    G_mb = extract_mb_subgraph(G, node_groups)

    mb_stats = qc_mb(G_mb, node_groups)
    (outdir / "fly_larva_mb_stats.json").write_text(json.dumps(mb_stats, indent=2), encoding="utf-8")

    # --- Exports ---
    # GraphML
    nx.write_graphml(G_mb, outdir / "fly_larva_mb.graphml")

    export_nodes_csv(G_mb, node_groups, outdir / "fly_larva_mb_nodes.csv")
    export_edges_csv(G_mb, outdir / "fly_larva_mb_edgelist.csv")

    if _HAS_PLT:
        plot_degree_hist(G_mb, outdir / "fly_larva_mb_degree_hist.png", title="fly_larva MB subgraph")

    metadata = {
        "generated_utc": _now_iso(),
        "input": str(in_path),
        "load_mode": load_mode,
        "min_syn": args.min_syn,
        "regex": {
            "PN": rx.pn,
            "KC": rx.kc,
            "MBON": rx.mbon,
            "APL": rx.apl,
            "DAN": rx.dan,
        },
        "full_graph": full_stats,
        "mb_stats": mb_stats,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[OK] Step A.1 complete")
    print(f"Full graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print(
        "MB subgraph: nodes={n} edges={e}".format(
            n=G_mb.number_of_nodes(), e=G_mb.number_of_edges()
        )
    )
    print(f"Outputs written to: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
