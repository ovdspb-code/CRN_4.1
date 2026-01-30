#!/usr/bin/env python3
"""
C. ELEGANS TOUCH CIRCUIT: HONEST TEST
======================================
Fixed parameters:
- Epsilon = 1.0 (biologically plausible)
- Target = 3 key motor neurons (VA1, VB1, DA1)
- Baselines: CRW, CTQW, CRN(κ-scan)
"""

import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, diags, kron, eye
from scipy.sparse.linalg import expm_multiply

# --- 1. CIRCUIT DATA (same as before) ---
SENSORY = ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM']
INTER = ['AVD', 'PVC', 'AVA', 'AVB']
MOTOR = [
    'VA1', 'VA2', 'VA3', 'VA4', 'DA1', 'DA2', 'DA3',
    'VB1', 'VB2', 'VB3', 'VB4', 'DB1', 'DB2', 'DB3'
]

GJ_EDGES = [
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'),
    ('PLML', 'PVC'), ('PLMR', 'PVC'), ('PVM', 'PVC'),
    ('AVD', 'AVA'), ('PVC', 'AVB'),
    ('AVA', 'VA1'), ('AVA', 'DA1'),
    ('AVB', 'VB1'), ('AVB', 'DB1'),
    ('ALML', 'ALMR'), ('PLML', 'PLMR')
]

CHEM_EDGES = [
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'),
    ('ALML', 'PVC'), ('ALMR', 'PVC'),
    ('PLML', 'PVC'), ('PLMR', 'PVC'),
    ('PLML', 'AVD'), ('PLMR', 'AVD'),
    ('AVA', 'VA1'), ('AVA', 'VA2'), ('AVA', 'VA3'), ('AVA', 'DA1'), ('AVA', 'DA2'),
    ('AVB', 'VB1'), ('AVB', 'VB2'), ('AVB', 'VB3'), ('AVB', 'DB1'), ('AVB', 'DB2'),
    ('PVC', 'DB1'), ('PVC', 'DB2'),
    ('AVD', 'VA1'), ('AVD', 'VA2')
]

def build_bio_circuit():
    G = nx.Graph()
    all_nodes = SENSORY + INTER + MOTOR
    G.add_nodes_from(all_nodes)
    
    for u, v in GJ_EDGES:
        if u in all_nodes and v in all_nodes:
            if G.has_edge(u, v): 
                G[u][v]['weight'] += 2.0
            else: 
                G.add_edge(u, v, weight=2.0)
    
    for u, v in CHEM_EDGES:
        if u in all_nodes and v in all_nodes:
            if G.has_edge(u, v): 
                G[u][v]['weight'] += 1.0
            else: 
                G.add_edge(u, v, weight=1.0)
    
    for i in range(len(MOTOR)-1):
        G.add_edge(MOTOR[i], MOTOR[i+1], weight=1.0)
    
    print(f"Built C. elegans circuit: {len(G.nodes)} neurons, {len(G.edges)} synapses")
    return G

# --- 2. HAMILTONIANS ---
def build_hamiltonian(G, epsilon=0.0):
    nodes = sorted(list(G.nodes()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    row, col, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = d.get('weight', 1.0)
        row.extend([i, j])
        col.extend([j, i])
        data.extend([w, w])
    
    A = csc_matrix((data, (row, col)), shape=(N, N))
    D = diags(np.array(A.sum(axis=1)).flatten())
    H_kin = -(D - A)
    
    if epsilon > 0:
        np.random.seed(42)
        V = np.random.uniform(-epsilon, epsilon, N)
        H_pot = diags(V)
        H = H_kin + H_pot
    else:
        H = H_kin
    
    return H, node_to_idx

# --- 3. LIOUVILLIAN (CRN / CTQW) ---
def get_liouvillian(H, kappa, gamma, target_indices, N):
    I = eye(N, format='csc')
    H_super = -1j * (kron(I, H) - kron(H, I))
    
    rows = np.arange(N*N)
    i_idx, j_idx = rows // N, rows % N
    
    # Dephasing
    deph = diags(np.where(i_idx == j_idx, 0.0, -kappa))
    
    # Sink
    t_set = set(target_indices)
    is_i = np.isin(i_idx, list(t_set)).astype(float)
    is_j = np.isin(j_idx, list(t_set)).astype(float)
    sink = diags(-(gamma/2.0) * (is_i + is_j))
    
    return H_super + deph + sink

# --- 4. CLASSICAL RW ---
def run_crw(G, start_idxs, target_idxs, epsilon=0.0, T=100.0, num=50):
    """Classical Random Walk with sink"""
    H, node_map = build_hamiltonian(G, epsilon=epsilon)
    N = H.shape[0]
    
    # Generator: -H (Laplacian already negative) + sink
    Q = H.toarray()  # Already -(D-A)
    for t in target_idxs:
        Q[t, t] -= 1.0  # Sink rate = 1.0
    
    p0 = np.zeros(N)
    for s in start_idxs:
        p0[s] = 1.0 / len(start_idxs)
    
    # Evolve
    from scipy.linalg import expm
    times = np.linspace(0, T, num)
    probs = []
    for t in times:
        p_t = expm(Q * t) @ p0
        P_sink = 1.0 - np.sum(p_t)
        probs.append(P_sink)
    
    return max(probs)

# --- 5. CTQW / CRN ---
def run_quantum(G, kappa, start_idxs, target_idxs, epsilon=0.0, T=100.0, num=50):
    """Quantum dynamics (CTQW if kappa=0, CRN otherwise)"""
    H, node_map = build_hamiltonian(G, epsilon=epsilon)
    N = H.shape[0]
    
    L_op = get_liouvillian(H, kappa, gamma=1.0, target_indices=target_idxs, N=N)
    
    rho0 = np.zeros(N*N, dtype=complex)
    for s in start_idxs:
        rho0[s*N + s] = 1.0 / len(start_idxs)
    
    traj = expm_multiply(L_op, rho0, start=0, stop=T, num=num)
    diag = np.arange(N)*N + np.arange(N)
    probs = [1.0 - np.sum(v[diag]).real for v in traj]
    
    return max(probs)

# --- 6. MAIN ---
def main():
    G = build_bio_circuit()
    _, node_map = build_hamiltonian(G)
    
    # Start: Sensory neurons
    start_idxs = [node_map[n] for n in SENSORY if n in node_map]
    
    # Target: Only 3 key motor neurons (HONEST TEST)
    TARGET_MOTOR = ['VA1', 'VB1', 'DA1']
    target_idxs = [node_map[n] for n in TARGET_MOTOR if n in node_map]
    
    print(f"Input: {len(start_idxs)} sensory neurons")
    print(f"Target: {len(target_idxs)} key motor neurons")
    
    # Parameters
    EPS = 1.0  # FIXED: biologically plausible
    T = 100.0
    
    print(f"\n=== HONEST TEST: Epsilon={EPS}, Targets={len(target_idxs)}/{len(node_map)} ===\n")
    
    # Classical Baseline
    P_crw = run_crw(G, start_idxs, target_idxs, epsilon=EPS, T=T)
    print(f"Classical RW (CRW):        P_max = {P_crw:.4f}")
    
    # Quantum Baseline (pure coherent)
    P_ctqw = run_quantum(G, kappa=0.0, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=EPS, T=T)
    print(f"Pure Quantum (CTQW):       P_max = {P_ctqw:.4f}")
    
    # CRN scan
    print(f"\n{'Kappa':<10} | {'P_max':<10} | {'Gain vs CTQW':<15} | {'Regime'}")
    print("-" * 60)
    
    kappas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0]
    results = []
    
    for k in kappas:
        P_crn = run_quantum(G, kappa=k, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=EPS, T=T)
        gain = P_crn / P_ctqw if P_ctqw > 0 else float('inf')
        
        if k < 0.01:
            regime = "Quantum"
        elif k > 20:
            regime = "Classical"
        else:
            regime = "CRN"
        
        print(f"{k:<10.3f} | {P_crn:<10.4f} | {gain:<15.2f} | {regime}")
        results.append((k, P_crn, gain, regime))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  CRW (classical):     {P_crw:.4f}")
    print(f"  CTQW (quantum):      {P_ctqw:.4f}")
    print(f"  CRN (optimal):       {max(r[1] for r in results):.4f} at κ={[r[0] for r in results if r[1] == max(r[1] for r in results)][0]}")
    print("="*60)
    
    # Interpretation
    best_crn = max(results, key=lambda x: x[1])
    print("\nINTERPRETATION:")
    if best_crn[1] > P_crw * 1.1:
        print("  ✓ CRN shows speedup over classical!")
    elif best_crn[1] > P_ctqw * 1.5:
        print("  ✓ CRN rescues quantum from localization (robustness recovery)")
    else:
        print("  ✗ No clear advantage on this graph topology")

if __name__ == "__main__":
    main()
