#!/usr/bin/env python3
"""
C. ELEGANS TOUCH CIRCUIT: THERMAL HOPPING + FULL STRESS TESTS
==============================================================

COMPREHENSIVE VALIDATION:
1. Main test: ε=1.0, T_env=1.0 (default)
2. Thermal temperature sweep: T_env ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
3. Disorder sweep: ε ∈ [0.0, 0.5, 1.0, 2.0, 3.0]
4. Network parameter robustness: target multiplicity
5. Output: CSV for plotting + console summary

Author: CRN Team
Date: 2026-01-24
"""

import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, diags, kron, eye
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import pandas as pd

# ============================================================
# 1. CIRCUIT DATA (C. elegans Touch Withdrawal)
# ============================================================

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
    """Build C. elegans touch circuit graph"""
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
    
    return G

# ============================================================
# 2. HAMILTONIANS
# ============================================================

def build_hamiltonian_with_potential(G, epsilon=0.0, seed=42):
    """Build kinetic + potential Hamiltonian."""
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
        np.random.seed(seed)
        V = np.random.uniform(-epsilon, epsilon, N)
    else:
        V = np.zeros(N)
    
    H_full = H_kin + diags(V, format='csc')
    
    return H_full, V, node_to_idx

# ============================================================
# 3. THERMAL HOPPING (CORRECT CLASSICAL)
# ============================================================

def run_thermal_hopping(G, start_idxs, target_idxs, epsilon=1.0, T_env=1.0, T_time=100.0, seed=42):
    """Classical transport using Metropolis thermal hopping."""
    _, V, _ = build_hamiltonian_with_potential(G, epsilon=epsilon, seed=seed)
    N = len(V)
    
    Q = np.zeros((N, N), dtype=float)
    A = nx.adjacency_matrix(G).toarray()
    
    for i in range(N):
        out_rate = 0.0
        for j in range(N):
            if i == j or A[i, j] == 0:
                continue
            
            delta_E = V[j] - V[i]
            rate = np.exp(-max(0.0, delta_E) / T_env)
            
            Q[j, i] += rate
            out_rate += rate
        
        Q[i, i] = -out_rate
    
    for t in target_idxs:
        Q[t, t] -= 1.0
    
    p0 = np.zeros(N)
    for s in start_idxs:
        p0[s] = 1.0 / len(start_idxs)
    
    p_final = expm(Q * T_time) @ p0
    P_sink = 1.0 - np.sum(p_final)
    
    return max(0.0, min(1.0, P_sink))

# ============================================================
# 4. QUANTUM
# ============================================================

def get_liouvillian(H, kappa, gamma, target_indices, N):
    """Build Liouvillian."""
    I = eye(N, format='csc')
    H_super = -1j * (kron(I, H) - kron(H, I))
    
    rows = np.arange(N*N)
    i_idx, j_idx = rows // N, rows % N
    deph = diags(np.where(i_idx == j_idx, 0.0, -kappa))
    
    t_set = set(target_indices)
    is_i = np.isin(i_idx, list(t_set)).astype(float)
    is_j = np.isin(j_idx, list(t_set)).astype(float)
    sink = diags(-(gamma / 2.0) * (is_i + is_j))
    
    return H_super + deph + sink

def run_quantum(G, kappa, start_idxs, target_idxs, epsilon=0.0, T_time=100.0, num=50, seed=42):
    """Run quantum CRN or CTQW."""
    H_full, _, _ = build_hamiltonian_with_potential(G, epsilon=epsilon, seed=seed)
    N = H_full.shape[0]
    
    L_op = get_liouvillian(H_full, kappa, gamma=1.0, target_indices=target_idxs, N=N)
    
    rho0 = np.zeros(N*N, dtype=complex)
    for s in start_idxs:
        rho0[s*N + s] = 1.0 / len(start_idxs)
    
    traj = expm_multiply(L_op, rho0, start=0, stop=T_time, num=num)
    
    diag = np.arange(N)*N + np.arange(N)
    probs = [max(0.0, min(1.0, 1.0 - np.sum(v[diag]).real)) for v in traj]
    
    return max(probs)

# ============================================================
# 5. STRESS TESTS
# ============================================================

def test_main(G, start_idxs, target_idxs, epsilon=1.0, T_env=1.0, T=100.0):
    """Main test."""
    
    print(f"\n{'='*80}")
    print(f"MAIN TEST: ε={epsilon}, T_env={T_env}")
    print(f"{'='*80}\n")
    
    P_thermal = run_thermal_hopping(G, start_idxs, target_idxs, epsilon=epsilon, T_env=T_env, T_time=T)
    P_ctqw = run_quantum(G, kappa=0.0, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
    
    print(f"Thermal Hopping (T_env={T_env}): P_max = {P_thermal:.4f}")
    print(f"Pure Quantum (CTQW):            P_max = {P_ctqw:.4f}\n")
    
    kappas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0]
    
    print(f"{'Kappa':<10} | {'P_max':<10} | {'Gain/Thermal':<15} | {'Gain/CTQW':<15} | {'Regime'}")
    print("-" * 80)
    
    results = []
    
    for k in kappas:
        P_crn = run_quantum(G, kappa=k, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
        gain_thermal = P_crn / P_thermal if P_thermal > 1e-6 else float('inf')
        gain_ctqw = P_crn / P_ctqw if P_ctqw > 1e-6 else float('inf')
        
        regime = "Quantum" if k < 0.01 else ("Classical" if k > 20 else "CRN")
        
        print(f"{k:<10.3f} | {P_crn:<10.4f} | {gain_thermal:<15.2f} | {gain_ctqw:<15.2f} | {regime}")
        results.append({
            'epsilon': epsilon,
            'T_env': T_env,
            'kappa': k,
            'P_crn': P_crn,
            'P_thermal': P_thermal,
            'P_ctqw': P_ctqw,
            'gain_thermal': gain_thermal,
            'gain_ctqw': gain_ctqw,
            'regime': regime
        })
    
    best = max(results, key=lambda x: x['P_crn'])
    
    print(f"\nOPTIMAL: κ={best['kappa']:.3f}, P_max={best['P_crn']:.4f}")
    print(f"  Gain vs Thermal: {best['gain_thermal']:.2f}×")
    print(f"  Gain vs CTQW:    {best['gain_ctqw']:.2f}×")
    
    return results, best

def test_thermal_sweep(G, start_idxs, target_idxs, epsilon=1.0, T=100.0):
    """Thermal temperature sweep."""
    
    print(f"\n{'='*80}")
    print(f"STRESS TEST 1: THERMAL TEMPERATURE SWEEP (ε={epsilon})")
    print(f"{'='*80}\n")
    
    T_envs = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"{'T_env':<10} | {'P_thermal':<12} | {'P_ctqw':<12} | {'P_crn(opt)':<12} | {'Κ_opt':<8} | {'Gain':<8}")
    print("-" * 80)
    
    results = []
    
    for T_env in T_envs:
        P_thermal = run_thermal_hopping(G, start_idxs, target_idxs, epsilon=epsilon, T_env=T_env, T_time=T)
        P_ctqw = run_quantum(G, kappa=0.0, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
        
        kappas_scan = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 50.0]
        P_crn_max = 0.0
        kappa_opt = 0.0
        
        for k in kappas_scan:
            P = run_quantum(G, kappa=k, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
            if P > P_crn_max:
                P_crn_max = P
                kappa_opt = k
        
        gain = P_crn_max / P_thermal if P_thermal > 1e-6 else float('inf')
        
        print(f"{T_env:<10.1f} | {P_thermal:<12.4f} | {P_ctqw:<12.4f} | {P_crn_max:<12.4f} | {kappa_opt:<8.3f} | {gain:<8.2f}×")
        
        results.append({
            'T_env': T_env,
            'P_thermal': P_thermal,
            'P_ctqw': P_ctqw,
            'P_crn_max': P_crn_max,
            'kappa_opt': kappa_opt,
            'gain': gain
        })
    
    return results

def test_disorder_sweep(G, start_idxs, target_idxs, T=100.0, T_env=1.0):
    """Disorder strength sweep."""
    
    print(f"\n{'='*80}")
    print(f"STRESS TEST 2: DISORDER STRENGTH SWEEP (T_env={T_env})")
    print(f"{'='*80}\n")
    
    epsilons = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print(f"{'Epsilon':<10} | {'P_thermal':<12} | {'P_ctqw':<12} | {'P_crn(opt)':<12} | {'Κ_opt':<8} | {'Gain':<8}")
    print("-" * 80)
    
    results = []
    
    for eps in epsilons:
        P_thermal = run_thermal_hopping(G, start_idxs, target_idxs, epsilon=eps, T_env=T_env, T_time=T)
        P_ctqw = run_quantum(G, kappa=0.0, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=eps, T_time=T)
        
        kappas_scan = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 50.0]
        P_crn_max = 0.0
        kappa_opt = 0.0
        
        for k in kappas_scan:
            P = run_quantum(G, kappa=k, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=eps, T_time=T)
            if P > P_crn_max:
                P_crn_max = P
                kappa_opt = k
        
        gain = P_crn_max / P_thermal if P_thermal > 1e-6 else float('inf')
        
        print(f"{eps:<10.2f} | {P_thermal:<12.4f} | {P_ctqw:<12.4f} | {P_crn_max:<12.4f} | {kappa_opt:<8.3f} | {gain:<8.2f}×")
        
        results.append({
            'epsilon': eps,
            'P_thermal': P_thermal,
            'P_ctqw': P_ctqw,
            'P_crn_max': P_crn_max,
            'kappa_opt': kappa_opt,
            'gain': gain
        })
    
    return results

def test_target_sensitivity(G, start_idxs, epsilon=1.0, T=100.0, T_env=1.0):
    """Target neuron multiplicity."""
    
    print(f"\n{'='*80}")
    print(f"STRESS TEST 3: TARGET SENSITIVITY (ε={epsilon})")
    print(f"{'='*80}\n")
    
    node_map = {}
    for i, n in enumerate(sorted(list(G.nodes()))):
        node_map[n] = i
    
    target_configs = [
        (['VA1'], 1),
        (['VA1', 'VB1'], 2),
        (['VA1', 'VB1', 'DA1'], 3),
        (['VA1', 'VA2', 'VB1', 'VB2', 'DA1', 'DB1'], 6),
        ([n for n in MOTOR if n in node_map], len([n for n in MOTOR if n in node_map]))
    ]
    
    print(f"{'N_targets':<12} | {'P_thermal':<12} | {'P_ctqw':<12} | {'P_crn(opt)':<12} | {'Κ_opt':<8} | {'Gain':<8}")
    print("-" * 80)
    
    results = []
    
    for target_names, n_targets in target_configs:
        target_idxs = [node_map[n] for n in target_names if n in node_map]
        
        P_thermal = run_thermal_hopping(G, start_idxs, target_idxs, epsilon=epsilon, T_env=T_env, T_time=T)
        P_ctqw = run_quantum(G, kappa=0.0, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
        
        kappas_scan = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 50.0]
        P_crn_max = 0.0
        kappa_opt = 0.0
        
        for k in kappas_scan:
            P = run_quantum(G, kappa=k, start_idxs=start_idxs, target_idxs=target_idxs, epsilon=epsilon, T_time=T)
            if P > P_crn_max:
                P_crn_max = P
                kappa_opt = k
        
        gain = P_crn_max / P_thermal if P_thermal > 1e-6 else float('inf')
        
        print(f"{n_targets:<12} | {P_thermal:<12.4f} | {P_ctqw:<12.4f} | {P_crn_max:<12.4f} | {kappa_opt:<8.3f} | {gain:<8.2f}×")
        
        results.append({
            'n_targets': n_targets,
            'P_thermal': P_thermal,
            'P_ctqw': P_ctqw,
            'P_crn_max': P_crn_max,
            'kappa_opt': kappa_opt,
            'gain': gain
        })
    
    return results

# ============================================================
# 6. MAIN
# ============================================================

def main():
    G = build_bio_circuit()
    
    _, _, node_map = build_hamiltonian_with_potential(G, epsilon=0.0)
    start_idxs = [node_map[n] for n in SENSORY if n in node_map]
    target_idxs = [node_map[n] for n in ['VA1', 'VB1', 'DA1']]
    
    print(f"\n{'#'*80}")
    print(f"# C. ELEGANS TOUCH CIRCUIT: COMPREHENSIVE STRESS TEST")
    print(f"#{'*'*78}#")
    print(f"# Network: {len(G.nodes)} neurons, {len(G.edges)} synapses")
    print(f"# Input:   {len(start_idxs)} sensory neurons")
    print(f"# Output:  {len(target_idxs)} motor neurons (VA1, VB1, DA1)")
    print(f"{'#'*80}")
    
    main_results, best_main = test_main(G, start_idxs, target_idxs, epsilon=1.0, T_env=1.0)
    thermal_results = test_thermal_sweep(G, start_idxs, target_idxs, epsilon=1.0)
    disorder_results = test_disorder_sweep(G, start_idxs, target_idxs, T_env=1.0)
    target_results = test_target_sensitivity(G, start_idxs, epsilon=1.0, T_env=1.0)
    
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}\n")
    
    df_main = pd.DataFrame(main_results)
    df_main.to_csv('crn_elegans_main_test.csv', index=False)
    print("✓ crn_elegans_main_test.csv")
    
    df_thermal = pd.DataFrame(thermal_results)
    df_thermal.to_csv('crn_elegans_thermal_sweep.csv', index=False)
    print("✓ crn_elegans_thermal_sweep.csv")
    
    df_disorder = pd.DataFrame(disorder_results)
    df_disorder.to_csv('crn_elegans_disorder_sweep.csv', index=False)
    print("✓ crn_elegans_disorder_sweep.csv")
    
    df_target = pd.DataFrame(target_results)
    df_target.to_csv('crn_elegans_target_sensitivity.csv', index=False)
    print("✓ crn_elegans_target_sensitivity.csv")
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"DEFAULT (ε=1.0, T_env=1.0, targets=3):")
    print(f"  Thermal: {best_main['P_thermal']:.4f}")
    print(f"  CTQW:    {best_main['P_ctqw']:.4f}")
    print(f"  CRN opt: {best_main['P_crn']:.4f} at κ={best_main['kappa']:.3f}")
    print(f"  Gain:    {best_main['gain_thermal']:.2f}× (vs Thermal), {best_main['gain_ctqw']:.2f}× (vs CTQW)")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
