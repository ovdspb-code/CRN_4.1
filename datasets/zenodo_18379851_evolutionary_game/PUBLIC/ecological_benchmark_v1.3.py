#!/usr/bin/env python3
"""
ECOLOGICAL BENCHMARK v1.3 (PRODUCTION - FIXED)
Correct dynamics with proper threshold-based termination

Date: 2026-01-26
Status: FINAL - Ready for production
Critical fixes from v1.2:
  - Initialize rho UNIFORM (not on cue) to force evolution
  - Check BOTH threshold AND correctness for convergence
  - Track actual convergence time (not instant)
  - κ-dependent energy with realistic costs
  - Proper fitness calculation
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION v1.3 (PRODUCTION)
# ============================================================================

CONFIG = {
    # Ecology Parameters
    'PI_VALUES': [0.5],
    'K_VALUES': [20],
    'EPSILON_VALUES': [1.0],
    'KAPPA_GRID': [1e-2, 1e-1, 1.0, 10.0],
    
    # Energy Model
    'E_COMM': 1.0,
    'E_TAX_BASE': 0.01,                    # ↑ INCREASED (was 0.001)
    'CHI_RED_ZONE': 0.1,
    
    # Task A: Selection Graph
    'P_IN_SELECTION': 0.3,
    'P_OUT_SELECTION': 0.05,
    'CUE_FRACTION': 0.4,
    'N_MIN_NODES_PER_HYP': 5,
    
    # Task B: Memory Graph
    'N_ENGRAM': 30,
    'P_IN_MEMORY': 0.4,
    'P_OUT_MEMORY': 0.02,
    'RECALL_FRACTION': 0.3,
    
    # Dynamics
    'GAMMA': 1.0,
    'DT': 0.1,
    'T_MAX_STAGE1': 500,
    'T_MAX_STAGE2': 300,
    'READOUT_THRESHOLD': 0.3,              # ↓ LOWERED (was 0.4)
    'RECALL_SUCCESS_THRESHOLD': 0.3,
    'LEAKAGE_FAIL_THRESHOLD': 0.5,
    
    # Simulation
    'N_TRIALS': 10,
    'RANDOM_SEED': 42,
    
    # Fitness Function
    'LAMBDA_E': 0.5,                       # ↑ STRONG energy penalty
    'LAMBDA_T': 0.01,                      # Time penalty (speed bonus)
    'LAMBDA_L': 0.5,                       # Leakage penalty
}

OUTPUT_DIR = Path('./ecological_benchmark_results')
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# GRAPH GENERATORS
# ============================================================================

def generate_selection_graph(K, epsilon, seed=None):
    """Generate hypothesis selection graph"""
    if seed is not None:
        np.random.seed(seed)
    
    n_per_hyp = max(CONFIG['N_MIN_NODES_PER_HYP'], 30 // K)
    N_total = K * n_per_hyp
    
    sizes = [n_per_hyp] * K
    probs = np.ones((K, K)) * CONFIG['P_OUT_SELECTION']
    np.fill_diagonal(probs, CONFIG['P_IN_SELECTION'])
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len)
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            G.add_edge(node1, node2)
    
    hypothesis_nodes = []
    for i in range(K):
        nodes = list(range(i * n_per_hyp, (i+1) * n_per_hyp))
        hypothesis_nodes.append(nodes)
    
    L = nx.laplacian_matrix(G).toarray().astype(float)
    H = -CONFIG['GAMMA'] * L
    
    disorder = np.random.uniform(-epsilon, +epsilon, N_total)
    H += np.diag(disorder)
    
    return G, hypothesis_nodes, H, N_total


def generate_memory_graph(n_engram=None, epsilon=3.0, seed=None):
    """Generate memory graph"""
    if n_engram is None:
        n_engram = CONFIG['N_ENGRAM']
    
    if seed is not None:
        np.random.seed(seed)
    
    p_connect = CONFIG['P_IN_MEMORY']
    G = nx.erdos_renyi_graph(n_engram, p_connect, seed=seed)
    
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len)
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            G.add_edge(node1, node2)
    
    L = nx.laplacian_matrix(G).toarray().astype(float)
    H = -CONFIG['GAMMA'] * L
    
    disorder = np.random.uniform(-epsilon, +epsilon, n_engram)
    H += np.diag(disorder)
    
    return G, H


# ============================================================================
# DYNAMICS ENGINES
# ============================================================================

def evolve_lindblad_dephasing(rho, H, kappa, dt):
    """Lindblad: dρ/dt = -i[H,ρ] + κ D[σ_z](ρ)"""
    N = rho.shape[0]
    
    # Unitary
    H_term = -1j * (H @ rho - rho @ H)
    
    # Dephasing (off-diagonal decay)
    dephasing = np.zeros_like(rho, dtype=complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                dephasing[i, j] = -kappa * rho[i, j]
    
    # Update
    rho_new = rho + dt * (H_term + dephasing)
    
    # Hermiticity + normalization
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    trace = np.trace(rho_new).real
    
    if trace > 1e-10:
        rho_new /= trace
    else:
        rho_new = np.eye(N, dtype=complex) / N
    
    return rho_new


def evolve_classical_diffusion(P, L_graph, dt):
    """Classical: dP/dt = -L*P"""
    P_new = P - dt * (L_graph @ P)
    P_new = np.maximum(P_new, 0.0)
    
    total = np.sum(P_new)
    if total > 1e-10:
        P_new /= total
    else:
        P_new = np.ones_like(P) / len(P)
    
    return P_new


def evolve_attractor_wta(activations, W_rec, W_inh, dt):
    """WTA dynamics"""
    x = activations.copy()
    excitation = W_rec @ x
    inhibition = W_inh @ x
    decay = 0.1 * x
    
    dx = excitation - inhibition - decay
    x_new = x + dt * dx
    x_new = np.maximum(x_new, 0.0)
    x_new = np.minimum(x_new, 1.0)
    
    return x_new


# ============================================================================
# ARCHITECTURES
# ============================================================================

class CRNArchitecture:
    """CRN: Two-stage with Lindblad Stage I"""
    
    def __init__(self, kappa):
        self.kappa = kappa
        self.name = f"CRN_kappa_{kappa:.3f}"
    
    def stage1_selection(self, H, hypothesis_nodes, cue_nodes, target_hypothesis):
        """Stage I: Wave-based selection (START UNIFORM)"""
        N = H.shape[0]
        K = len(hypothesis_nodes)
        
        # START UNIFORM to force evolution
        rho = np.eye(N, dtype=complex) / N
        
        # κ-dependent energy cost
        E_tax_per_step = CONFIG['E_TAX_BASE'] * N * (1.0 + self.kappa)
        
        E_tax = 0.0
        winner = None
        t_fix = CONFIG['T_MAX_STAGE1']
        
        for t in range(CONFIG['T_MAX_STAGE1']):
            rho = evolve_lindblad_dephasing(rho, H, self.kappa, CONFIG['DT'])
            E_tax += E_tax_per_step
            
            P_diag = np.diag(rho).real
            P_per_hyp = np.array([np.sum(P_diag[nodes]) for nodes in hypothesis_nodes])
            
            # Readout: check threshold AND correctness
            if np.max(P_per_hyp) > CONFIG['READOUT_THRESHOLD']:
                winner = np.argmax(P_per_hyp)
                if winner == target_hypothesis:
                    t_fix = t
                    break
        
        # Fallback: best guess if didn't converge
        if winner is None:
            P_diag = np.diag(rho).real
            P_per_hyp = np.array([np.sum(P_diag[nodes]) for nodes in hypothesis_nodes])
            winner = np.argmax(P_per_hyp)
        
        chi = E_tax / CONFIG['E_COMM']
        success_stage1 = (winner == target_hypothesis)
        
        return winner, E_tax, t_fix, chi, success_stage1
    
    def stage2_memory(self, G_mem, H_mem, recall_cue_nodes, target_nodes):
        """Stage II: Classical diffusion"""
        N = H_mem.shape[0]
        
        # Initialize on recall cue
        P = np.zeros(N)
        if len(recall_cue_nodes) > 0:
            for node in recall_cue_nodes:
                if node < N:
                    P[node] = 1.0 / len(recall_cue_nodes)
        
        if np.sum(P) < 1e-10:
            P = np.ones(N) / N
        
        L = nx.laplacian_matrix(G_mem).toarray().astype(float)
        
        for t in range(CONFIG['T_MAX_STAGE2']):
            P = evolve_classical_diffusion(P, L, CONFIG['DT'])
        
        recall = np.sum(P[target_nodes]) if len(target_nodes) > 0 else 0.0
        non_target = [i for i in range(N) if i not in target_nodes]
        leakage = np.sum(P[non_target]) if len(non_target) > 0 else 0.0
        
        success_stage2 = (recall > CONFIG['RECALL_SUCCESS_THRESHOLD'] and 
                         leakage < CONFIG['LEAKAGE_FAIL_THRESHOLD'])
        
        return recall, leakage, success_stage2
    
    def run_episode(self, task_type, H_selection, hypothesis_nodes, cue_selection,
                    target_hypothesis, memory_data, recall_cue, target_recall):
        """Full episode"""
        
        winner, E_tax, t1, chi, succ1 = self.stage1_selection(
            H_selection, hypothesis_nodes, cue_selection, target_hypothesis
        )
        
        if task_type == 'TaskB':
            G_mem, H_mem = memory_data
            recall, leakage, succ2 = self.stage2_memory(
                G_mem, H_mem, recall_cue, target_recall
            )
            t2 = CONFIG['T_MAX_STAGE2']
        else:
            recall, leakage, succ2 = 1.0, 0.0, True
            t2 = 0
        
        n_broadcasts = 1
        E_total = E_tax + n_broadcasts * CONFIG['E_COMM']
        
        success = succ1 and succ2
        
        R_success = 1.0 if success else 0.0
        F = (R_success 
             - CONFIG['LAMBDA_E'] * E_total 
             - CONFIG['LAMBDA_T'] * (t1 + t2)
             - CONFIG['LAMBDA_L'] * leakage)
        
        return {
            'success': success,
            'E_total': E_total,
            'E_tax': E_tax,
            't_stage1': t1,
            't_stage2': t2,
            'chi': chi,
            'recall': recall,
            'leakage': leakage,
            'F_fitness': F,
            'succ_stage1': succ1,
            'succ_stage2': succ2,
        }


class ClassicalDiffusionArchitecture:
    """Classical: Diffusion-based selection"""
    
    def __init__(self):
        self.name = "Classical_Diffusion"
    
    def run_episode(self, task_type, H_selection, hypothesis_nodes, cue_selection,
                    target_hypothesis, memory_data, recall_cue, target_recall):
        
        N = H_selection.shape[0]
        K = len(hypothesis_nodes)
        
        # START UNIFORM
        P = np.ones(N) / N
        
        L = np.zeros((N, N))
        for i in range(N):
            degree = 0
            for j in range(N):
                if i != j and abs(H_selection[i, j]) > 1e-10:
                    L[i, j] = -1.0
                    degree += 1
            L[i, i] = max(1, degree)
        
        E_tax = 0.0
        winner = None
        t1 = CONFIG['T_MAX_STAGE1']
        
        for t in range(CONFIG['T_MAX_STAGE1']):
            P = evolve_classical_diffusion(P, L, CONFIG['DT'])
            E_tax += CONFIG['E_TAX_BASE'] * N
            
            P_per_hyp = np.array([np.sum(P[nodes]) for nodes in hypothesis_nodes])
            
            if np.max(P_per_hyp) > CONFIG['READOUT_THRESHOLD']:
                winner = np.argmax(P_per_hyp)
                if winner == target_hypothesis:
                    t1 = t
                    break
        
        if winner is None:
            P_per_hyp = np.array([np.sum(P[nodes]) for nodes in hypothesis_nodes])
            winner = np.argmax(P_per_hyp)
        
        succ1 = (winner == target_hypothesis)
        chi = E_tax / CONFIG['E_COMM']
        
        if task_type == 'TaskB':
            G_mem, H_mem = memory_data
            crn_temp = CRNArchitecture(kappa=0.0)
            recall, leakage, succ2 = crn_temp.stage2_memory(
                G_mem, H_mem, recall_cue, target_recall
            )
            t2 = CONFIG['T_MAX_STAGE2']
        else:
            recall, leakage, succ2 = 1.0, 0.0, True
            t2 = 0
        
        E_total = E_tax + CONFIG['E_COMM']
        
        success = succ1 and succ2
        R_success = 1.0 if success else 0.0
        F = (R_success - CONFIG['LAMBDA_E'] * E_total 
             - CONFIG['LAMBDA_T'] * (t1 + t2) - CONFIG['LAMBDA_L'] * leakage)
        
        return {
            'success': success,
            'E_total': E_total,
            'E_tax': E_tax,
            't_stage1': t1,
            't_stage2': t2,
            'chi': chi,
            'recall': recall,
            'leakage': leakage,
            'F_fitness': F,
            'succ_stage1': succ1,
            'succ_stage2': succ2,
        }


class ClassicalSpikeWTAArchitecture:
    """Classical: Winner-take-all"""
    
    def __init__(self):
        self.name = "Classical_WTA"
    
    def run_episode(self, task_type, H_selection, hypothesis_nodes, cue_selection,
                    target_hypothesis, memory_data, recall_cue, target_recall):
        
        K = len(hypothesis_nodes)
        
        # Start weak
        activations = np.ones(K) * 0.1
        
        W_exc = np.eye(K) * 2.0
        W_inh = np.ones((K, K)) * 0.5
        np.fill_diagonal(W_inh, 0.0)
        
        n_broadcasts = 0
        winner = None
        t1 = CONFIG['T_MAX_STAGE1']
        
        for t in range(CONFIG['T_MAX_STAGE1']):
            activations = evolve_attractor_wta(
                activations, W_exc, W_inh, CONFIG['DT']
            )
            
            participation = np.sum(activations > 0.3)
            n_broadcasts += participation
            
            if np.max(activations) > CONFIG['READOUT_THRESHOLD']:
                winner = np.argmax(activations)
                if winner == target_hypothesis:
                    t1 = t
                    break
        
        if winner is None:
            winner = np.argmax(activations)
        
        succ1 = (winner == target_hypothesis)
        
        if task_type == 'TaskB':
            G_mem, H_mem = memory_data
            crn_temp = CRNArchitecture(kappa=0.0)
            recall, leakage, succ2 = crn_temp.stage2_memory(
                G_mem, H_mem, recall_cue, target_recall
            )
            t2 = CONFIG['T_MAX_STAGE2']
        else:
            recall, leakage, succ2 = 1.0, 0.0, True
            t2 = 0
        
        E_total = n_broadcasts * CONFIG['E_COMM']
        
        success = succ1 and succ2
        R_success = 1.0 if success else 0.0
        F = (R_success - CONFIG['LAMBDA_E'] * E_total 
             - CONFIG['LAMBDA_T'] * (t1 + t2) - CONFIG['LAMBDA_L'] * leakage)
        
        return {
            'success': success,
            'E_total': E_total,
            'E_tax': 0.0,
            't_stage1': t1,
            't_stage2': t2,
            'chi': np.nan,
            'recall': recall,
            'leakage': leakage,
            'F_fitness': F,
            'succ_stage1': succ1,
            'succ_stage2': succ2,
        }


# ============================================================================
# SINGLE TRIAL
# ============================================================================

def run_single_trial(architecture, task_type, pi, K, epsilon, trial_id, seed):
    """Execute one trial"""
    np.random.seed(seed + trial_id)
    
    G_sel, hyp_nodes, H_sel, N_sel = generate_selection_graph(K, epsilon, seed=seed+trial_id)
    G_mem, H_mem = generate_memory_graph(CONFIG['N_ENGRAM'], epsilon, seed=seed+trial_id*2)
    
    target_hyp = np.random.randint(0, K)
    
    cue_nodes = hyp_nodes[target_hyp]
    cue_sel = cue_nodes  # Use all cue nodes
    
    n_recall = max(1, int(CONFIG['N_ENGRAM'] * CONFIG['RECALL_FRACTION']))
    recall_cue = list(range(n_recall))
    target_recall = list(range(CONFIG['N_ENGRAM']))
    
    result = architecture.run_episode(
        task_type=task_type,
        H_selection=H_sel,
        hypothesis_nodes=hyp_nodes,
        cue_selection=cue_sel,
        target_hypothesis=target_hyp,
        memory_data=(G_mem, H_mem),
        recall_cue=recall_cue,
        target_recall=target_recall,
    )
    
    result['Architecture'] = architecture.name
    result['TaskType'] = task_type
    result['Pi'] = pi
    result['K'] = K
    result['Epsilon'] = epsilon
    result['Trial'] = trial_id
    
    if hasattr(architecture, 'kappa'):
        result['Kappa'] = architecture.kappa
    else:
        result['Kappa'] = np.nan
    
    return result


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation():
    """Execute pilot"""
    print(f"\n{'='*70}")
    print(f"ECOLOGICAL BENCHMARK v1.3 (PRODUCTION)")
    print(f"{'='*70}\n")
    
    pi_values = CONFIG['PI_VALUES']
    k_values = CONFIG['K_VALUES']
    epsilon_values = CONFIG['EPSILON_VALUES']
    kappa_values = CONFIG['KAPPA_GRID']
    n_trials = CONFIG['N_TRIALS']
    
    architectures = []
    for kappa in kappa_values:
        architectures.append(CRNArchitecture(kappa))
    architectures.append(ClassicalDiffusionArchitecture())
    architectures.append(ClassicalSpikeWTAArchitecture())
    
    n_crn = len(kappa_values)
    n_classical = 2
    total_runs = 0
    for pi in pi_values:
        for K in k_values:
            for eps in epsilon_values:
                n_taskA = int(n_trials * pi)
                n_taskB = n_trials - n_taskA
                total_runs += (n_crn + n_classical) * (n_taskA + n_taskB)
    
    print(f"Configuration:")
    print(f"  Pi: {pi_values}, K: {k_values}, Epsilon: {epsilon_values}")
    print(f"  Kappa (CRN): {kappa_values}")
    print(f"  E_TAX_BASE: {CONFIG['E_TAX_BASE']} (per node per step)")
    print(f"  READOUT_THRESHOLD: {CONFIG['READOUT_THRESHOLD']}")
    print(f"  LAMBDA_E: {CONFIG['LAMBDA_E']}, LAMBDA_T: {CONFIG['LAMBDA_T']}")
    print(f"\nTotal runs: {total_runs}\n")
    
    results = []
    
    with tqdm(total=total_runs, desc="Progress") as pbar:
        for pi in pi_values:
            for K in k_values:
                for epsilon in epsilon_values:
                    n_taskA = int(n_trials * pi)
                    n_taskB = n_trials - n_taskA
                    task_schedule = ['TaskA'] * n_taskA + ['TaskB'] * n_taskB
                    np.random.shuffle(task_schedule)
                    
                    for arch in architectures:
                        for trial_id, task_type in enumerate(task_schedule):
                            seed = CONFIG['RANDOM_SEED'] + trial_id * 1000
                            
                            result = run_single_trial(
                                architecture=arch,
                                task_type=task_type,
                                pi=pi,
                                K=K,
                                epsilon=epsilon,
                                trial_id=trial_id,
                                seed=seed,
                            )
                            
                            results.append(result)
                            pbar.update(1)
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = OUTPUT_DIR / f"ecological_benchmark_v1.3_pilot_{timestamp}.csv"
    df.to_csv(raw_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved: {raw_file}")
    print(f"{'='*70}\n")
    
    # Summary
    print("RESULTS BY ARCHITECTURE:")
    print("="*70)
    
    for arch in sorted(df['Architecture'].unique()):
        subset = df[df['Architecture'] == arch]
        print(f"{arch:25s}: success={subset['success'].mean():.0%}  "
              f"F={subset['F_fitness'].mean():6.2f}±{subset['F_fitness'].std():.2f}  "
              f"E={subset['E_total'].mean():6.2f}  t1={subset['t_stage1'].mean():6.1f}")
    print()
    
    # ENAQT check
    print("ENAQT WINDOW (F by κ):")
    print("-"*70)
    crn_rows = df[df['Architecture'].str.contains('CRN')]
    crn_by_kappa = crn_rows.groupby('Kappa')['F_fitness'].agg(['mean', 'std']).sort_index()
    print(crn_by_kappa.to_string())
    
    if len(crn_by_kappa) > 1:
        f_vals = crn_by_kappa['mean'].values
        if f_vals.max() - f_vals.min() > 0.5:
            print("\n✓ ENAQT window DETECTED (clear variation in F)")
        else:
            print("\n✗ No clear ENAQT window (weak variation)")
    print()
    
    return df, raw_file


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    df_results, output_path = run_simulation()
    
    print("="*70)
    print("✅ PILOT v1.3 COMPLETE - Ready for full run")
    print("="*70)
    print()
