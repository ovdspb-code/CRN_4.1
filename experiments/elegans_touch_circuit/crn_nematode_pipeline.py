"""
PROJECT NEMATODE: HARDCODED TOUCH CIRCUIT
=========================================
Источник: Chalfie et al. (1985), "The Neural Circuit for Touch Sensitivity in C. elegans"
Точная реконструкция связей переднего (Anterior) и заднего (Posterior) рефлекса.

Это "золотой стандарт" проверки: мы не генерируем случайный граф, 
а берем эволюционно сформированную цепь.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, diags, kron, eye
from scipy.sparse.linalg import expm_multiply

# --- 1. ВШИТЫЕ ДАННЫЕ КОННЕКТОМА (Touch Circuit) ---
# Sensory Neurons (Вход)
SENSORY = ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM']

# Interneurons (Командный центр - "Горлышко")
INTER = ['AVD', 'PVC', 'AVA', 'AVB'] 

# Motor Neurons (Выход) - только часть для примера
MOTOR = [
    'VA1', 'VA2', 'VA3', 'VA4', 'DA1', 'DA2', 'DA3', # Backward motion
    'VB1', 'VB2', 'VB3', 'VB4', 'DB1', 'DB2', 'DB3'  # Forward motion
]

# Список связей (Source, Target, Type/Weight)
# Gap Junctions (Электрические) - двунаправленные
GJ_EDGES = [
    # Sensory <-> Inter
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'), # Anterior touch -> Backward command
    ('PLML', 'PVC'), ('PLMR', 'PVC'), ('PVM', 'PVC'), # Posterior touch -> Forward command
    
    # Inter <-> Inter (Cross-inhibition / coordination)
    ('AVD', 'AVA'), ('PVC', 'AVB'), 
    
    # Inter <-> Motor (Drive)
    ('AVA', 'VA1'), ('AVA', 'DA1'),
    ('AVB', 'VB1'), ('AVB', 'DB1'),
    
    # Sensory <-> Sensory (Coupling)
    ('ALML', 'ALMR'), ('PLML', 'PLMR')
]

# Chemical Synapses (Химические) - добавим их как симметричные "hopping" для упрощения модели
CHEM_EDGES = [
    # Sensory -> Inter
    ('ALML', 'AVD'), ('ALMR', 'AVD'), ('AVM', 'AVD'),
    ('ALML', 'PVC'), ('ALMR', 'PVC'), # Weak connection
    ('PLML', 'PVC'), ('PLMR', 'PVC'),
    ('PLML', 'AVD'), ('PLMR', 'AVD'), # Weak connection
    
    # Inter -> Motor (Massive divergence)
    ('AVA', 'VA1'), ('AVA', 'VA2'), ('AVA', 'VA3'), ('AVA', 'DA1'), ('AVA', 'DA2'),
    ('AVB', 'VB1'), ('AVB', 'VB2'), ('AVB', 'VB3'), ('AVB', 'DB1'), ('AVB', 'DB2'),
    ('PVC', 'DB1'), ('PVC', 'DB2'),
    ('AVD', 'VA1'), ('AVD', 'VA2')
]

def build_bio_circuit():
    G = nx.Graph()
    
    # Добавляем все узлы
    all_nodes = SENSORY + INTER + MOTOR
    G.add_nodes_from(all_nodes)
    
    # Добавляем связи (Gap Junctions - вес 2.0, Chemical - вес 1.0)
    for u, v in GJ_EDGES:
        if u in all_nodes and v in all_nodes:
            if G.has_edge(u, v): G[u][v]['weight'] += 2.0
            else: G.add_edge(u, v, weight=2.0)
            
    for u, v in CHEM_EDGES:
        if u in all_nodes and v in all_nodes:
            if G.has_edge(u, v): G[u][v]['weight'] += 1.0
            else: G.add_edge(u, v, weight=1.0)
            
    # Добавляем "цепочку" моторных нейронов (они связаны друг с другом)
    for i in range(len(MOTOR)-1):
        # Простая линейная связь вдоль тела
        G.add_edge(MOTOR[i], MOTOR[i+1], weight=1.0)
            
    print(f"Built C. elegans Touch Circuit: {len(G.nodes)} neurons, {len(G.edges)} synapses.")
    return G

# --- 2. CRN ENGINE (Тот же самый) ---
def build_hamiltonian(G, epsilon_anderson=0.0):
    nodes = sorted(list(G.nodes()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    row, col, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = d.get('weight', 1.0)
        row.extend([i, j]); col.extend([j, i]); data.extend([w, w])
        
    A = csc_matrix((data, (row, col)), shape=(N, N))
    D = diags(np.array(A.sum(axis=1)).flatten())
    H_kin = -1.0 * (D - A) 
    
    if epsilon_anderson > 0:
        np.random.seed(42) # Фиксируем болезнь
        energies = np.random.uniform(-epsilon_anderson, epsilon_anderson, N)
        H_pot = diags(energies)
        H_net = H_kin + H_pot
    else:
        H_net = H_kin
        
    return H_net, node_to_idx

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

def run_simulation():
    G = build_bio_circuit()
    
    # Map Names to Indices
    _, node_map = build_hamiltonian(G)
    
    # Start at Sensory (Touch)
    start_idxs = [node_map[n] for n in SENSORY if n in node_map]
    # End at Motor (Muscle contraction)
    target_idxs = [node_map[n] for n in MOTOR if n in node_map]
    
    print(f"Input: {len(start_idxs)} sensory neurons")
    print(f"Target: {len(target_idxs)} motor neurons")
    
    # Run Stress Test with REAL Anderson Disorder
    # Для малого графа (30 узлов) локализация требует ОЧЕНЬ сильного беспорядка.
    # Ставим Epsilon = 10.0 (так как веса связей теперь 1.0-2.0, барьеры должны быть выше)
    EPS = 15.0 
    
    H, _ = build_hamiltonian(G, epsilon_anderson=EPS)
    N = H.shape[0]
    
    rho0 = np.zeros(N*N, dtype=complex)
    for s in start_idxs: rho0[s*N + s] = 1.0/len(start_idxs)
    
    print(f"\n=== C. ELEGANS TOUCH REFLEX (Disorder Eps={EPS}) ===")
    print(f"{'Kappa':<8} | {'P_max':<10} | {'Regime'}")
    print("-" * 35)
    
    # Скан по каппе
    kappas = [0.001, 0.1, 1.0, 5.0, 10.0, 50.0]
    
    for k in kappas:
        L_op = get_liouvillian(H, k, 1.0, target_idxs, N)
        # Увеличим время, так как цепь длинная
        traj = expm_multiply(L_op, rho0, start=0, stop=100.0, num=50)
        
        diag = np.arange(N)*N + np.arange(N)
        probs = [1.0 - np.sum(v[diag]).real for v in traj]
        
        mode = "Quantum" if k < 0.01 else ("Classical" if k > 20 else "CRN")
        print(f"{k:<8} | {max(probs):.4f}     | {mode}")

if __name__ == "__main__":
    run_simulation()
    # В run_simulation()
EPS = 1.0  # FIX 1
TARGET_MOTOR = ['VA1', 'VB1', 'DA1']  # FIX 2: только 3 ключевых
target_idxs = [node_map[n] for n in TARGET_MOTOR if n in node_map]

kappas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 50.0]  # FIX 3
