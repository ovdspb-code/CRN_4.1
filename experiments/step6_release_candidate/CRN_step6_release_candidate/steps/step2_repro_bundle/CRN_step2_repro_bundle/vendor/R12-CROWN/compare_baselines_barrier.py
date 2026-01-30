import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import json
from pathlib import Path


def gksl_psuccess_chain_barrier(
    N,
    gamma,
    eta,
    V,
    kappa,
    T_max,
    dt,
    initial_node=0,
    target_node=None,
):
    """GKSL wave dynamics with pure dephasing and a lossy target node.

    Returns times and cumulative success probability P_success(t) = 1 - Tr(rho(t)).
    """
    if target_node is None:
        target_node = N - 1
    V = np.asarray(V, dtype=float)
    assert V.shape == (N,)

    # Adjacency for a chain
    adj = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0

    # Hamiltonian: -gamma * A + diag(V)
    H = -gamma * adj + np.diag(V)

    I = np.eye(N)
    # Unitary part: -i[H, rho]
    L_unitary = -1j * (np.kron(H, I) - np.kron(I, H.T))

    # Pure dephasing (kills off-diagonals)
    L_deph = np.zeros((N * N, N * N), dtype=complex)
    if kappa > 0:
        np.fill_diagonal(L_deph, -kappa)
        for i in range(N):
            idx = i * N + i  # diagonal element in column-major vec
            L_deph[idx, idx] = 0.0

    # Lossy sink at target: -(eta/2){|t><t|, rho}
    P_t = np.zeros((N, N), dtype=float)
    P_t[target_node, target_node] = 1.0
    L_sink = -0.5 * eta * (np.kron(P_t, I) + np.kron(I, P_t))

    L = L_unitary + L_deph + L_sink

    # Initial density matrix |init><init|
    rho0 = np.zeros((N, N), dtype=complex)
    rho0[initial_node, initial_node] = 1.0
    rho_vec = rho0.reshape(N * N, order="F")

    steps = int(T_max / dt) + 1
    times = np.linspace(0.0, T_max, steps)

    prop = expm(L * dt)

    psucc = np.zeros_like(times, dtype=float)
    for si, _t in enumerate(times):
        # trace of rho is sum of diagonal entries
        tr = float(np.real(np.sum(rho_vec[:: N + 1])))
        psucc[si] = max(0.0, min(1.0, 1.0 - tr))
        rho_vec = prop @ rho_vec

    return times, psucc


def classical_thermal_psuccess(
    N,
    gamma,
    eta,
    V,
    T,
    T_max,
    dt,
    initial_node=0,
    target_node=None,
):
    """Continuous-time thermal random walk with Arrhenius/Metropolis-like hops."""
    if target_node is None:
        target_node = N - 1
    V = np.asarray(V, dtype=float)
    assert V.shape == (N,)

    adj = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0

    Q = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            if adj[i, j] <= 0:
                continue
            dE = V[j] - V[i]
            if dE > 0:
                rate = gamma * np.exp(-dE / T)
            else:
                rate = gamma
            Q[j, i] += rate
            Q[i, i] -= rate

    # Add absorption at target
    Q[target_node, target_node] -= eta

    P = np.zeros(N, dtype=float)
    P[initial_node] = 1.0

    steps = int(T_max / dt) + 1
    times = np.linspace(0.0, T_max, steps)

    prop = expm(Q * dt)

    psucc = np.zeros_like(times, dtype=float)
    for si, _t in enumerate(times):
        mass = float(np.sum(P))
        psucc[si] = max(0.0, min(1.0, 1.0 - mass))
        P = prop @ P

    return times, psucc


if __name__ == "__main__":
    out_png = Path("/mnt/data/Supplementary_Figure_S6_barrier_crossing.png")
    out_csv = Path("/mnt/data/Supplementary_Figure_S6_barrier_crossing.csv")
    out_json = Path("/mnt/data/Supplementary_Figure_S6_barrier_crossing_metadata.json")

    N = 10
    gamma = 1.0
    eta = 1.0
    T_max = 80.0
    dt = 0.5

    # Energy barrier profile: V=3 on nodes 3..6
    V = np.zeros(N)
    V[3:7] = 3.0

    # CRN curves
    t, ps_opt = gksl_psuccess_chain_barrier(N, gamma, eta, V, kappa=1.0, T_max=T_max, dt=dt)
    _, ps_low = gksl_psuccess_chain_barrier(N, gamma, eta, V, kappa=0.01, T_max=T_max, dt=dt)

    # Classical thermal baseline
    _, ps_cl = classical_thermal_psuccess(N, gamma, eta, V, T=1.0, T_max=T_max, dt=dt)

    plt.figure(figsize=(10, 6))
    plt.plot(t, ps_cl, "k--", linewidth=2, label="Classical thermal walk (T=1.0)")
    plt.plot(t, ps_opt, linewidth=3, label=r"CRN (optimal noise, $\kappa = 1.0$)")
    plt.plot(t, ps_low, linewidth=2, label=r"CRN (coherent/low noise, $\kappa = 0.01$)")

    plt.title("Barrier-limited baseline: kinetic advantage across an energy barrier")
    plt.xlabel("time (a.u.)")
    plt.ylabel(r"cumulative success probability $P_{success}(t)$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    # Save CSV
    import csv

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "P_classical", "P_crn_opt", "P_crn_low"])
        for ti, a, b, c in zip(t, ps_cl, ps_opt, ps_low):
            w.writerow([float(ti), float(a), float(b), float(c)])

    meta = {
        "N": N,
        "gamma": gamma,
        "eta": eta,
        "kappa_opt": 1.0,
        "kappa_low": 0.01,
        "T_classical": 1.0,
        "V": V.tolist(),
        "T_max": T_max,
        "dt": dt,
        "notes": "Illustrative chain barrier scenario; classical baseline uses Arrhenius/Metropolis-like uphill suppression.",
    }
    out_json.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
