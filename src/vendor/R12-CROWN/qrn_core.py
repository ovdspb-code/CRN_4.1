"""QRN Core Simulation Module (Release 4.0)

Portable GKSL (Lindblad) master-equation engine on a graph.
Dependencies: numpy, scipy

Design notes (aligned with Release-4 Roadmap):
- The readout sink is implemented as an explicit additional state |s>, so the
  dynamics are trace-preserving and a stationary state exists (needed for a
  meaningful Liouvillian gap).
- A minimal nonlinear (threshold) sink is supported via a population-dependent
  rate eta(I) based on rho_tt.
- Vectorization uses column-major convention (vec stacks columns), consistent
  with the standard identities used in superoperator construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


class TopologyGenerator:
    """Lightweight (numpy-only) generators for common graph topologies.

    Returned matrices are *raw* adjacency Hamiltonians on the hypothesis
    subspace (size N×N) with off-diagonal entries -1 where an edge exists.
    The overall hopping scale is applied later via multiplication by gamma.

    Notes:
    - Implementations are intentionally simple and deterministic when a seed
      is provided.
    - No heavy graph libraries (networkx) are required (Colab-friendly).
    """

    @staticmethod
    def chain(N: int) -> np.ndarray:
        adj = np.zeros((N, N), dtype=int)
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1
        return (-1.0 * adj).astype(complex)

    @staticmethod
    def lollipop(N: int, ring_size: int = 6) -> np.ndarray:
        """Ring-lollipop graph: a ring of size m connected to a path tail.

        Nodes 0..(m-1) form a ring, and node (m-1) is connected to a chain
        (m-1)-m-(m+1)-...-(N-1). This simple topology is a standard toy
        construction that can exhibit interference/dark-state trapping in the
        coherent limit, and a noise-assisted transport optimum under dephasing.

        Parameters
        ----------
        N:
            Total number of nodes.
        ring_size:
            Size of the ring subgraph (m). Must satisfy 3 <= m <= N.
        """
        N = int(N)
        m = int(ring_size)
        if N < 4:
            return TopologyGenerator.chain(N)
        if m < 3:
            m = 3
        if m > N:
            m = N

        adj = np.zeros((N, N), dtype=int)

        # Ring on 0..m-1
        for i in range(m):
            j = (i + 1) % m
            adj[i, j] = adj[j, i] = 1

        # Tail chain from m-1 to N-1
        for i in range(m - 1, N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1

        return (-1.0 * adj).astype(complex)

    @staticmethod
    def watts_strogatz(
        N: int,
        k_neighbors: int = 4,
        p_rewire: float = 0.3,
        seed: Optional[int] = 1,
    ) -> np.ndarray:
        """Small-world ring lattice + rewiring (undirected).

        k_neighbors should be even and < N.
        """
        if N < 3:
            return TopologyGenerator.chain(N)

        k_neighbors = int(k_neighbors)
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if k_neighbors >= N:
            k_neighbors = N - 1
        if k_neighbors % 2 == 1:
            k_neighbors += 1

        rng = np.random.default_rng(seed)

        adj = np.zeros((N, N), dtype=int)
        half = k_neighbors // 2
        # Regular ring lattice
        for i in range(N):
            for j in range(1, half + 1):
                a = i
                b = (i + j) % N
                adj[a, b] = adj[b, a] = 1

        # Rewire edges (only "forward" ones to avoid double-processing)
        for i in range(N):
            for j in range(1, half + 1):
                if rng.random() < p_rewire:
                    old = (i + j) % N
                    # Pick a new target avoiding self-loops and duplicates.
                    candidates = np.ones(N, dtype=bool)
                    candidates[i] = False
                    candidates[adj[i] == 1] = False
                    if not np.any(candidates):
                        continue
                    new = int(rng.choice(np.where(candidates)[0]))
                    # Remove old, add new
                    adj[i, old] = adj[old, i] = 0
                    adj[i, new] = adj[new, i] = 1

        return (-1.0 * adj).astype(complex)

    @staticmethod
    def modular(
        N: int,
        n_clusters: int = 2,
        p_in: float = 0.8,
        p_out: float = 0.05,
        seed: Optional[int] = 1,
    ) -> np.ndarray:
        """Simple undirected stochastic block model (SBM) with equal clusters."""
        n_clusters = int(max(1, n_clusters))
        if n_clusters == 1:
            return TopologyGenerator.fully_connected(N)

        rng = np.random.default_rng(seed)

        # Cluster assignment (nearly equal sizes).
        reps = int(np.ceil(N / n_clusters))
        clusters = np.repeat(np.arange(n_clusters), reps)[:N]
        adj = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(i + 1, N):
                prob = p_in if clusters[i] == clusters[j] else p_out
                if rng.random() < prob:
                    adj[i, j] = adj[j, i] = 1
        return (-1.0 * adj).astype(complex)

    @staticmethod
    def fully_connected(N: int) -> np.ndarray:
        adj = np.ones((N, N), dtype=int)
        np.fill_diagonal(adj, 0)
        return (-1.0 * adj).astype(complex)


def _vec_col_major(rho: np.ndarray) -> np.ndarray:
    return rho.reshape((-1,), order="F")


def _mat_col_major(vec: np.ndarray, d: int) -> np.ndarray:
    return vec.reshape((d, d), order="F")


@dataclass
class SinkConfig:
    target: int
    sink: int
    threshold_type: str  # 'linear' or 'threshold'
    eta_base: float
    eta_high: float
    I_th: float
    steepness: float


class QRNSystem:
    """Open-system wave dynamics on a graph with node-local dephasing and a sink."""

    def __init__(
        self,
        N: int,
        topology: str = "chain",
        gamma: float = 1.0,
        potential: Optional[Sequence[float]] = None,
        topology_kwargs: Optional[dict] = None,
    ):
        if N < 2:
            raise ValueError("N must be >= 2")
        self.N = int(N)
        self.gamma = float(gamma)
        self.topology = str(topology)
        self.topology_kwargs = dict(topology_kwargs) if topology_kwargs is not None else {}

        # System dimension expands to N+1 after sink insertion.
        self.d = self.N
        self.sink_cfg: Optional[SinkConfig] = None

        self.H = self._build_hamiltonian(self.topology, self.d)
        self.V = np.zeros((self.d, self.d), dtype=complex)
        if potential is not None:
            self.set_potential(potential)

        # jump_ops: list of (L, rate)
        self.jump_ops: List[Tuple[np.ndarray, float]] = []

    def _build_hamiltonian(self, topology: str, d: int) -> np.ndarray:
        """Hamiltonian on hypothesis nodes; sink (if present) is left uncoupled."""
        H = np.zeros((d, d), dtype=complex)
        n = self.N

        # --- Deterministic / parameterized generators (Release-4 supplement) ---
        if topology == "chain":
            raw = TopologyGenerator.chain(n)
            H[:n, :n] = self.gamma * raw
        elif topology == "lollipop":
            raw = TopologyGenerator.lollipop(
                n, ring_size=int(self.topology_kwargs.get("ring_size", 6))
            )
            H[:n, :n] = self.gamma * raw
        elif topology == "watts_strogatz":
            raw = TopologyGenerator.watts_strogatz(
                n,
                k_neighbors=int(self.topology_kwargs.get("k_neighbors", 4)),
                p_rewire=float(self.topology_kwargs.get("p_rewire", 0.3)),
                seed=self.topology_kwargs.get("seed", 1),
            )
            H[:n, :n] = self.gamma * raw
        elif topology == "modular":
            raw = TopologyGenerator.modular(
                n,
                n_clusters=int(self.topology_kwargs.get("n_clusters", 2)),
                p_in=float(self.topology_kwargs.get("p_in", 0.8)),
                p_out=float(self.topology_kwargs.get("p_out", 0.05)),
                seed=self.topology_kwargs.get("seed", 1),
            )
            H[:n, :n] = self.gamma * raw
        elif topology == "ring":
            for i in range(n):
                H[i, (i + 1) % n] = H[(i + 1) % n, i] = -self.gamma
        elif topology == "star":
            hub = 0
            for i in range(1, n):
                H[hub, i] = H[i, hub] = -self.gamma
        elif topology == "fully_connected":
            H[:n, :n] = -self.gamma
            np.fill_diagonal(H[:n, :n], 0.0)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        return H

    def set_potential(self, V: Sequence[float]) -> None:
        """Set diagonal potential on hypothesis nodes (length N)."""
        V = np.asarray(V, dtype=float)
        if V.shape[0] != self.N:
            raise ValueError(f"Potential must have length N={self.N}")

        self.V = np.zeros((self.d, self.d), dtype=complex)
        for i in range(self.N):
            self.V[i, i] = float(V[i])

    def add_dephasing(self, kappa: float) -> None:
        """Add uniform node-local dephasing on hypothesis nodes.

        Lindblad operators: L_k = |k><k| with rate kappa.
        Calling this method overwrites previous dephasing operators.
        """
        kappa = float(kappa)
        if kappa < 0:
            raise ValueError("kappa must be non-negative")

        self.jump_ops = []
        for k in range(self.N):
            P_k = np.zeros((self.d, self.d), dtype=complex)
            P_k[k, k] = 1.0
            self.jump_ops.append((P_k, kappa))

    def add_sink(
        self,
        target_node: int,
        eta_base: float,
        threshold_type: str = "linear",
        I_th: float = 0.5,
        eta_high: float = 5.0,
        steepness: float = 20.0,
    ) -> None:
        """Add a sink as an explicit (N+1)-th state.

        Jump operator: J = |s><t|.

        threshold_type:
          - 'linear': eta(I) = eta_base
          - 'threshold': eta(I) = eta_base + (eta_high-eta_base) * sigmoid(steepness*(I-I_th))
            where I = rho_tt.
        """
        target_node = int(target_node)
        if not (0 <= target_node < self.N):
            raise ValueError(f"target_node must be in [0, N-1]; got {target_node}")
        if eta_base <= 0:
            raise ValueError("eta_base must be positive")
        if threshold_type not in {"linear", "threshold"}:
            raise ValueError("threshold_type must be 'linear' or 'threshold'")

        # Expand Hilbert space by one sink state if not already expanded.
        if self.sink_cfg is None:
            old_d = self.d
            self.d = self.N + 1
            sink_idx = self.N

            H_new = np.zeros((self.d, self.d), dtype=complex)
            H_new[:old_d, :old_d] = self.H
            self.H = H_new

            V_new = np.zeros((self.d, self.d), dtype=complex)
            V_new[:old_d, :old_d] = self.V
            self.V = V_new

            new_jump_ops: List[Tuple[np.ndarray, float]] = []
            for op, rate in self.jump_ops:
                op_new = np.zeros((self.d, self.d), dtype=complex)
                op_new[:old_d, :old_d] = op
                new_jump_ops.append((op_new, rate))
            self.jump_ops = new_jump_ops
        else:
            sink_idx = self.sink_cfg.sink

        self.sink_cfg = SinkConfig(
            target=target_node,
            sink=sink_idx,
            threshold_type=threshold_type,
            eta_base=float(eta_base),
            eta_high=float(eta_high),
            I_th=float(I_th),
            steepness=float(steepness),
        )

    def _eta_instantaneous(self, rho: np.ndarray) -> float:
        assert self.sink_cfg is not None
        cfg = self.sink_cfg
        if cfg.threshold_type == "linear":
            return cfg.eta_base
        I_val = float(np.real(rho[cfg.target, cfg.target]))
        x = cfg.steepness * (I_val - cfg.I_th)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return cfg.eta_base + (cfg.eta_high - cfg.eta_base) * sigmoid

    def liouvillian_superoperator(self, current_rho_vec: Optional[np.ndarray] = None) -> csr_matrix:
        """Construct Liouvillian L (d^2 x d^2) as CSR sparse matrix."""
        d = self.d
        I = np.eye(d, dtype=complex)
        H_eff = self.H + self.V

        # Column-major convention:
        # vec(H rho) = (I ⊗ H) vec(rho)
        # vec(rho H) = (H^T ⊗ I) vec(rho)
        L_coh = -1j * (np.kron(I, H_eff) - np.kron(H_eff.T, I))

        L_diss = np.zeros((d * d, d * d), dtype=complex)

        for op, rate in self.jump_ops:
            if rate <= 0:
                continue
            op_dag_op = op.conj().T @ op
            term = rate * (
                np.kron(op.conj(), op)
                - 0.5 * (np.kron(I, op_dag_op) + np.kron(op_dag_op.T, I))
            )
            L_diss += term

        if self.sink_cfg is not None:
            cfg = self.sink_cfg
            J = np.zeros((d, d), dtype=complex)
            J[cfg.sink, cfg.target] = 1.0

            eta = cfg.eta_base
            if cfg.threshold_type == "threshold":
                if current_rho_vec is None:
                    raise ValueError("current_rho_vec must be provided for threshold sink")
                rho = _mat_col_major(current_rho_vec, d)
                eta = self._eta_instantaneous(rho)

            J_dag_J = J.conj().T @ J
            term = eta * (
                np.kron(J.conj(), J)
                - 0.5 * (np.kron(I, J_dag_J) + np.kron(J_dag_J.T, I))
            )
            L_diss += term

        return csr_matrix(L_coh + L_diss)

    def solve_dynamics(
        self,
        t_span: Tuple[float, float],
        rho0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate rho_dot = L(rho) rho for linear or threshold sink."""
        if rho0.shape != (self.d, self.d):
            raise ValueError(f"rho0 must have shape {(self.d, self.d)}")

        y0 = _vec_col_major(rho0)

        if self.sink_cfg is None or self.sink_cfg.threshold_type == "linear":
            L_static = self.liouvillian_superoperator(None)

            def dydt(_t: float, y: np.ndarray) -> np.ndarray:
                return L_static @ y

        else:

            def dydt(_t: float, y: np.ndarray) -> np.ndarray:
                L = self.liouvillian_superoperator(y)
                return L @ y

        res = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        if not res.success:
            raise RuntimeError(f"Dynamics integration failed: {res.message}")
        return res.t, res.y

    def sink_population(self, rho: np.ndarray) -> float:
        if self.sink_cfg is None:
            raise ValueError("Sink is not configured")
        return float(np.real(rho[self.sink_cfg.sink, self.sink_cfg.sink]))

    def get_p_success(self, rho_vec: np.ndarray) -> float:
        """Return success probability as missing trace on the hypothesis subspace.

        In the effective interpretation used in the paper, the sink is an
        external dissipative basin. If the sink degree of freedom is not
        explicitly modeled, the capture probability appears as a trace decrease:

            P_success(t) = 1 - Tr( rho_nodes(t) ).

        In this implementation we *do* include an explicit sink state |s> to
        keep the dynamics trace-preserving (required for spectral-gap analysis).
        Therefore Tr(rho_full)=1 always, and we operationalize the expression
        above as 1 - Tr(rho_nodes), which is equal to the sink population rho_ss.
        """
        d = self.d
        rho = _mat_col_major(rho_vec, d)
        rho_nodes = rho[: self.N, : self.N]
        tr_nodes = float(np.real(np.trace(rho_nodes)))
        return float(np.clip(1.0 - tr_nodes, 0.0, 1.0))

    def success_probability_trace(self, y_traj: np.ndarray) -> np.ndarray:
        """P_success(t) from a trajectory of vec(rho).

        Reported as 1 - Tr(rho_nodes(t)) (equivalently, sink population).
        """
        if self.sink_cfg is None:
            raise ValueError("Sink is not configured")
        out = np.zeros((y_traj.shape[1],), dtype=float)
        for i in range(y_traj.shape[1]):
            out[i] = self.get_p_success(y_traj[:, i])
        return out

    def expected_potential_trace(self, y_traj: np.ndarray) -> np.ndarray:
        """E[V](t) = Tr(rho_nodes Vhat) (hypothesis-node block only)."""
        d = self.d
        V_diag = np.real(np.diag(self.V)[: self.N])
        out = np.zeros((y_traj.shape[1],), dtype=float)
        for i in range(y_traj.shape[1]):
            rho = _mat_col_major(y_traj[:, i], d)
            rho_nodes = rho[: self.N, : self.N]
            out[i] = float(np.sum(np.real(np.diag(rho_nodes)) * V_diag))
        return out
