"""
QRN R12-CROWN Large-N Stress Test (N=1000)
Pure numpy + scipy (+ matplotlib for figures).

What this script produces (reproducible with fixed seed):
- outputs/largeN_kappa_sweep.csv     : P_success(T) vs kappa for linear vs threshold sink
- outputs/largeN_energy_sweep.csv    : coarse-grained reset bottleneck sweep (throughput + energy tax ratio)
- outputs/largeN_stress_test.png     : 3-panel figure (ENAQT window + reset bottleneck + threshold readout)
- outputs/metadata_largeN.json       : config + computed summary metrics

Run:
  python run_largeN_stress_test.py --config config_largeN.json

Notes:
- This is an intentionally "portable" large-N proxy that avoids density-matrix scaling (O(N^2)).
- We use a stochastic wave-amplitude model with phase kicks implementing dephasing in ensemble
  (Haken–Strobl style), and a non-Hermitian sink drain so that P_success(t) = 1 - ||psi(t)||^2.
- The purpose is a stress-test: do ENAQT-like non-monotonicity and the sink/readout robustness survive at N=1000?

"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def watts_strogatz_sparse(N: int, k_neighbors: int, p_rewire: float, seed: int) -> csr_matrix:
    """
    Lightweight Watts–Strogatz generator (undirected) returning a CSR adjacency (with -1 on edges).
    """
    rng = np.random.default_rng(seed)
    k_neighbors = int(k_neighbors)
    if k_neighbors % 2 == 1:
        k_neighbors += 1
    half = k_neighbors // 2
    neighbors = [set() for _ in range(N)]

    # ring lattice
    for i in range(N):
        for j in range(1, half + 1):
            b = (i + j) % N
            neighbors[i].add(b)
            neighbors[b].add(i)

    # rewire forward edges
    for i in range(N):
        for j in range(1, half + 1):
            b = (i + j) % N
            if rng.random() < p_rewire:
                if b in neighbors[i]:
                    neighbors[i].remove(b)
                    neighbors[b].remove(i)

                # pick a new endpoint not equal to i and not already connected
                while True:
                    new = int(rng.integers(0, N))
                    if new != i and new not in neighbors[i]:
                        break
                neighbors[i].add(new)
                neighbors[new].add(i)

    # CSR
    rows, cols, data = [], [], []
    for i in range(N):
        for j in neighbors[i]:
            rows.append(i)
            cols.append(j)
            data.append(-1.0)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)


def shortest_path_distances_from_target(A: csr_matrix, target: int) -> np.ndarray:
    """
    BFS distances on an unweighted graph stored as CSR adjacency.
    """
    N = A.shape[0]
    indptr = A.indptr
    indices = A.indices
    dist = np.full(N, -1, dtype=int)
    dist[target] = 0
    q = deque([target])
    while q:
        v = q.popleft()
        for u in indices[indptr[v]:indptr[v + 1]]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


@dataclass
class LargeNModel:
    N: int
    gamma: float
    eta_base: float
    k_neighbors: int
    p_rewire: float
    seed: int
    V_scale: float

    def build(self) -> Tuple[csr_matrix, np.ndarray, int]:
        A = watts_strogatz_sparse(self.N, self.k_neighbors, self.p_rewire, self.seed)
        H = self.gamma * A
        target = self.N - 1
        dist = shortest_path_distances_from_target(A, target)
        maxd = dist.max() if dist.max() > 0 else 1
        V = self.V_scale * dist.astype(float) / maxd  # 0 at target, up to V_scale
        return H, V, target


def rk4_step(psi: np.ndarray, H: csr_matrix, V: np.ndarray, target: int, eta_inst: float, dt: float) -> np.ndarray:
    """
    One RK4 step for:
      dpsi/dt = -i*(H psi + V*psi) - (eta_inst/2) |t><t| psi
    """
    def f(x: np.ndarray) -> np.ndarray:
        y = -1j * (H.dot(x) + V * x)
        y[target] += -0.5 * eta_inst * x[target]
        return y

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    return psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_trajectory(
    H: csr_matrix,
    V: np.ndarray,
    target: int,
    kappa: float,
    T: float,
    dt: float,
    rng: np.random.Generator,
    eta_base: float,
    threshold: bool = False,
    I_th: float = 0.1,
    eta_high: float = 5.0,
    steepness: float = 200.0,
    record: bool = False,
) -> Dict[str, np.ndarray]:
    steps = int(T / dt)
    psi = np.zeros(H.shape[0], dtype=np.complex128)
    psi[0] = 1.0

    if record:
        times = np.linspace(0.0, T, steps + 1)
        psuccess = np.zeros(steps + 1)
        Itarget = np.zeros(steps + 1)
        eta_t = np.zeros(steps + 1)
        psuccess[0] = 0.0
        Itarget[0] = abs(psi[target]) ** 2
        eta_t[0] = eta_base

    for s in range(steps):
        if threshold:
            I = abs(psi[target]) ** 2
            x = steepness * (I - I_th)
            sig = 1.0 / (1.0 + np.exp(-x))
            eta_inst = eta_base + (eta_high - eta_base) * sig
        else:
            eta_inst = eta_base

        # deterministic (RK4)
        psi = rk4_step(psi, H, V, target, eta_inst, dt)

        # dephasing: independent phase kicks, Var(phi)=kappa*dt
        if kappa > 0:
            phases = rng.normal(0.0, math.sqrt(kappa * dt), size=psi.shape[0])
            psi *= np.exp(-1j * phases)

        if record:
            psuccess[s + 1] = 1.0 - float(np.vdot(psi, psi).real)
            Itarget[s + 1] = abs(psi[target]) ** 2
            eta_t[s + 1] = eta_inst

    out = {"psuccess_T": 1.0 - float(np.vdot(psi, psi).real)}
    if record:
        out.update({"t": times, "psuccess": psuccess, "Itarget": Itarget, "eta_t": eta_t})
    return out


def run_kappa_sweep(
    model: LargeNModel,
    kappa_grid: np.ndarray,
    T: float,
    dt: float,
    n_traj: int,
    threshold: bool,
    sink_cfg: Dict,
) -> np.ndarray:
    H, V, target = model.build()
    vals = []
    for kappa in kappa_grid:
        ps = []
        for tr in range(n_traj):
            rng = np.random.default_rng(model.seed + 10_000 + tr)
            out = run_trajectory(
                H, V, target,
                kappa=float(kappa),
                T=T,
                dt=dt,
                rng=rng,
                eta_base=model.eta_base,
                threshold=threshold,
                I_th=sink_cfg["I_th"],
                eta_high=sink_cfg["eta_high"],
                steepness=sink_cfg["steepness"],
                record=False,
            )
            ps.append(out["psuccess_T"])
        vals.append(float(np.mean(ps)))
    return np.array(vals)


def fwhm_window_decades(kappa_grid: np.ndarray, vals: np.ndarray) -> Tuple[float, float, float, float]:
    """
    FWHM defined relative to the floor: threshold = min + 0.5*(max-min)
    Returns: (W_dec, k_lo, k_hi, thr)
    """
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    thr = vmin + 0.5 * (vmax - vmin)
    mask = vals >= thr
    if not np.any(mask):
        return 0.0, float("nan"), float("nan"), thr
    k_lo = float(np.min(kappa_grid[mask]))
    k_hi = float(np.max(kappa_grid[mask]))
    W_dec = math.log10(k_hi / k_lo)
    return W_dec, k_lo, k_hi, thr


def energy_regime_sweep(cfg: Dict) -> pd.DataFrame:
    E_comm = float(cfg["E_comm"])
    P_tax = float(cfg["P_tax"])
    tau_fix = float(cfg["tau_fix"])
    tau_trans = float(cfg["tau_trans"])
    tau_reset = np.logspace(math.log10(cfg["tau_reset_min"]), math.log10(cfg["tau_reset_max"]), int(cfg["tau_reset_points"]))
    tau_cycle = tau_trans + tau_fix + tau_reset
    f_fix = 1.0 / tau_cycle
    E_tax = P_tax * tau_cycle
    tax_ratio = E_tax / E_comm
    return pd.DataFrame({"tau_reset": tau_reset, "f_fix": f_fix, "tax_ratio": tax_ratio})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_largeN.json")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    outdir = cfg_path.parent / "outputs"
    outdir.mkdir(exist_ok=True)

    model = LargeNModel(
        N=int(cfg["N"]),
        gamma=float(cfg["hamiltonian"]["gamma"]),
        eta_base=float(cfg["sink"]["eta_base"]),
        k_neighbors=int(cfg["topology"]["k_neighbors"]),
        p_rewire=float(cfg["topology"]["p_rewire"]),
        seed=int(cfg["topology"]["seed"]),
        V_scale=float(cfg["hamiltonian"]["V_scale"]),
    )

    # Sweep
    sweep = cfg["sweep"]
    kappa_grid = np.logspace(math.log10(sweep["kappa_min"]), math.log10(sweep["kappa_max"]), int(sweep["n_points"]))
    T = float(sweep["T"])
    dt = float(sweep["dt"])
    n_traj = int(sweep["n_traj"])
    sink_cfg = cfg["sink"]["threshold"]

    ps_lin = run_kappa_sweep(model, kappa_grid, T=T, dt=dt, n_traj=n_traj, threshold=False, sink_cfg=sink_cfg)
    ps_thr = run_kappa_sweep(model, kappa_grid, T=T, dt=dt, n_traj=n_traj, threshold=True, sink_cfg=sink_cfg)

    df = pd.DataFrame({"kappa": kappa_grid, "P_success_linear": ps_lin, "P_success_threshold": ps_thr})
    df.to_csv(outdir / "largeN_kappa_sweep.csv", index=False)

    # Metrics
    k_opt = float(kappa_grid[int(np.argmax(ps_lin))])
    W_dec, k_lo, k_hi, thr = fwhm_window_decades(kappa_grid, ps_lin)

    # Energy regime
    dfE = energy_regime_sweep(cfg["energy_regime"])
    dfE.to_csv(outdir / "largeN_energy_sweep.csv", index=False)

    # Trace (one representative trajectory at k_opt)
    H, V, target = model.build()
    rng = np.random.default_rng(model.seed + 999)
    tr_lin = run_trajectory(H, V, target, kappa=k_opt, T=T, dt=dt, rng=rng, eta_base=model.eta_base, threshold=False, **sink_cfg, record=True)
    rng2 = np.random.default_rng(model.seed + 999)
    tr_thr = run_trajectory(H, V, target, kappa=k_opt, T=T, dt=dt, rng=rng2, eta_base=model.eta_base, threshold=True, **sink_cfg, record=True)

    # Determine a "trigger time" marker (illustrative): eta(t) >= eta_base + 0.25*(eta_high-eta_base)
    eta_base = float(model.eta_base)
    eta_high = float(sink_cfg["eta_high"])
    trigger_level = eta_base + 0.25 * (eta_high - eta_base)
    cross = np.argmax(tr_thr["eta_t"] >= trigger_level)
    t_trigger = float(tr_thr["t"][cross])

    # Plot 3-panel figure
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.2])

    axA = fig.add_subplot(gs[0, 0])
    axA.set_xscale("log")
    axA.plot(kappa_grid, ps_lin, label="Linear sink")
    axA.plot(kappa_grid, ps_thr, label="Threshold sink")
    axA.axvline(k_opt, linestyle="--", linewidth=1)
    axA.axvspan(k_lo, k_hi, alpha=0.12)
    axA.set_xlabel(r"Dephasing rate $\kappa$ (a.u.)")
    axA.set_ylabel(r"$P_{\mathrm{success}}(T)$")
    axA.set_title(f"ENAQT-like window (N=1000, WS)\npeak at κ≈{k_opt:.2g}, FWHM≈{W_dec:.2f} decades")
    axA.grid(True, which="both", alpha=0.3)
    axA.legend(frameon=False)

    axB = fig.add_subplot(gs[0, 1])
    axB.set_xscale("log")
    axB.plot(dfE["tau_reset"], dfE["f_fix"])
    axB.set_xlabel(r"Reset time $\tau_{\mathrm{reset}}$ [s]")
    axB.set_ylabel(r"Fixation rate $f_{\mathrm{fix}}$ [Hz]")
    axB.set_title("Reset bottleneck (coarse-grained)")
    axB.grid(True, which="both", alpha=0.3)

    axB2 = axB.twinx()
    axB2.set_yscale("log")
    axB2.plot(dfE["tau_reset"], dfE["tax_ratio"], linestyle="--")
    axB2.axhline(1e-1, linestyle=":", linewidth=1)
    axB2.set_ylabel(r"Energy tax ratio $E_{\mathrm{tax}}/E_{\mathrm{comm}}$")

    axC = fig.add_subplot(gs[0, 2])
    axC.plot(tr_lin["t"], tr_lin["psuccess"], label="Linear sink")
    axC.plot(tr_thr["t"], tr_thr["psuccess"], label="Threshold sink")
    axC.axvline(t_trigger, linestyle="--", linewidth=1)
    axC.set_xlabel("t (a.u.)")
    axC.set_ylabel(r"$P_{\mathrm{success}}(t)$")
    axC.set_title(f"Threshold readout (κ={k_opt:.2g})")
    axC.grid(True, alpha=0.3)
    axC.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(outdir / "largeN_stress_test.png", dpi=300)
    plt.close(fig)

    meta = {
        "config": cfg,
        "computed": {
            "kappa_opt_linear": k_opt,
            "P_success_max_linear": float(np.max(ps_lin)),
            "P_success_min_linear": float(np.min(ps_lin)),
            "fwhm_threshold": thr,
            "fwhm_kappa_lo": k_lo,
            "fwhm_kappa_hi": k_hi,
            "fwhm_width_decades": W_dec,
            "trigger_time_marker": t_trigger,
            "trigger_level": trigger_level,
        },
    }
    (outdir / "metadata_largeN.json").write_text(json.dumps(meta, indent=2))

    print("DONE")
    print(f"- {outdir/'largeN_stress_test.png'}")
    print(f"- {outdir/'largeN_kappa_sweep.csv'}")
    print(f"- {outdir/'largeN_energy_sweep.csv'}")
    print(f"- {outdir/'metadata_largeN.json'}")


if __name__ == "__main__":
    main()
