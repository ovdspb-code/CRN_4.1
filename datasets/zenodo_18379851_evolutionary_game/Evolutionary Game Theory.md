# Dataset: Evolutionary Game Theory Reveals Modular Specialization as a Universal Solution to Neural Trade-offs

**Author:** Oleg Dolgikh  
**Date:** January 26, 2026  
**Version:** 1.0 (Priority Claim)  
**DOI:** 10.5281/zenodo.18379851
**Related Pre-print/Hypothesis:** 
*"Thermodynamic Advantage of Transient Coherent Dynamics in Hierarchical Decision Architectures: Coherent Resonant Netting (CRN) via Open System Formalism"*;  DOI: 10.5281/zenodo.18249249

## Abstract
This dataset contains the simulation results, source code, and analysis of an evolutionary game theory model investigating neural architectural trade-offs. We prove that modular neural architectures (segregated transport and memory) emerge as evolutionarily stable strategies (ESS) when cognitive systems face conflicting computational demands (high-noise transport vs. low-noise memory).

Using replicator dynamics with 90 competing strategies across 441 environmental conditions, the data demonstrates that modular specialization yields a +6% to +31% reproductive fitness advantage over global architectures, leading to a 99.86% population takeover.

## Core Findings
1.  **Physical Trade-off:** Optimal noise for transport (κ=0.5) conflicts with memory stability (κ=0.01).
2.  **Fitness Advantage:** Modular specialization provides a significant fitness gain compared to monolithic strategies.
3.  **Evolutionary Inevitability:** Modular strategies achieve fixation (domination) in nearly all tested environments.
4.  **Robustness:** Results hold across varying metabolic costs and environmental parameters, vanishing only in the control group where the trade-off is artificially removed.

## File Inventory (Description of Data)

### 1. Simulation Code
* **`ecological_benchmark_v1.3_FULL.py`**: The main Python simulation engine. Implements the ecological environment, energy calculations, and fitness evaluation.
* **`ecological_benchmark_v1.3.py`**: Previous stable version of the benchmark logic.

### 2. Core Results (Evolution & Strategy)
* **`replicator_dynamics_example.csv`**: Time-series data showing the evolution of strategy populations over generations. Shows the takeover of modular strategies.
* **`strategy_catalog.csv`**: Full list of 90 competing strategies (Global vs. Modular) with their specific noise parameters (κ_transport, κ_memory).
* **`fitness_landscape.csv`**: Raw fitness values for every strategy across different environmental conditions (P_transport, Modular Cost).
* **`best_strategy_grid.csv`**: Summary table identifying the winning strategy and the calculated "Modular Gain" for each point in the parameter space.

### 3. Robustness & Validation
* **`robustness_consensus_grid.csv`**: Boolean grid showing where the Modular strategy wins (>50% trials) across the full parameter space. Used to generate phase diagrams.
* **`control_no_tradeoff.csv`**: Control experiment data where the physical trade-off was removed. Shows zero modular advantage (validation of the model).
* **`boundary_thresholds.csv`** & **`boundary_compare_step4_vs_consensus.csv`**: Analysis of the stability boundaries and sensitivity to the winning threshold definition.

### 4. Supporting Data
* **`combined_tradeoff.csv`**: Data showing the fundamental conflict between transport efficiency and memory retention at different noise levels.
* **`fitness_weight_sweep.csv`**: Analysis of how the weighting of tasks (Transport vs. Memory) affects the optimal architecture.
* **`transport_*.csv` & `memory_*.csv`**: Raw logs from the component-level stress tests (Stage I and Stage II).

## Methodology
The dataset was generated using an agent-based evolutionary model. 
* **Stage I (Transport):** Modeled using GKSL/Lindblad dynamics on graph structures.
* **Stage II (Memory):** Modeled using attractor dynamics in Stochastic Block Models (SBM).
*  **Evolution:** Replicator dynamics equation (ẋᵢ = xᵢ(fᵢ - f̄)) was used to simulate competition between architectures.

## Keywords
Neural Architecture, Evolutionary Game Theory, Modular Systems, Coherent Resonant Netting (CRN), Biophysics, Replicator Dynamics, Trade-offs.

## License
* **Data:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Code:** MIT License

## Contact
**Correspondence:** ovdspb@me.com  
**ORCID:** 0009-0008-0159-1718