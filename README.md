# EnergyUQ: Uncertainty Quantification for Energy & Performance in HPC

EnergyUQ applies Uncertainty Quantification (UQ) and sensitivity analysis to evaluate and model how hardware configuration knobs (threads, frequency, affinity, placement, boost) affect energy consumption, execution time, and energy-delay efficiency in parallel HPC applications.

---

## Methodology Overview

### 1. Morris Screening (Method of Elementary Effects)
* **What it does**: A global sensitivity screening technique that evaluates parameter influence by traversing randomized one-at-a-time (OAT) trajectories across the configuration space. For each parameter, it estimates the overall influence ($\mu^*$) and interaction/non-linearity ($\sigma$).
* **Why it fits our case**: **Fast dimension reduction**. Modern HPC systems expose many tuning knobs (`N_THREADS`, `CLK`, `PLACES`, `BINDING`, `BOOST`). Constructing high-order surrogate models across all dimensions simultaneously can be computationally prohibitive. Morris screening rapidly separates dominant knobs from negligible ones in very few runs ($O(k)$ evaluations), allowing uninfluential parameters to be safely frozen to machine defaults before deeper analysis.

### 2. Dimension-Adaptive Stochastic Collocation (Sparse Grids)
* **What it does**: Non-intrusively constructs a polynomial surrogate model (interpolant) by evaluating the benchmark at multidimensional sparse grid collocation nodes (Smolyak / combination technique). The algorithm adaptively refines the grid by placing new evaluation points only in the parameter directions that exhibit high variance error.
* **Why it fits our case**: **High accuracy with minimal physical runs**. Because physical energy and runtime measurements on HPC hardware cannot be accelerated arbitrarily and require exclusive node access, Monte Carlo sampling is not practical. Adaptive Stochastic Collocation provides exponential convergence for smooth responses, dynamically spending execution budget only where parameter interactions are strong rather than wasting runs on uniform grids.

### 3. Sobol Sensitivity Analysis (Variance Decomposition)
* **What it does**: Decomposes the total output variance into percentage contributions attributable to individual parameters (first-order indices) and coupled parameter interactions (higher-order / total indices). These indices are computed analytically directly from the surrogate model.
* **Why it fits our case**: **Rigorous attribution of energy and performance drivers**. It reveals not only which hardware knob has the greatest individual impact, but also quantifies coupled interactions (such as how thread scalability changes under different CPU frequency governors or boost states).

### 4. Multi-Objective Pareto & Energy-Delay Analysis (EDP)
* **What it does**: Analyzes the multi-objective trade-off between energy consumption ($\mu\text{J}$) and execution time ($s$), deriving compound efficiency metrics such as the Energy-Delay Product ($\text{EDP} = \text{Energy} \times \text{Time}$) and extracting the 2D Pareto frontier.
* **Why it fits our case**: **Optimal operating point selection**. Minimizing runtime and minimizing energy are often conflicting goals. Pareto analysis isolates the non-dominated configurations, identifying optimal sweet spots between energy savings and performance loss.

---

## Hardware Knobs & Metrics

### Controlled Knobs
* **Thread Count (`N_THREADS`)**: Number of OpenMP/parallel threads.
* **Clock Frequency (`CLK`)**: CPU frequency scaling level via `cpufreq`.
* **Thread Placement (`PLACES`)**: OpenMP places, from narrower to wider (`threads`, `cores`, `sockets`).
* **Affinity Distance (`BINDING`)**: OpenMP proc bind, from full binding to no binding (`true`, `close`, `spread`, `false`).
* **Turbo Boost (`BOOST`)**: Enable or disable dynamic CPU boost.

### Quantities of Interest (QoIs)
* **`energy_uj`**: Total package energy consumed (in microjoules, measured via RAPL).
* **`time`**: Wall-clock execution time (in seconds).
* **`EDP`**: Energy-Delay Product (in $\text{J}\cdot\text{s}$).
